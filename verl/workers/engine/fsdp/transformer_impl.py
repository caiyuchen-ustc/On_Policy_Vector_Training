# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from ..base import BaseEngine, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

import gc
import warnings
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.nn as nn


class TrainableTokenVectorHook(nn.Module):
    """
    只有一个可训练参数: self.steer_vector

    功能:
        在指定 transformer decoder layer 的输出 hidden_states 上加一个向量。

    支持:
        hidden_states: [batch, seq_len, hidden_size]
        hidden_states: [total_tokens, hidden_size]  # remove_padding 场景

    作用:
        hidden_states = hidden_states + steer_vector
    """

    def __init__(self, hidden_size: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.steer_vector = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def _add_vector(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.size(-1) != self.steer_vector.numel():
            raise RuntimeError(
                f"Hidden size mismatch: hidden_states.size(-1)={hidden_states.size(-1)}, "
                f"steer_vector.numel()={self.steer_vector.numel()}"
            )

        delta = self.steer_vector.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        view_shape = [1] * hidden_states.dim()
        view_shape[-1] = -1

        return hidden_states + delta.view(*view_shape)

    def hook(self, module, inputs, output):
        """
        HF decoder layer 通常返回 tuple，第一个元素是 hidden_states。
        """

        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = self._add_vector(hidden_states)
            return (hidden_states,) + output[1:]

        if torch.is_tensor(output):
            return self._add_vector(output)

        return output


def get_decoder_layers(model):
    """
    兼容常见 HuggingFace CausalLM 结构。

    LLaMA / Qwen / Mistral / Gemma:
        model.model.layers

    GPT-NeoX:
        model.gpt_neox.layers

    GPT-2:
        model.transformer.h
    """

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    raise RuntimeError(
        "Cannot find decoder layers for this model. "
        "Please adapt get_decoder_layers() for your model architecture."
    )


def get_hidden_size(model) -> int:
    config = model.config

    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)

    if hasattr(config, "n_embd"):
        return int(config.n_embd)

    if hasattr(config, "d_model"):
        return int(config.d_model)

    raise RuntimeError(
        "Cannot infer hidden size from model.config. "
        "Please adapt get_hidden_size() for your model architecture."
    )


def install_single_trainable_vector_hook(
    model,
    layer_idx: int,
    dtype: Optional[torch.dtype] = None,
):
    """
    安装唯一的 trainable vector hook。

    关键点:
        1. 这里只注册一个 nn.Parameter: steer_vector
        2. 必须通过 layer.add_module(...) 注册成子模块
        3. 否则 steer_vector 不会进入 model.named_parameters()
        4. 必须在 FSDP wrap 之前调用
    """

    layers = get_decoder_layers(model)
    num_layers = len(layers)

    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx={layer_idx}, model has {num_layers} layers.")

    layer = layers[layer_idx]
    hidden_size = get_hidden_size(model)

    hook_name = "_single_trainable_vector_hook"

    if hasattr(layer, hook_name):
        raise RuntimeError(f"Hook already installed on layer {layer_idx}")

    hook_module = TrainableTokenVectorHook(
        hidden_size=hidden_size,
        dtype=dtype,
    )

    layer.add_module(hook_name, hook_module)

    handle = layer.register_forward_hook(hook_module.hook)

    model._single_trainable_vector_hook_handle = handle
    model._single_trainable_vector_layer_idx = layer_idx

    return hook_module


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

        self.single_trainable_vector_hook = None

    def _use_single_trainable_vector(self) -> bool:
        """
        是否启用额外可训练向量。

        默认 False:
            不安装 hook
            不冻结参数
            optimizer 使用原本的 module.parameters()

        True:
            安装 hook
            冻结除 steer_vector 之外的所有参数
            optimizer 只训练这一个 steer_vector
        """

        return bool(getattr(self.model_config, "enable_trainable_token_vector", False))

    def _use_qkv_bias_only(self) -> bool:
        """
        是否只训练 attention 的 q/k/v_proj bias。

        默认 False。
        True 时:
            不安装任何 hook（这些 bias 是模型原生参数）
            冻结除 self_attn.{q,k,v}_proj.bias 之外的所有参数
            optimizer 只训练这些 bias
        与 enable_trainable_token_vector 互斥。
        """

        return bool(getattr(self.model_config, "train_q_bias_only", False))

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # breakpoint()
        # import pdb
        # pdb.set_trace()
        self._build_model_optimizer()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload model during init", logger=logger)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.optimizer)
            log_gpu_memory_usage("After offload optimizer during init", logger=logger)

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_contents=self.checkpoint_config,
        )

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size

        dp_size = self.get_data_parallel_size()

        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(dp_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
    
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)
            # breakpoint()
            module = auto_class.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            use_liger = self.model_config.use_liger

            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None)
                if fused_kernel_options is not None
                else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels

            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        # breakpoint()
        if not self.engine_config.forward_only and self._use_single_trainable_vector():
            layer_idx = getattr(
                self.model_config,
                "trainable_token_vector_layer_idx",
                16,
            )

            try:
                vector_dtype = next(module.parameters()).dtype
            except StopIteration:
                vector_dtype = torch_dtype

            hook_module = install_single_trainable_vector_hook(
                model=module,
                layer_idx=layer_idx,
                dtype=vector_dtype,
            )

            self.single_trainable_vector_hook = hook_module

            if self.rank == 0:
                print(
                    f"Installed ONE trainable vector at layer {layer_idx}: "
                    f"shape={tuple(hook_module.steer_vector.shape)}, "
                    f"dtype={hook_module.steer_vector.dtype}"
                )
        else:
            if self.rank == 0 and not self.engine_config.forward_only:
                print("Trainable token vector disabled. Keep original model construction.")

        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)

        if lora_adapter_path is not None:
            from peft import PeftModel
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")

            local_adapter_path = copy_to_local(
                lora_adapter_path,
                use_shm=self.model_config.use_shm,
            )

            module = PeftModel.from_pretrained(
                module,
                local_adapter_path,
                is_trainable=True,
            )

            peft_config = module.peft_config["default"]

            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM

        else:
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }

            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _freeze_all_except_single_vector(self, module):
        """
        只有 enable_trainable_token_vector=True 时才调用。

        效果:
            冻结所有参数
            只保留唯一一个 steer_vector 可训练
        """

        for _, p in module.named_parameters():
            p.requires_grad_(False)

        vector_params = []

        for name, p in module.named_parameters():
            if "_single_trainable_vector_hook.steer_vector" in name:
                p.requires_grad_(True)
                vector_params.append((name, p))

        if len(vector_params) != 1:
            names = [name for name, _ in vector_params]
            raise RuntimeError(
                f"Expected exactly ONE steer_vector, but found {len(vector_params)}: {names}"
            )

        if self.rank == 0:
            name, p = vector_params[0]
            print("Only this ONE parameter will be trained:")
            print(f"  {name}, shape={tuple(p.shape)}, dtype={p.dtype}, numel={p.numel()}")

        return vector_params[0][1]

    def _is_qkv_bias_param(self, name: str) -> bool:
        if not name.endswith("self_attn.q_proj.bias"):
            return False
        lo = getattr(self.model_config, "q_bias_layer_start", None)
        hi = getattr(self.model_config, "q_bias_layer_end", None)
        if lo is None and hi is None:
            return True
        import re as _re

        m = _re.search(r"\.layers\.(\d+)\.", name)
        if m is None:
            return False
        idx = int(m.group(1))
        return (0 if lo is None else int(lo)) <= idx <= (idx if hi is None else int(hi))

    def _freeze_all_except_qkv_bias(self, module):
        """
        冻结所有参数，只保留 attention 的 q_proj bias 可训练（可选层范围）。
        这些 bias 是模型原生参数（Qwen2 q_proj bias=True），
        vLLM 全量权重同步会自动带上，无需自定义 patch。
        """

        for _, p in module.named_parameters():
            p.requires_grad_(False)

        bias_params = []
        for name, p in module.named_parameters():
            if self._is_qkv_bias_param(name):
                p.requires_grad_(True)
                bias_params.append((name, p))

        if len(bias_params) == 0:
            raise RuntimeError(
                "train_qkv_bias_only=True but found no self_attn.q_proj.bias params. "
                "This model likely has attention bias disabled."
            )

        if self.rank == 0:
            total_numel = sum(p.numel() for _, p in bias_params)
            print(f"Only q_proj bias will be trained: count={len(bias_params)}, total_numel={total_numel}")
            for name, p in bias_params:
                print(f"  {name}, shape={tuple(p.shape)}, dtype={p.dtype}, numel={p.numel()}")

        return [p for _, p in bias_params]

    def _build_fsdp_module(self, module):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision

        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if self.engine_config.strategy == "fsdp":
            cpu_offload = None

            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )

        elif self.engine_config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )

            offload_policy = None

            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }

            full_state = module.state_dict()

            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)

        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(
                module,
                self.engine_config.strategy,
                enable_gradient_checkpointing,
            )

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )

        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        if self._use_qkv_bias_only():
            bias_params = [p for name, p in module.named_parameters() if self._is_qkv_bias_param(name)]

            if len(bias_params) == 0:
                raise RuntimeError("Optimizer expected q_proj bias params but found none.")

            for p in bias_params:
                if not p.requires_grad:
                    raise RuntimeError("q_proj bias exists but requires_grad=False")

            if self.rank == 0:
                total_numel = sum(p.numel() for p in bias_params)
                print(f"Optimizer will update q_proj bias: count={len(bias_params)}, total_numel={total_numel}")

            return build_optimizer(bias_params, self.optimizer_config)

        if self._use_single_trainable_vector():
            vector_params = []

            for name, p in module.named_parameters():
                if "_single_trainable_vector_hook.steer_vector" in name:
                    vector_params.append((name, p))

            if len(vector_params) != 1:
                names = [name for name, _ in vector_params]
                raise RuntimeError(
                    f"Optimizer expected exactly ONE steer_vector, "
                    f"but found {len(vector_params)}: {names}"
                )

            name, vector_param = vector_params[0]

            if not vector_param.requires_grad:
                raise RuntimeError(f"{name} exists but requires_grad=False")

            if self.rank == 0:
                print("Optimizer will update exactly ONE parameter:")
                print(
                    f"  {name}, shape={tuple(vector_param.shape)}, "
                    f"numel={vector_param.numel()}, dtype={vector_param.dtype}"
                )

            optimizer = build_optimizer(
                [vector_param],
                self.optimizer_config,
            )

            return optimizer

        if self.rank == 0:
            print("Trainable token vector disabled. Optimizer uses original module.parameters().")

        optimizer = build_optimizer(module.parameters(), self.optimizer_config)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles

        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
            )

        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )

        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return lr_scheduler

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        module = self._build_module()

        if self._is_lora:
            module = self._build_lora_module(module)

        if not self.engine_config.forward_only and self._use_single_trainable_vector():
            self._freeze_all_except_single_vector(module)

        if not self.engine_config.forward_only and self._use_qkv_bias_only():
            self._freeze_all_except_qkv_bias(module)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(module)

            if not self.engine_config.forward_only:
                if self._use_single_trainable_vector() or self._use_qkv_bias_only():
                    print("Trainable parameters before FSDP:")
                    for name, p in module.named_parameters():
                        if p.requires_grad:
                            print(f"  {name}, shape={tuple(p.shape)}, dtype={p.dtype}, numel={p.numel()}")
                else:
                    print("Trainable token vector disabled. Keep original trainable parameters before FSDP.")

        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        log_gpu_memory_usage("Before FSDP", logger=None)

        if not self.engine_config.forward_only and (self._use_single_trainable_vector() or self._use_qkv_bias_only()):
            if self.engine_config.strategy == "fsdp" and not self.engine_config.use_orig_params:
                raise RuntimeError(
                    "enable_trainable_token_vector=True or train_qkv_bias_only=True requires "
                    "use_orig_params=True under FSDP, because only a subset of parameters is trainable "
                    "while all other parameters are frozen."
                )

        module = self._build_fsdp_module(module)

        log_gpu_memory_usage("After FSDP", logger=None)

        if self.rank == 0 and not self.engine_config.forward_only:
            if self._use_single_trainable_vector() or self._use_qkv_bias_only():
                print("Trainable parameters after FSDP:")
                for name, p in module.named_parameters():
                    if p.requires_grad:
                        print(f"  {name}, shape={tuple(p.shape)}, dtype={p.dtype}, numel={p.numel()}")
            else:
                print("Trainable token vector disabled. Keep original trainable parameters after FSDP.")

        if not self.engine_config.forward_only:
            optimizer = self._build_optimizer(module)
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self)

    def eval_mode(self):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def forward_backward_batch(
        self,
        data: TensorDict,
        loss_function: Callable,
        forward_only=False,
    ) -> list[TensorDict]:
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())

        torch.distributed.all_reduce(
            batch_num_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_data_parallel_group(),
        )

        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(
                    micro_batch,
                    loss_function=loss_function,
                    forward_only=forward_only,
                )

                if not forward_only:
                    loss.backward()

            output_lst.append(meta_info)

        return postprocess_batch_func(
            output_lst=output_lst,
            indices=indices,
            data=data,
        )

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)

        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.module.parameters(),
                max_norm=self.optimizer_config.clip_grad,
            )

        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(),
                max_norm=self.optimizer_config.clip_grad,
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        return grad_norm.item()

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        if self.engine_config.forward_only:
            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")

        if device == device_name:
            if not self.engine_config.param_offload:
                if model:
                    load_fsdp_model_to_gpu(self.module)

                if optimizer and self.optimizer is not None:
                    load_fsdp_optimizer(self.optimizer, device)

            gc.collect()

        elif device == "cpu":
            if not self.engine_config.param_offload:
                if model:
                    offload_fsdp_model_to_cpu(self.module)

                if optimizer and self.optimizer is not None:
                    offload_fsdp_optimizer(self.optimizer)

        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        torch.distributed.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: int = True,
        **kwargs,
    ) -> None:
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        torch.distributed.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(
        self,
        layered_summon=False,
        base_sync_done=False,
    ):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)

        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)

            params = collect_lora_params(
                module=self.module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
            )

            if not base_sync_done:
                params = {
                    replace_lora_wrapper(k, peft_config): v
                    for k, v in params.items()
                }

        else:
            params = self.module.state_dict()

        params = convert_weight_keys(
            params,
            getattr(self.module, "_fsdp_wrapped_module", self.module),
        )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params

        else:
            device = get_device_id()

            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        return per_tensor_param


class EngineEvalModeCtx:
    def __init__(self, engine: FSDPEngine):
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.module)

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.module)
        self.engine.mode = None


class EngineTrainModeCtx:
    def __init__(self, engine: FSDPEngine):
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "train"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.engine.optimizer, device_id=get_torch_device().current_device())

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()

        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.engine.optimizer)
        self.engine.mode = None


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]

        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:
                    # vlm model's inputs will be sliced after embedding
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]

            if use_fused_kernels:
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

            # gather log_prob if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits
                logits.div_(temperature)

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss,
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        if use_remove_padding:
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape

            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = output[2].squeeze(0).unsqueeze(-1)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz, 1)
                # critic model arch is like Qwen3ForTokenClassfication and num_labels=1
                # so we squeeze the last dimension here to get the value for each token
                values_rmpad = values_rmpad.squeeze(-1)

            # gather output if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values = output[2]
            else:
                values = output.logits

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}
