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
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()
_SINGLE_VECTOR_HOOK_KEY = "_single_trainable_vector_hook."
_MULTI_VECTOR_HOOK_KEY = "_multi_trainable_vector_hook."
_STEER_VECTOR_PARAM_SUFFIXES = (
    "_single_trainable_vector_hook.steer_vector",
    "_single_trainable_vector_hook.alpha",
    "_multi_trainable_vector_hook.basis_vectors",
    "_multi_trainable_vector_hook.alpha",
)


def _is_trainable_vector_param_name(name: str) -> bool:
    return any(suffix in name for suffix in _STEER_VECTOR_PARAM_SUFFIXES)


def _is_q_bias_param_name(name: str, layer_start: Optional[int] = None, layer_end: Optional[int] = None) -> bool:
    """attention 的 q_proj bias（模型原生参数），用于 train_q_bias_only 模式。

    仅 q_proj bias，冻结 k/v。若给定 [layer_start, layer_end]（闭区间），
    只匹配该层范围内的 q_proj bias；否则匹配所有层。
    """
    if not name.endswith("self_attn.q_proj.bias"):
        return False
    if layer_start is None and layer_end is None:
        return True
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    idx = int(m.group(1))
    lo = 0 if layer_start is None else int(layer_start)
    hi = idx if layer_end is None else int(layer_end)
    return lo <= idx <= hi


def _is_single_layer_param_name(name: str, layer_idx: int) -> bool:
    """匹配指定 decoder layer 的所有参数，用于 train_single_layer_only 模式。"""
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    return int(m.group(1)) == int(layer_idx)


def _is_single_mlp_param_name(name: str, layer_idx: int) -> bool:
    """匹配指定 decoder layer 的 MLP down_proj 参数，用于 train_single_mlp_only 模式。仅 down_proj，冻结 gate/up。"""
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    return int(m.group(1)) == int(layer_idx) and ".mlp.down_proj." in name


def _get_core_decoder_layers(module: nn.Module):
    wrapped = getattr(module, "_fsdp_wrapped_module", module)
    core_module = getattr(wrapped, "module", wrapped)

    if hasattr(core_module, "model") and hasattr(core_module.model, "layers"):
        return core_module.model.layers
    if hasattr(core_module, "gpt_neox") and hasattr(core_module.gpt_neox, "layers"):
        return core_module.gpt_neox.layers
    if hasattr(core_module, "transformer") and hasattr(core_module.transformer, "h"):
        return core_module.transformer.h
    return None


def _get_trainable_vector_hook_modules(module: nn.Module) -> list[nn.Module]:
    layers = _get_core_decoder_layers(module)
    if layers is None:
        return []

    hook_modules: list[nn.Module] = []
    for layer in layers:
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is not None:
                hook_modules.append(hook)
    return hook_modules


_TRAINABLE_VECTOR_ALPHA_CACHE: dict[int, torch.Tensor] = {}
_TRAINABLE_VECTOR_ALPHA_GLOO_GROUP = None


def _get_trainable_alpha_sync_group():
    global _TRAINABLE_VECTOR_ALPHA_GLOO_GROUP
    if _TRAINABLE_VECTOR_ALPHA_GLOO_GROUP is None and dist.is_initialized():
        _TRAINABLE_VECTOR_ALPHA_GLOO_GROUP = dist.new_group(backend="gloo")
    return _TRAINABLE_VECTOR_ALPHA_GLOO_GROUP


def _cache_trainable_alpha(alpha_param: torch.Tensor) -> None:
    alpha_view = alpha_param.full_tensor() if hasattr(alpha_param, "full_tensor") else alpha_param
    alpha_flat = alpha_view.detach().reshape(-1)
    if alpha_flat.numel() == 0:
        return
    _TRAINABLE_VECTOR_ALPHA_CACHE[id(alpha_param)] = alpha_flat[0].cpu()


def _cache_trainable_alpha_from_scalar(alpha_param: torch.Tensor, alpha_scalar: torch.Tensor | float) -> None:
    if torch.is_tensor(alpha_scalar):
        alpha_scalar = alpha_scalar.detach().reshape(()).cpu()
    else:
        alpha_scalar = torch.tensor(float(alpha_scalar))
    _TRAINABLE_VECTOR_ALPHA_CACHE[id(alpha_param)] = alpha_scalar


def _sync_trainable_alpha_via_gloo(alpha_param: torch.Tensor) -> torch.Tensor:
    local_flat = alpha_param.detach().reshape(-1)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        if local_flat.numel() > 0:
            return local_flat[0]
        raise RuntimeError("alpha shard is empty and distributed sync is unavailable")

    group = _get_trainable_alpha_sync_group()
    local_len = torch.tensor([local_flat.numel()], dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_lens, local_len, group=group)
    lengths = [int(item.item()) for item in gathered_lens]

    owner_rank = next((rank for rank, length in enumerate(lengths) if length > 0), None)
    if owner_rank is None:
        raise RuntimeError("failed to locate owning rank for learnable alpha")

    alpha_cpu = torch.zeros(1, dtype=torch.float32)
    if dist.get_rank() == owner_rank and local_flat.numel() > 0:
        alpha_cpu[0] = local_flat[0].float().cpu()
    dist.broadcast(alpha_cpu, src=owner_rank, group=group)
    return alpha_cpu[0].to(device=alpha_param.device, dtype=alpha_param.dtype)


def _materialize_trainable_alpha(alpha_param: torch.Tensor, *, allow_gloo_sync: bool = False) -> torch.Tensor:
    if hasattr(alpha_param, "full_tensor"):
        full_alpha = alpha_param.full_tensor().detach()
        if full_alpha.numel() > 0:
            return full_alpha.reshape(-1)[0]

    if allow_gloo_sync:
        alpha_value = _sync_trainable_alpha_via_gloo(alpha_param)
        _cache_trainable_alpha_from_scalar(alpha_param, alpha_value)
        return alpha_value

    local_flat = alpha_param.detach().reshape(-1)
    if local_flat.numel() > 0:
        return local_flat[0]

    cached_alpha = _TRAINABLE_VECTOR_ALPHA_CACHE.get(id(alpha_param))
    if cached_alpha is not None:
        return cached_alpha.to(device=alpha_param.device, dtype=alpha_param.dtype)

    raise RuntimeError("alpha shard is empty and no local cache is available")


def _sync_trainable_vector_state_across_ranks(module: nn.Module) -> None:
    """Synchronize FSDP-ignored trainable-vector hook state across ranks.

    Trainable token vector hooks are intentionally passed through FSDP
    ``ignored_states`` so they stay as ordinary parameters and can be optimized
    directly. The downside is that FSDP will not synchronize their initial
    values. If different ranks keep different local hook tensors, rollout/vLLM
    will patch different steer vectors and later distributed collectives can
    deadlock because some ranks are still generating while others already moved
    on to timing reductions.
    """
    layers = _get_core_decoder_layers(module)
    if layers is None or not dist.is_initialized() or dist.get_world_size() == 1:
        return

    group = _get_trainable_alpha_sync_group()
    rank = dist.get_rank()

    for layer in layers:
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is None:
                continue

            vector_name = "basis_vectors" if hasattr(hook, "basis_vectors") else "steer_vector"
            vector_param = getattr(hook, vector_name, None)
            if vector_param is not None:
                if rank == 0:
                    vector_payload = vector_param.detach().float().cpu()
                else:
                    vector_payload = torch.empty_like(vector_param.detach().float().cpu())
                dist.broadcast(vector_payload, src=0, group=group)
                vector_param.data.copy_(vector_payload.to(device=vector_param.device, dtype=vector_param.dtype))
                if hasattr(hook, "_cached_sample_coefficients"):
                    hook._cached_sample_coefficients = None

            alpha_param = getattr(hook, "alpha", None)
            if alpha_param is not None:
                if rank == 0:
                    alpha_payload = _materialize_trainable_alpha(alpha_param).detach().reshape(1).float().cpu()
                else:
                    alpha_payload = torch.zeros(1, dtype=torch.float32)
                dist.broadcast(alpha_payload, src=0, group=group)
                alpha_param.data.copy_(alpha_payload.to(device=alpha_param.device, dtype=alpha_param.dtype))
                _cache_trainable_alpha_from_scalar(alpha_param, alpha_payload[0])
            break



def _collect_trainable_alpha_by_layer(module: nn.Module) -> dict[int, torch.Tensor]:
    layers = _get_core_decoder_layers(module)
    if layers is None:
        return {}

    alpha_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, layer in enumerate(layers):
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is None:
                continue

            alpha_param = getattr(hook, "alpha", None)
            if alpha_param is None:
                ref = getattr(hook, "steer_vector", None)
                if ref is None:
                    ref = getattr(hook, "basis_vectors", None)
                alpha_by_layer[layer_idx] = torch.tensor(
                    float(getattr(hook, "vector_scale", 1.0)),
                    dtype=ref.dtype if ref is not None else torch.float32,
                    device=ref.device if ref is not None else None,
                ).cpu()
                break

            alpha_by_layer[layer_idx] = _materialize_trainable_alpha(alpha_param, allow_gloo_sync=True).reshape(()).cpu()
            break

    return alpha_by_layer


def _broadcast_vector_tensor_from_rank0(vector_tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast a (already gathered, CPU) steer-vector tensor from rank 0 to all ranks.

    Trainable-vector hooks are passed to FSDP via ``ignored_states`` so FSDP never
    synchronizes them, and on non-rank-0 the meta-tensor ``to_empty`` materialization
    leaves them as uninitialized garbage. If each rank then ships its own local tensor
    to vLLM, ranks steer with different vectors, generate different-length sequences,
    and the next collective (e.g. ALLGATHER) deadlocks. Broadcasting here guarantees
    every rank feeds vLLM a byte-identical vector. Mirrors the gloo-based alpha sync.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return vector_tensor

    group = _get_trainable_alpha_sync_group()
    payload = vector_tensor.detach().float().cpu().contiguous()
    dist.broadcast(payload, src=0, group=group)
    return payload.to(dtype=vector_tensor.dtype)


def _collect_trainable_vector_payload_by_layer(module: nn.Module) -> Optional[dict[int, dict[str, Any]]]:
    layers = _get_core_decoder_layers(module)
    if layers is None:
        return None

    payload_by_layer: dict[int, dict[str, Any]] = {}
    for layer_idx, layer in enumerate(layers):
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is None:
                continue

            is_multi = bool(getattr(hook, "num_vectors", 1) > 1 or hook_name == "_multi_trainable_vector_hook")
            vector = getattr(hook, "basis_vectors" if is_multi else "steer_vector", None)
            if vector is None:
                continue

            vector_tensor = vector.detach()
            if hasattr(vector_tensor, "full_tensor"):
                vector_tensor = vector_tensor.full_tensor()

            # The steer-vector params are FSDP-ignored and never synchronized, so
            # non-rank-0 copies may be uninitialized. Broadcast rank 0's tensor to
            # all ranks so every rank feeds vLLM the same vector (prevents divergent
            # generation lengths and the resulting NCCL collective deadlock).
            vector_tensor = _broadcast_vector_tensor_from_rank0(vector_tensor.cpu())

            payload: dict[str, Any] = {
                "vector": vector_tensor.cpu(),
                "is_multi": is_multi,
            }

            alpha_param = getattr(hook, "alpha", None)
            if alpha_param is not None:
                payload["alpha"] = _materialize_trainable_alpha(alpha_param, allow_gloo_sync=True).reshape(()).cpu()

            payload_by_layer[layer_idx] = payload
            break

    return payload_by_layer or None


def _filter_rollout_incompatible_params(
    params: dict[str, torch.Tensor],
    *,
    enable_trainable_token_vector: bool,
) -> dict[str, torch.Tensor]:
    """Drop trainer-only parameters that rollout models do not contain.

    vLLM / HF rollout side keeps the original backbone parameter names.
    When single-trainable-vector hook is enabled, FSDP state_dict includes
    keys like `layers.N._single_trainable_vector_hook.steer_vector`, which do
    not exist in rollout model param dict and trigger KeyError during load.
    """
    if not enable_trainable_token_vector:
        return params

    filtered: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    incompatible_hook_keys = (_SINGLE_VECTOR_HOOK_KEY, _MULTI_VECTOR_HOOK_KEY)
    for name, tensor in params.items():
        if any(hook_key in name for hook_key in incompatible_hook_keys):
            dropped.append(name)
            continue
        filtered[name] = tensor

    if dropped:
        preview = dropped[:3]
        logger.warning(
            "Filtered %d rollout-incompatible parameters (single vector hook), sample=%s",
            len(dropped),
            preview,
        )

    return filtered


def _infer_layer_idx_from_single_vector_key(param_name: str) -> Optional[int]:
    parts = param_name.split(".")
    for idx, token in enumerate(parts[:-1]):
        if token != "layers" or idx + 1 >= len(parts):
            continue
        try:
            return int(parts[idx + 1])
        except ValueError:
            continue
    return None


def _extract_single_vector_payload(
    params: dict[str, torch.Tensor],
    *,
    enable_trainable_token_vector: bool,
    fallback_layer_idx: Optional[int],
) -> tuple[Optional[dict[int, torch.Tensor]], Optional[int], dict[int, bool]]:
    if not enable_trainable_token_vector:
        return None, None, {}

    single_matches = [
        (name, tensor)
        for name, tensor in params.items()
        if name.endswith("_single_trainable_vector_hook.steer_vector")
    ]
    multi_matches = [
        (name, tensor)
        for name, tensor in params.items()
        if name.endswith("_multi_trainable_vector_hook.basis_vectors")
    ]
    matches = single_matches + multi_matches
    if not matches:
        return None, fallback_layer_idx, {}

    vector_by_layer: dict[int, torch.Tensor] = {}
    is_multi_by_layer: dict[int, bool] = {}
    for name, tensor in single_matches:
        layer_idx = _infer_layer_idx_from_single_vector_key(name)
        if layer_idx is None:
            if fallback_layer_idx is None:
                raise RuntimeError(f"Cannot infer layer idx from parameter name: {name}")
            layer_idx = fallback_layer_idx
        if layer_idx in vector_by_layer:
            raise RuntimeError(f"Duplicated steer_vector for layer {layer_idx}: {name}")
        vector_by_layer[layer_idx] = tensor
        is_multi_by_layer[layer_idx] = False

    for name, tensor in multi_matches:
        layer_idx = _infer_layer_idx_from_single_vector_key(name)
        if layer_idx is None:
            if fallback_layer_idx is None:
                raise RuntimeError(f"Cannot infer layer idx from parameter name: {name}")
            layer_idx = fallback_layer_idx
        if layer_idx in vector_by_layer:
            raise RuntimeError(f"Layer {layer_idx} has both single and multi steer vectors")
        vector_by_layer[layer_idx] = tensor
        is_multi_by_layer[layer_idx] = True

    return vector_by_layer, fallback_layer_idx, is_multi_by_layer


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


class TrainableTokenVectorHook(nn.Module):
    """A trainable vector or vector basis added to hidden states via forward hook."""

    def __init__(
        self,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        num_vectors: int = 1,
        sampling_method: str = "hypersphere",
        vector_scale: float = 1.0,
        learnable_alpha: bool = False,
        alpha_init: float = 1.0,
        curriculum: str = "none",
        warmup_steps: int = 0,
        warmup_end_step: Optional[int] = None,
        secondary_freeze_steps: int = 0,
        secondary_end_step: Optional[int] = None,
        primary_scale: float = 1.0,
        secondary_scale: float = 1.0,
        freeze_primary_after_warmup: bool = False,
    ):
        super().__init__()
        if num_vectors < 1:
            raise ValueError(f"num_vectors must be >= 1, got {num_vectors}")
        if sampling_method not in {"hypersphere", "interpolation"}:
            raise ValueError(
                "sampling_method must be 'hypersphere' or 'interpolation', "
                f"got {sampling_method!r}"
            )
        if vector_scale <= 0:
            raise ValueError(f"vector_scale must be > 0, got {vector_scale}")
        if curriculum not in {"none", "warmup_expand"}:
            raise ValueError(f"curriculum must be 'none' or 'warmup_expand', got {curriculum!r}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_end_step is not None and warmup_end_step < 0:
            raise ValueError(f"warmup_end_step must be >= 0 when set, got {warmup_end_step}")
        if secondary_freeze_steps < 0:
            raise ValueError(f"secondary_freeze_steps must be >= 0, got {secondary_freeze_steps}")
        if secondary_end_step is not None and secondary_end_step < 0:
            raise ValueError(f"secondary_end_step must be >= 0 when set, got {secondary_end_step}")
        if warmup_end_step is not None and secondary_end_step is not None and secondary_end_step < warmup_end_step:
            raise ValueError(
                f"secondary_end_step must be >= warmup_end_step, got {secondary_end_step} < {warmup_end_step}"
            )
        if primary_scale <= 0:
            raise ValueError(f"primary_scale must be > 0, got {primary_scale}")
        if secondary_scale <= 0:
            raise ValueError(f"secondary_scale must be > 0, got {secondary_scale}")

        self.num_vectors = int(num_vectors)
        self.sampling_method = sampling_method
        self.vector_scale = float(vector_scale)
        self.learnable_alpha = bool(learnable_alpha)
        self.raw_single_vector = bool(self.num_vectors == 1 and not self.learnable_alpha)
        self.curriculum = curriculum
        self.warmup_steps = int(warmup_steps)
        self.warmup_end_step = None if warmup_end_step is None else int(warmup_end_step)
        self.secondary_freeze_steps = int(secondary_freeze_steps)
        self.secondary_end_step = None if secondary_end_step is None else int(secondary_end_step)
        self.primary_scale = float(primary_scale)
        self.secondary_scale = float(secondary_scale)
        self.freeze_primary_after_warmup = bool(freeze_primary_after_warmup)
        self._current_global_step = 0
        self._orthogonalized_after_warmup = False
        self._restart_notice_printed = False
        self._response_token_mask: Optional[torch.Tensor] = None
        self._cached_sample_coefficients: Optional[torch.Tensor] = None

        if self.learnable_alpha:
            alpha_dtype = dtype if dtype is not None else torch.float32
            self.alpha = nn.Parameter(torch.tensor([float(alpha_init)], dtype=alpha_dtype))
            _cache_trainable_alpha(self.alpha)
        else:
            self.register_parameter("alpha", None)

        if self.num_vectors == 1:
            if self.raw_single_vector:
                self.steer_vector = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
            else:
                self.steer_vector = nn.Parameter(self._sample_unit_vector(hidden_size, dtype=dtype))
        else:
            self.basis_vectors = nn.Parameter(
                self._normalize_basis_rows(torch.randn(self.num_vectors, hidden_size, dtype=dtype))
            )
            if self.curriculum != "warmup_expand":
                self.orthogonalize_basis()

    def _sample_unit_vector(
        self,
        hidden_size: int,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        vec = torch.randn(hidden_size, dtype=dtype, device=device)
        norm = torch.linalg.vector_norm(vec.float()).clamp_min(torch.finfo(torch.float32).eps)
        return vec / norm.to(dtype=vec.dtype)

    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector.float())
        eps = torch.finfo(torch.float32).eps
        if torch.isfinite(norm) and norm > eps:
            return vector / norm.clamp_min(eps).to(dtype=vector.dtype)

        fallback = torch.zeros_like(vector)
        if fallback.numel() > 0:
            fallback.reshape(-1)[0] = 1
        return fallback

    def _normalize_basis_rows(self, basis: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(basis)
        for idx in range(basis.size(0)):
            normalized[idx] = self._normalize_vector(basis[idx])
        return normalized

    def get_effective_alpha(self) -> torch.Tensor:
        if self.alpha is None:
            ref = self.steer_vector if self.num_vectors == 1 else self.basis_vectors
            return torch.tensor(self.vector_scale, dtype=ref.dtype, device=ref.device)
        alpha_value = _materialize_trainable_alpha(self.alpha)
        _cache_trainable_alpha_from_scalar(self.alpha, alpha_value)
        return alpha_value.to(device=self.alpha.device, dtype=self.alpha.dtype)

    def set_current_global_step(self, global_step: Optional[int]) -> None:
        if global_step is None:
            global_step = 0
        self._current_global_step = max(int(global_step), 0)

    def set_response_token_mask(self, response_token_mask: Optional[torch.Tensor]) -> None:
        self._response_token_mask = response_token_mask

    def clear_response_token_mask(self) -> None:
        self._response_token_mask = None

    def _effective_warmup_end_step(self) -> int:
        if self.warmup_end_step is not None:
            return self.warmup_end_step
        return self.warmup_steps

    def _effective_secondary_end_step(self) -> int:
        if self.secondary_end_step is not None:
            return self.secondary_end_step
        return self._effective_warmup_end_step() + self.secondary_freeze_steps

    def _in_warmup_phase(self) -> bool:
        return (
            self.curriculum == "warmup_expand"
            and self.num_vectors > 1
            and self._current_global_step <= self._effective_warmup_end_step()
        )

    def _warmup_completed(self) -> bool:
        return (
            self.curriculum == "warmup_expand"
            and self.num_vectors > 1
            and self._current_global_step > self._effective_warmup_end_step()
        )

    def _in_second_warmup_phase(self) -> bool:
        if not self._warmup_completed():
            return False
        warmup_end_step = self._effective_warmup_end_step()
        secondary_end_step = self._effective_secondary_end_step()
        if secondary_end_step <= warmup_end_step:
            return False
        return self._current_global_step <= secondary_end_step

    def maybe_activate_expanded_basis(self) -> None:
        if not self._warmup_completed() or self._orthogonalized_after_warmup:
            return
        self.orthogonalize_basis()
        self._orthogonalized_after_warmup = True
        warmup_end_step = self._effective_warmup_end_step()
        secondary_end_step = self._effective_secondary_end_step()
        print(
            f"[TrainableTokenVectorHook] curriculum activated at global_step={self._current_global_step}: "
            f"uniform_mix_end_step={warmup_end_step}, second_phase_end_step={secondary_end_step}, "
            f"primary_scale={self.primary_scale}, secondary_scale={self.secondary_scale}, "
            f"freeze_primary_after_warmup={self.freeze_primary_after_warmup}",
            flush=True,
        )

    def _should_restart_collapsed_basis(self) -> bool:
        return self.curriculum == "warmup_expand" and self.num_vectors > 1 and self._warmup_completed()

    def _sample_orthogonal_restart_vector(
        self,
        orthogonal_basis: torch.Tensor,
        idx: int,
        *,
        eps: float,
        attempts: int = 8,
    ) -> Optional[torch.Tensor]:
        for _ in range(attempts):
            vec = torch.randn_like(orthogonal_basis[idx])
            for prev_idx in range(idx):
                prev = orthogonal_basis[prev_idx]
                prev_norm_sq = torch.dot(prev, prev)
                if not torch.isfinite(prev_norm_sq) or prev_norm_sq <= eps:
                    continue
                vec = vec - (torch.dot(vec, prev) / prev_norm_sq) * prev

            vec_norm = torch.linalg.vector_norm(vec)
            if torch.isfinite(vec_norm) and vec_norm > eps:
                return vec / vec_norm
        return None

    def orthogonalize_basis(self) -> None:
        if self.num_vectors <= 1:
            return
        restarted_indices: list[int] = []
        with torch.no_grad():
            basis = self.basis_vectors.data.float()

            if self._in_warmup_phase():
                self.basis_vectors.data.copy_(self._normalize_basis_rows(basis).to(dtype=self.basis_vectors.dtype))
                return

            orthogonal_basis = torch.zeros_like(basis)
            eps = 1e-8
            restart_enabled = self._should_restart_collapsed_basis()

            for idx in range(basis.size(0)):
                vec = basis[idx].clone()
                original_norm = torch.linalg.vector_norm(vec)
                for prev_idx in range(idx):
                    prev = orthogonal_basis[prev_idx]
                    prev_norm_sq = torch.dot(prev, prev)
                    if not torch.isfinite(prev_norm_sq) or prev_norm_sq <= eps:
                        continue
                    vec = vec - (torch.dot(vec, prev) / prev_norm_sq) * prev

                vec_norm = torch.linalg.vector_norm(vec)
                tol = max(eps, float(original_norm.item()) * 1e-6)
                if torch.isfinite(vec_norm) and vec_norm > tol:
                    orthogonal_basis[idx] = vec / vec_norm
                    continue

                if not restart_enabled:
                    continue

                restart_vec = self._sample_orthogonal_restart_vector(orthogonal_basis, idx, eps=eps)
                if restart_vec is None:
                    continue
                orthogonal_basis[idx] = restart_vec
                restarted_indices.append(idx)

            self.basis_vectors.data.copy_(orthogonal_basis.to(dtype=self.basis_vectors.dtype))

        if restarted_indices and not self._restart_notice_printed:
            print(
                f"[TrainableTokenVectorHook] restarted collapsed basis vectors at global_step={self._current_global_step}: "
                f"indices={restarted_indices}",
                flush=True,
            )
            self._restart_notice_printed = True

    def _sample_simplex_coefficients(
        self,
        basis_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> torch.Tensor:
        coeffs = torch.rand(basis_size, device=device, dtype=dtype) + eps
        coeff_sum = coeffs.sum()
        if not torch.isfinite(coeff_sum) or coeff_sum <= eps:
            coeffs = torch.ones_like(coeffs)
            coeff_sum = coeffs.sum()
        return coeffs / coeff_sum.clamp_min(eps)

    def _sample_positive_hypersphere_coefficients(
        self,
        basis_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> torch.Tensor:
        coeffs = torch.rand(basis_size, device=device, dtype=dtype) + eps
        coeff_norm = torch.linalg.vector_norm(coeffs)
        if not torch.isfinite(coeff_norm) or coeff_norm <= eps:
            coeffs = torch.ones_like(coeffs)
            coeff_norm = torch.linalg.vector_norm(coeffs)
        return coeffs / coeff_norm.clamp_min(eps)

    def _renormalize_coefficients(self, coeffs: torch.Tensor, *, eps: float) -> torch.Tensor:
        if self.sampling_method == "interpolation":
            coeff_sum = coeffs.sum()
            if not torch.isfinite(coeff_sum) or coeff_sum <= eps:
                coeffs = torch.ones_like(coeffs)
                coeff_sum = coeffs.sum()
            return coeffs / coeff_sum.clamp_min(eps)

        coeff_norm = torch.linalg.vector_norm(coeffs)
        if not torch.isfinite(coeff_norm) or coeff_norm <= eps:
            coeffs = torch.ones_like(coeffs)
            coeff_norm = torch.linalg.vector_norm(coeffs)
        return coeffs / coeff_norm.clamp_min(eps)

    def _curriculum_bias_weights(self, basis_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weights = torch.empty(basis_size, device=device, dtype=dtype)
        for idx in range(basis_size):
            weights[idx] = self.primary_scale / (self.secondary_scale ** idx)
        return weights

    def _sample_coefficients(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        work_dtype = torch.float32
        eps = torch.finfo(work_dtype).eps
        basis_size = self.basis_vectors.size(0)

        if self._in_warmup_phase():
            coeffs = torch.ones(basis_size, device=device, dtype=work_dtype)
            return coeffs.to(dtype=dtype)

        bias_weights = None
        if self.curriculum == "warmup_expand" and basis_size > 1 and self._warmup_completed():
            bias_weights = self._curriculum_bias_weights(
                basis_size,
                device=device,
                dtype=work_dtype,
            )

        if self._in_second_warmup_phase():
            coeffs = bias_weights
            if not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
                coeffs = torch.ones(basis_size, device=device, dtype=work_dtype)
            return coeffs.to(dtype=dtype)

        if self.sampling_method == "interpolation":
            coeffs = self._sample_simplex_coefficients(
                basis_size,
                device=device,
                dtype=work_dtype,
                eps=eps,
            )
        else:
            coeffs = self._sample_positive_hypersphere_coefficients(
                basis_size,
                device=device,
                dtype=work_dtype,
                eps=eps,
            )

        if bias_weights is not None:
            coeffs = coeffs * bias_weights
            if not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
                coeffs = bias_weights.clone()
            coeffs = self._renormalize_coefficients(coeffs, eps=eps)

        return coeffs.to(dtype=dtype)

    def resample_cached_coefficients(self) -> None:
        if self.num_vectors <= 1:
            self._cached_sample_coefficients = None
            return
        coeffs = self._sample_coefficients(device=self.basis_vectors.device, dtype=self.basis_vectors.dtype)
        self._cached_sample_coefficients = coeffs.detach()

    def project_gradients_to_tangent(self) -> None:
        if self.num_vectors == 1:
            if self.raw_single_vector:
                return
            if self.steer_vector.grad is None:
                return

            with torch.no_grad():
                vec = self.steer_vector.data.float()
                grad = self.steer_vector.grad.data.float()
                eps = 1e-8
                vec_norm_sq = torch.dot(vec, vec)
                if torch.isfinite(vec_norm_sq) and vec_norm_sq > eps:
                    grad = grad - (torch.dot(grad, vec) / vec_norm_sq) * vec
                    self.steer_vector.grad.data.copy_(grad.to(dtype=self.steer_vector.grad.dtype))
            return

        if self.basis_vectors.grad is None:
            return

        with torch.no_grad():
            basis = self.basis_vectors.data.float()
            grad = self.basis_vectors.grad.data.float()
            eps = 1e-8

            for idx in range(basis.size(0)):
                vec = basis[idx]
                vec_norm_sq = torch.dot(vec, vec)
                if not torch.isfinite(vec_norm_sq) or vec_norm_sq <= eps:
                    continue
                grad[idx] = grad[idx] - (torch.dot(grad[idx], vec) / vec_norm_sq) * vec

            self.basis_vectors.grad.data.copy_(grad.to(dtype=self.basis_vectors.grad.dtype))

    def renormalize_direction_parameters(self) -> None:
        if self.num_vectors == 1:
            if self.raw_single_vector:
                return
            with torch.no_grad():
                normalized = self._normalize_vector(self.steer_vector.data.float())
                self.steer_vector.data.copy_(normalized.to(dtype=self.steer_vector.dtype))
            return

        self.orthogonalize_basis()

    def sample_vector(self) -> torch.Tensor:
        if self.num_vectors == 1:
            if self.raw_single_vector:
                return self.steer_vector
            alpha = self.get_effective_alpha()
            direction = self._normalize_vector(self.steer_vector)
            return direction * alpha.to(device=direction.device, dtype=direction.dtype)

        alpha = self.get_effective_alpha()
        device = self.basis_vectors.device
        dtype = self.basis_vectors.dtype
        coeffs = self._cached_sample_coefficients
        if coeffs is None or coeffs.device != device or coeffs.dtype != dtype:
            coeffs = self._sample_coefficients(device=device, dtype=dtype)

        vector = torch.sum(self.basis_vectors * coeffs.unsqueeze(-1), dim=0)
        vector = self._normalize_vector(vector)
        return vector * alpha.to(device=device, dtype=dtype)

    def get_all_vectors(self) -> list[torch.Tensor]:
        if self.num_vectors == 1:
            return [self.steer_vector]
        return [self.basis_vectors[idx] for idx in range(self.basis_vectors.size(0))]

    def _normalize_token_mask(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        token_mask = self._response_token_mask
        if token_mask is None:
            return None

        target_dims = hidden_states.dim() - 1
        while token_mask.dim() < target_dims:
            token_mask = token_mask.unsqueeze(0)
        while token_mask.dim() > target_dims and token_mask.size(-1) == 1:
            token_mask = token_mask.squeeze(-1)

        expected_shape = hidden_states.shape[:-1]
        if tuple(token_mask.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"Response mask shape mismatch: mask={tuple(token_mask.shape)}, expected={tuple(expected_shape)}"
            )
        return token_mask

    def _add_vector(self, hidden_states: torch.Tensor) -> torch.Tensor:
        delta_param = self.sample_vector()
        if hidden_states.size(-1) != delta_param.numel():
            raise RuntimeError(
                f"Hidden size mismatch: hidden_states.size(-1)={hidden_states.size(-1)}, "
                f"steer_vector.numel()={delta_param.numel()}"
            )

        delta = delta_param.to(device=hidden_states.device, dtype=hidden_states.dtype)
        view_shape = [1] * hidden_states.dim()
        view_shape[-1] = -1

        token_mask = self._normalize_token_mask(hidden_states)
        if token_mask is None:
            return hidden_states + delta.view(*view_shape)

        token_mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return hidden_states + delta.view(*view_shape) * token_mask

    def hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = self._add_vector(output[0])
            return (hidden_states,) + output[1:]
        if torch.is_tensor(output):
            return self._add_vector(output)
        return output


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers for this model architecture.")


def get_hidden_size(model) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "n_embd"):
        return int(config.n_embd)
    if hasattr(config, "d_model"):
        return int(config.d_model)
    raise RuntimeError("Cannot infer hidden size from model.config.")


def install_single_trainable_vector_hook(
    model,
    layer_idx: int,
    dtype: Optional[torch.dtype] = None,
    *,
    num_vectors: int = 1,
    sampling_method: str = "hypersphere",
    vector_scale: float = 1.0,
    learnable_alpha: bool = False,
    alpha_init: float = 1.0,
    curriculum: str = "none",
    warmup_steps: int = 0,
    warmup_end_step: Optional[int] = None,
    secondary_freeze_steps: int = 0,
    secondary_end_step: Optional[int] = None,
    primary_scale: float = 1.0,
    secondary_scale: float = 1.0,
    freeze_primary_after_warmup: bool = False,
):
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    if layer_idx < 0:
        layer_idx += num_layers
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx={layer_idx}, model has {num_layers} layers.")

    layer = layers[layer_idx]
    hidden_size = get_hidden_size(model)
    hook_name = "_single_trainable_vector_hook" if int(num_vectors) == 1 else "_multi_trainable_vector_hook"

    if hasattr(layer, hook_name):
        raise RuntimeError(f"Hook already installed on layer {layer_idx}")

    hook_module = TrainableTokenVectorHook(
        hidden_size=hidden_size,
        dtype=dtype,
        num_vectors=int(num_vectors),
        sampling_method=sampling_method,
        vector_scale=vector_scale,
        learnable_alpha=learnable_alpha,
        alpha_init=alpha_init,
        curriculum=curriculum,
        warmup_steps=warmup_steps,
        warmup_end_step=warmup_end_step,
        secondary_freeze_steps=secondary_freeze_steps,
        secondary_end_step=secondary_end_step,
        primary_scale=primary_scale,
        secondary_scale=secondary_scale,
        freeze_primary_after_warmup=freeze_primary_after_warmup,
    )
    layer.add_module(hook_name, hook_module)
    handle = layer.register_forward_hook(hook_module.hook)
    handles = getattr(model, "_single_trainable_vector_hook_handles", [])
    handles.append(handle)
    model._single_trainable_vector_hook_handles = handles
    model._single_trainable_vector_layer_idx = layer_idx
    return hook_module


def orthogonalize_trainable_vector_hooks(module: nn.Module) -> None:
    layers = _get_core_decoder_layers(module)
    if layers is None:
        return

    for layer in layers:
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is None:
                continue
            hook.maybe_activate_expanded_basis()
            renorm_fn = getattr(hook, "renormalize_direction_parameters", None)
            if renorm_fn is not None:
                renorm_fn()


def set_trainable_vector_global_step(module: nn.Module, global_step: Optional[int]) -> None:
    layers = _get_core_decoder_layers(module)
    if layers is None:
        return

    for layer in layers:
        for hook_name in ("_single_trainable_vector_hook", "_multi_trainable_vector_hook"):
            hook = getattr(layer, hook_name, None)
            if hook is not None:
                set_step_fn = getattr(hook, "set_current_global_step", None)
                if set_step_fn is not None:
                    set_step_fn(global_step)
                activate_fn = getattr(hook, "maybe_activate_expanded_basis", None)
                if activate_fn is not None:
                    activate_fn()


def install_trainable_token_vector_hooks(
    model,
    *,
    layer_idx: int,
    all_layers: bool,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    num_vectors: int = 1,
    sampling_method: str = "hypersphere",
    vector_scale: float = 1.0,
    learnable_alpha: bool = False,
    alpha_init: float = 1.0,
    curriculum: str = "none",
    warmup_steps: int = 0,
    warmup_end_step: Optional[int] = None,
    secondary_freeze_steps: int = 0,
    secondary_end_step: Optional[int] = None,
    primary_scale: float = 1.0,
    secondary_scale: float = 1.0,
    freeze_primary_after_warmup: bool = False,
) -> list[tuple[int, TrainableTokenVectorHook]]:
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    if (layer_start is None) ^ (layer_end is None):
        raise ValueError("trainable_token_vector_layer_start and layer_end must be both set or both unset")

    def _normalize_idx(idx: int) -> int:
        if idx < 0:
            idx += num_layers
        return idx

    target_indices: list[int]
    if all_layers:
        target_indices = list(range(num_layers))
    elif layer_start is not None and layer_end is not None:
        start = _normalize_idx(int(layer_start))
        end = _normalize_idx(int(layer_end))
        if start < 0 or start >= num_layers or end < 0 or end >= num_layers:
            raise ValueError(
                f"Invalid layer range [{layer_start}, {layer_end}], model has {num_layers} layers."
            )
        if start > end:
            raise ValueError(
                f"Invalid layer range: start ({start}) must be <= end ({end})."
            )
        target_indices = list(range(start, end + 1))
    else:
        idx = _normalize_idx(layer_idx)
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Invalid layer_idx={layer_idx}, model has {num_layers} layers.")
        target_indices = [idx]

    installed: list[tuple[int, TrainableTokenVectorHook]] = []
    for idx in target_indices:
        hook_module = install_single_trainable_vector_hook(
            model=model,
            layer_idx=idx,
            dtype=dtype,
            num_vectors=num_vectors,
            sampling_method=sampling_method,
            vector_scale=vector_scale,
            learnable_alpha=learnable_alpha,
            alpha_init=alpha_init,
            curriculum=curriculum,
            warmup_steps=warmup_steps,
            warmup_end_step=warmup_end_step,
            secondary_freeze_steps=secondary_freeze_steps,
            secondary_end_step=secondary_end_step,
            primary_scale=primary_scale,
            secondary_scale=secondary_scale,
            freeze_primary_after_warmup=freeze_primary_after_warmup,
        )
        installed.append((idx, hook_module))

    model._single_trainable_vector_layer_indices = [idx for idx, _ in installed]
    return installed


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
        self.acc_rate = 1
        self._current_global_step = 0

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        use_trainable_vector = bool(self.config.model.get("enable_trainable_token_vector", False))
        use_qkv_bias_only = bool(self.config.model.get("train_q_bias_only", False))
        q_bias_layer_start = self.config.model.get("q_bias_layer_start", None)
        q_bias_layer_end = self.config.model.get("q_bias_layer_end", None)
        use_single_layer_only = bool(self.config.model.get("train_single_layer_only", False))
        single_layer_idx = self.config.model.get("train_layer_idx", None)
        use_single_mlp_only = bool(self.config.model.get("train_single_mlp_only", False))
        single_mlp_layer_idx = self.config.model.get("train_mlp_layer_idx", None)
        trainable_vector_mode = str(self.config.model.get("trainable_token_vector_mode", "single"))
        trainable_vector_num = int(self.config.model.get("trainable_token_vector_num", 1))
        trainable_vector_sampling_method = str(
            self.config.model.get("trainable_token_vector_sampling_method", "hypersphere")
        )
        trainable_vector_scale = float(self.config.model.get("trainable_token_vector_scale", 1.0))
        trainable_vector_learnable_alpha = bool(
            self.config.model.get("trainable_token_vector_learnable_alpha", False)
        )
        trainable_vector_alpha_init = float(self.config.model.get("trainable_token_vector_alpha_init", 1.0))
        trainable_vector_curriculum = str(self.config.model.get("trainable_token_vector_curriculum", "none"))
        trainable_vector_warmup_steps = int(self.config.model.get("trainable_token_vector_warmup_steps", 0))
        trainable_vector_warmup_end_step = self.config.model.get("trainable_token_vector_warmup_end_step", None)
        trainable_vector_secondary_freeze_steps = int(
            self.config.model.get("trainable_token_vector_secondary_freeze_steps", 0)
        )
        trainable_vector_secondary_end_step = self.config.model.get("trainable_token_vector_secondary_end_step", None)
        trainable_vector_primary_scale = float(self.config.model.get("trainable_token_vector_primary_scale", 1.0))
        trainable_vector_secondary_scale = float(self.config.model.get("trainable_token_vector_secondary_scale", 1.0))
        trainable_vector_freeze_primary_after_warmup = bool(
            self.config.model.get("trainable_token_vector_freeze_primary_after_warmup", False)
        )
        if trainable_vector_mode not in {"single", "multi"}:
            raise ValueError(
                f"trainable_token_vector_mode must be 'single' or 'multi', got {trainable_vector_mode!r}"
            )
        if trainable_vector_sampling_method not in {"hypersphere", "interpolation"}:
            raise ValueError(
                "trainable_token_vector_sampling_method must be 'hypersphere' or 'interpolation', "
                f"got {trainable_vector_sampling_method!r}"
            )
        if trainable_vector_scale <= 0:
            raise ValueError(f"trainable_token_vector_scale must be > 0, got {trainable_vector_scale}")
        if trainable_vector_curriculum not in {"none", "warmup_expand"}:
            raise ValueError(
                "trainable_token_vector_curriculum must be 'none' or 'warmup_expand', "
                f"got {trainable_vector_curriculum!r}"
            )
        if trainable_vector_warmup_steps < 0:
            raise ValueError(
                f"trainable_token_vector_warmup_steps must be >= 0, got {trainable_vector_warmup_steps}"
            )
        if trainable_vector_warmup_end_step is not None:
            trainable_vector_warmup_end_step = int(trainable_vector_warmup_end_step)
            if trainable_vector_warmup_end_step < 0:
                raise ValueError(
                    "trainable_token_vector_warmup_end_step must be >= 0 when set, "
                    f"got {trainable_vector_warmup_end_step}"
                )
        if trainable_vector_secondary_freeze_steps < 0:
            raise ValueError(
                "trainable_token_vector_secondary_freeze_steps must be >= 0, "
                f"got {trainable_vector_secondary_freeze_steps}"
            )
        if trainable_vector_secondary_end_step is not None:
            trainable_vector_secondary_end_step = int(trainable_vector_secondary_end_step)
            if trainable_vector_secondary_end_step < 0:
                raise ValueError(
                    "trainable_token_vector_secondary_end_step must be >= 0 when set, "
                    f"got {trainable_vector_secondary_end_step}"
                )
        if (
            trainable_vector_warmup_end_step is not None
            and trainable_vector_secondary_end_step is not None
            and trainable_vector_secondary_end_step < trainable_vector_warmup_end_step
        ):
            raise ValueError(
                "trainable_token_vector_secondary_end_step must be >= trainable_token_vector_warmup_end_step, "
                f"got {trainable_vector_secondary_end_step} < {trainable_vector_warmup_end_step}"
            )
        if trainable_vector_primary_scale <= 0:
            raise ValueError(
                f"trainable_token_vector_primary_scale must be > 0, got {trainable_vector_primary_scale}"
            )
        if trainable_vector_secondary_scale <= 0:
            raise ValueError(
                f"trainable_token_vector_secondary_scale must be > 0, got {trainable_vector_secondary_scale}"
            )
        if trainable_vector_mode == "single":
            trainable_vector_num = 1
        elif trainable_vector_num < 2:
            raise ValueError("trainable_token_vector_mode='multi' requires trainable_token_vector_num >= 2")
        layer_idx = int(self.config.model.get("trainable_token_vector_layer_idx", 16))
        all_layers = bool(self.config.model.get("trainable_token_vector_all_layers", False))
        layer_start = self.config.model.get("trainable_token_vector_layer_start", None)
        layer_end = self.config.model.get("trainable_token_vector_layer_end", None)
        force_all_tokens = bool(self.config.model.get("trainable_token_vector_force_all_tokens", False))

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:

                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if role == "actor" and optim_config is not None and use_trainable_vector:
            try:
                vector_dtype = next(actor_module.parameters()).dtype
            except StopIteration:
                vector_dtype = torch_dtype

            installed_hooks = install_trainable_token_vector_hooks(
                model=actor_module,
                layer_idx=layer_idx,
                all_layers=all_layers,
                layer_start=layer_start,
                layer_end=layer_end,
                dtype=vector_dtype,
                num_vectors=trainable_vector_num,
                sampling_method=trainable_vector_sampling_method,
                vector_scale=trainable_vector_scale,
                learnable_alpha=trainable_vector_learnable_alpha,
                alpha_init=trainable_vector_alpha_init,
                curriculum=trainable_vector_curriculum,
                warmup_steps=trainable_vector_warmup_steps,
                warmup_end_step=trainable_vector_warmup_end_step,
                secondary_freeze_steps=trainable_vector_secondary_freeze_steps,
                secondary_end_step=trainable_vector_secondary_end_step,
                primary_scale=trainable_vector_primary_scale,
                secondary_scale=trainable_vector_secondary_scale,
                freeze_primary_after_warmup=trainable_vector_freeze_primary_after_warmup,
            )

            if self.rank == 0:
                layer_indices = [idx for idx, _ in installed_hooks]
                print(
                        f"Installed {len(installed_hooks)} trainable vector hooks at layers={layer_indices}, "
                        f"mode={trainable_vector_mode}, vectors_per_layer={trainable_vector_num}, "
                        f"sampling={trainable_vector_sampling_method}, scale={trainable_vector_scale}, "
                        f"learnable_alpha={trainable_vector_learnable_alpha}, alpha_init={trainable_vector_alpha_init}, "
                        f"curriculum={trainable_vector_curriculum}, warmup_steps={trainable_vector_warmup_steps}, "
                        f"warmup_end={trainable_vector_warmup_end_step}, "
                        f"secondary_freeze_steps={trainable_vector_secondary_freeze_steps}, "
                        f"secondary_end={trainable_vector_secondary_end_step}"
                    )

            # Make actor-side forward respect force_all_tokens mode as well (not only vLLM rollout).
            actor_module._single_trainable_vector_force_all_tokens = force_all_tokens

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        trainable_vector_ignored_states = None
        if role == "actor" and optim_config is not None and use_trainable_vector:
            if fsdp_config.get("strategy", self.config.actor.strategy) == "fsdp" and not fsdp_config.get(
                "use_orig_params", False
            ):
                raise RuntimeError(
                    "enable_trainable_token_vector=True requires actor.fsdp_config.use_orig_params=True under FSDP"
                )

            for _, p in actor_module.named_parameters():
                p.requires_grad_(False)

            vector_params = []
            for name, p in actor_module.named_parameters():
                if _is_trainable_vector_param_name(name):
                    p.requires_grad_(True)
                    vector_params.append((name, p))

            if len(vector_params) == 0:
                names = [n for n, _ in vector_params]
                raise RuntimeError(f"Expected at least one steer_vector, found {len(vector_params)}: {names}")

            trainable_vector_ignored_states = _get_trainable_vector_hook_modules(actor_module)

            if self.rank == 0:
                print(f"Only trainable steer_vector parameters will be updated: count={len(vector_params)}")
                preview = vector_params[:3]
                for name, p in preview:
                    print(f"  {name}, shape={tuple(p.shape)}, dtype={p.dtype}, numel={p.numel()}")
                if len(vector_params) > len(preview):
                    print(f"  ... and {len(vector_params) - len(preview)} more")
                print(f"FSDP ignored trainable vector hook modules: count={len(trainable_vector_ignored_states)}")

        if role == "actor" and optim_config is not None and use_qkv_bias_only:
            if fsdp_config.get("strategy", self.config.actor.strategy) == "fsdp" and not fsdp_config.get(
                "use_orig_params", False
            ):
                raise RuntimeError(
                    "train_q_bias_only=True requires actor.fsdp_config.use_orig_params=True under FSDP"
                )

            for _, p in actor_module.named_parameters():
                p.requires_grad_(False)

            qkv_bias_params = []
            for name, p in actor_module.named_parameters():
                if _is_q_bias_param_name(name, q_bias_layer_start, q_bias_layer_end):
                    p.requires_grad_(True)
                    qkv_bias_params.append((name, p))

            if len(qkv_bias_params) == 0:
                raise RuntimeError(
                    "train_q_bias_only=True but found no self_attn.q_proj.bias params in range "
                    f"[{q_bias_layer_start}, {q_bias_layer_end}]. "
                    "Check the model has attention bias and the layer range is valid."
                )

            if self.rank == 0:
                total_numel = sum(p.numel() for _, p in qkv_bias_params)
                print(
                    f"Only q_proj bias parameters will be updated (layers [{q_bias_layer_start}, "
                    f"{q_bias_layer_end}]): count={len(qkv_bias_params)}, total_numel={total_numel}"
                )

        if role == "actor" and optim_config is not None and use_single_layer_only:
            if single_layer_idx is None:
                raise RuntimeError("train_single_layer_only=True requires train_layer_idx to be set.")
            if fsdp_config.get("strategy", self.config.actor.strategy) == "fsdp" and not fsdp_config.get(
                "use_orig_params", False
            ):
                raise RuntimeError(
                    "train_single_layer_only=True requires actor.fsdp_config.use_orig_params=True under FSDP"
                )

            for _, p in actor_module.named_parameters():
                p.requires_grad_(False)

            single_layer_params = []
            for name, p in actor_module.named_parameters():
                if _is_single_layer_param_name(name, single_layer_idx):
                    p.requires_grad_(True)
                    single_layer_params.append((name, p))

            if len(single_layer_params) == 0:
                raise RuntimeError(
                    f"train_single_layer_only=True but found no params in layer {single_layer_idx}. "
                    "Check train_layer_idx is a valid decoder layer index."
                )

            if self.rank == 0:
                total_numel = sum(p.numel() for _, p in single_layer_params)
                print(
                    f"Only layer {single_layer_idx} params will be updated: "
                    f"count={len(single_layer_params)}, total_numel={total_numel}"
                )

        if role == "actor" and optim_config is not None and use_single_mlp_only:
            if single_mlp_layer_idx is None:
                raise RuntimeError("train_single_mlp_only=True requires train_mlp_layer_idx to be set.")
            if fsdp_config.get("strategy", self.config.actor.strategy) == "fsdp" and not fsdp_config.get(
                "use_orig_params", False
            ):
                raise RuntimeError(
                    "train_single_mlp_only=True requires actor.fsdp_config.use_orig_params=True under FSDP"
                )

            for _, p in actor_module.named_parameters():
                p.requires_grad_(False)

            single_mlp_params = []
            for name, p in actor_module.named_parameters():
                if _is_single_mlp_param_name(name, single_mlp_layer_idx):
                    p.requires_grad_(True)
                    single_mlp_params.append((name, p))

            if len(single_mlp_params) == 0:
                raise RuntimeError(
                    f"train_single_mlp_only=True but found no mlp down_proj params in layer {single_mlp_layer_idx}. "
                    "Check train_mlp_layer_idx is a valid decoder layer index."
                )

            if self.rank == 0:
                total_numel = sum(p.numel() for _, p in single_mlp_params)
                print(
                    f"Only layer {single_mlp_layer_idx} MLP down_proj params will be updated: "
                    f"count={len(single_mlp_params)}, total_numel={total_numel}"
                )

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
                ignored_states=trainable_vector_ignored_states,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            if use_trainable_vector:
                vector_params = [
                    (name, p)
                    for name, p in actor_module.named_parameters()
                    if _is_trainable_vector_param_name(name)
                ]

                if len(vector_params) == 0:
                    names = [n for n, _ in vector_params]
                    raise RuntimeError(
                        f"Optimizer expected at least one steer_vector after FSDP, found {len(vector_params)}: {names}"
                    )

                for name, vector_param in vector_params:
                    if not vector_param.requires_grad:
                        raise RuntimeError(f"{name} exists but requires_grad=False")

                if self.rank == 0:
                    total_numel = sum(p.numel() for _, p in vector_params)
                    print(
                        f"Optimizer will update steer_vector parameters: count={len(vector_params)}, total_numel={total_numel}"
                    )

                actor_optimizer = build_optimizer([p for _, p in vector_params], optim_config)
                orthogonalize_trainable_vector_hooks(actor_module)
            elif use_qkv_bias_only:
                qkv_bias_params = [
                    (name, p)
                    for name, p in actor_module.named_parameters()
                    if _is_q_bias_param_name(name, q_bias_layer_start, q_bias_layer_end)
                ]

                if len(qkv_bias_params) == 0:
                    raise RuntimeError("Optimizer expected q_proj bias params after FSDP, found none.")

                for name, p in qkv_bias_params:
                    if not p.requires_grad:
                        raise RuntimeError(f"{name} exists but requires_grad=False")

                if self.rank == 0:
                    total_numel = sum(p.numel() for _, p in qkv_bias_params)
                    print(
                        f"Optimizer will update q_proj bias parameters: "
                        f"count={len(qkv_bias_params)}, total_numel={total_numel}"
                    )

                actor_optimizer = build_optimizer([p for _, p in qkv_bias_params], optim_config)
            elif use_single_layer_only:
                single_layer_params = [
                    (name, p)
                    for name, p in actor_module.named_parameters()
                    if _is_single_layer_param_name(name, single_layer_idx)
                ]

                if len(single_layer_params) == 0:
                    raise RuntimeError(
                        f"Optimizer expected layer {single_layer_idx} params after FSDP, found none."
                    )

                for name, p in single_layer_params:
                    if not p.requires_grad:
                        raise RuntimeError(f"{name} exists but requires_grad=False")

                if self.rank == 0:
                    total_numel = sum(p.numel() for _, p in single_layer_params)
                    print(
                        f"Optimizer will update layer {single_layer_idx} params: "
                        f"count={len(single_layer_params)}, total_numel={total_numel}"
                    )

                actor_optimizer = build_optimizer([p for _, p in single_layer_params], optim_config)
            elif use_single_mlp_only:
                single_mlp_params = [
                    (name, p)
                    for name, p in actor_module.named_parameters()
                    if _is_single_mlp_param_name(name, single_mlp_layer_idx)
                ]

                if len(single_mlp_params) == 0:
                    raise RuntimeError(
                        f"Optimizer expected layer {single_mlp_layer_idx} MLP down_proj params after FSDP, found none."
                    )

                for name, p in single_mlp_params:
                    if not p.requires_grad:
                        raise RuntimeError(f"{name} exists but requires_grad=False")

                if self.rank == 0:
                    total_numel = sum(p.numel() for _, p in single_mlp_params)
                    print(
                        f"Optimizer will update layer {single_mlp_layer_idx} MLP down_proj params: "
                        f"count={len(single_mlp_params)}, total_numel={total_numel}"
                    )

                actor_optimizer = build_optimizer([p for _, p in single_mlp_params], optim_config)
            else:
                actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        vector_only_rollout_preloaded = bool(
            self.config.rollout.name == "vllm"
            and not self._is_lora
            and self.config.model.get("enable_trainable_token_vector", False)
        )
        if vector_only_rollout_preloaded and str(self.config.rollout.load_format).startswith("dummy"):
            if self.rank == 0:
                print(
                    "Trainable token vector enabled: overriding rollout.load_format from dummy to auto "
                    "so vLLM preloads base weights and rollout stays vector-only."
                )
            self.config.rollout.load_format = "auto"

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        enable_trainable_token_vector = bool(self.config.model.get("enable_trainable_token_vector", False))
        single_vector_layer_idx = int(self.config.model.get("trainable_token_vector_layer_idx", 16))
        steer_vector_sampling_method = str(
            self.config.model.get("trainable_token_vector_sampling_method", "hypersphere")
        )
        steer_vector_scale = float(self.config.model.get("trainable_token_vector_scale", 1.0))
        steer_vector_curriculum = str(self.config.model.get("trainable_token_vector_curriculum", "none"))
        steer_vector_warmup_steps = int(self.config.model.get("trainable_token_vector_warmup_steps", 0))
        steer_vector_warmup_end_step = self.config.model.get("trainable_token_vector_warmup_end_step", None)
        steer_vector_secondary_freeze_steps = int(
            self.config.model.get("trainable_token_vector_secondary_freeze_steps", 0)
        )
        steer_vector_secondary_end_step = self.config.model.get("trainable_token_vector_secondary_end_step", None)
        steer_vector_primary_scale = float(self.config.model.get("trainable_token_vector_primary_scale", 1.0))
        steer_vector_secondary_scale = float(self.config.model.get("trainable_token_vector_secondary_scale", 1.0))
        steer_vector_freeze_primary_after_warmup = bool(
            self.config.model.get("trainable_token_vector_freeze_primary_after_warmup", False)
        )
        force_all_tokens = bool(self.config.model.get("trainable_token_vector_force_all_tokens", False))

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        if self._is_actor and enable_trainable_token_vector:
            _sync_trainable_vector_state_across_ranks(self.actor_module_fsdp)

        peft_config = None
        actor_wrapped = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        peft_model = getattr(actor_wrapped, "module", actor_wrapped)
        steer_vector_tensor_by_layer = _collect_trainable_vector_payload_by_layer(self.actor_module_fsdp)
        steer_vector_layer_idx = single_vector_layer_idx if steer_vector_tensor_by_layer is not None else None
        vector_only_rollout_update = bool(
            enable_trainable_token_vector
            and not hasattr(peft_model, "peft_config")
            and self.base_sync_done
            and steer_vector_tensor_by_layer is not None
        )
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        elif vector_only_rollout_update:
            params = {}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(params, peft_model)
        params = _filter_rollout_incompatible_params(
            params,
            enable_trainable_token_vector=enable_trainable_token_vector,
        )

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(base_model_params, peft_model)
            base_model_params = _filter_rollout_incompatible_params(
                base_model_params,
                enable_trainable_token_vector=enable_trainable_token_vector,
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        if vector_only_rollout_update:
            per_tensor_param = iter(())
        elif peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(
            per_tensor_param,
            peft_config=peft_config,
            base_sync_done=self.base_sync_done,
            vector_only_update=vector_only_rollout_update,
            steer_vector=steer_vector_tensor_by_layer,
            steer_vector_layer_idx=steer_vector_layer_idx,
            steer_vector_sampling_method=steer_vector_sampling_method,
            steer_vector_scale=steer_vector_scale,
            steer_vector_curriculum=steer_vector_curriculum,
            steer_vector_warmup_steps=steer_vector_warmup_steps,
            steer_vector_warmup_end_step=steer_vector_warmup_end_step,
            steer_vector_secondary_freeze_steps=steer_vector_secondary_freeze_steps,
            steer_vector_secondary_end_step=steer_vector_secondary_end_step,
            steer_vector_primary_scale=steer_vector_primary_scale,
            steer_vector_secondary_scale=steer_vector_secondary_scale,
            steer_vector_freeze_primary_after_warmup=steer_vector_freeze_primary_after_warmup,
            steer_vector_global_step=self._current_global_step,
            steer_vector_force_all_tokens=force_all_tokens,
            steer_vector_debug_forward_once=True,
        )
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )
            self._current_global_step = 0
            self.actor.set_trainable_vector_global_step(self._current_global_step)

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        # Initialize base models for corrected reward computation
        # Actor's base model (for computing base_log_prob)
        self.base_policy = None
        self._has_base_model = False
        base_model_path = self.config.model.get("base_model_path", None)
        if base_model_path is not None and self._is_actor:
            if self.rank == 0:
                print(f"Actor base model: {base_model_path}")
            local_base_path = copy_to_local(base_model_path, use_shm=use_shm)
            self.base_module_fsdp = self._build_model_optimizer(
                model_path=local_base_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),  # Use ref fsdp config for base model
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",  # Use ref role for CPU offload
            )[0]
            # Create a config for base policy similar to ref config
            base_config = OmegaConf.create(OmegaConf.to_container(self.config.ref))
            OmegaConf.set_struct(base_config, True)
            with open_dict(base_config):
                base_config.use_remove_padding = use_remove_padding
                base_config.use_fused_kernels = use_fused_kernels
            self.base_policy = DataParallelPPOActor(config=base_config, actor_module=self.base_module_fsdp)
            self._has_base_model = True
            if self.rank == 0:
                print(f"Actor base model initialized successfully from {base_model_path}")

        # Ref's base model (for computing base_ref_log_prob)
        self.base_ref_policy = None
        self._has_base_ref_model = False
        ref_model_config = self.config.ref.get("model", {})
        ref_base_model_path = ref_model_config.get("base_model_path", None) if ref_model_config else None
        if ref_base_model_path is not None and self._is_ref:
            if self.rank == 0:
                print(f"Ref base model: {ref_base_model_path}")
            local_ref_base_path = copy_to_local(ref_base_model_path, use_shm=use_shm)
            self.base_ref_module_fsdp = self._build_model_optimizer(
                model_path=local_ref_base_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            # Create a config for base ref policy
            base_ref_config = OmegaConf.create(OmegaConf.to_container(self.config.ref))
            OmegaConf.set_struct(base_ref_config, True)
            with open_dict(base_ref_config):
                base_ref_config.use_remove_padding = use_remove_padding
                base_ref_config.use_fused_kernels = use_fused_kernels
            self.base_ref_policy = DataParallelPPOActor(config=base_ref_config, actor_module=self.base_ref_module_fsdp)
            self._has_base_ref_model = True
            if self.rank == 0:
                print(f"Ref base model initialized successfully from {ref_base_model_path}")

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            self._current_global_step = int(data.meta_info.get("global_step", self._current_global_step))
            self.actor.set_trainable_vector_global_step(self._current_global_step)

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            if data.meta_info.get("enable_iterative_test", False):
                test_data = data.meta_info.get("test_batches", None)
                max_test_iterations = data.meta_info.get("max_test_iterations", 5)  
                val_reward_fn = data.meta_info.get("val_reward_fn", None)

                global_step = data.meta_info.get("global_step", 1)
                if test_data is not None and global_step > 0 and global_step % 1 == 0: 
                # if test_data is not None and global_step % 2 == 0:   
                    test_results = self._iterative_test_and_modify(
                        test_data,
                        max_test_iterations,
                        val_reward_fn=val_reward_fn,
                        global_step=global_step,
                        total_training_steps=data.meta_info.get("total_training_steps", 1),
                    )
                    # 标量 metrics 写入 wandb
                    metrics["iterative_test/best_score"] = test_results["best_score"]
                    metrics["iterative_test/initial_score"] = test_results["history"][0]["score"]
                    metrics["iterative_test/num_iters"] = len(test_results["history"]) - 1

                    # 每次迭代的 score 单独记录
                    for entry in test_results["history"]:
                        i = entry["iteration"]
                        metrics[f"iterative_test/iter_{i}_score"] = entry["score"]
                        metrics[f"iterative_test/iter_{i}_accepted"] = int(entry["accepted"])

                    # 完整 history 打印到终端
                    if self.rank == 0:
                        import json
                        print(f"[Iterative Test] history:\n{json.dumps(test_results['history'], indent=2)}", flush=True)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output
    

    def _run_iterative_test_on_batch(
        self,
        batch_dict: dict,
        val_reward_fn,
    ) -> list[float]:
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from copy import deepcopy

        if val_reward_fn is None:
            raise ValueError("val_reward_fn is None")

        # 1. 拷贝缓存里的 test_batch，避免 pop / union 等操作污染 all_batches
        test_batch = deepcopy(batch_dict["test_batch"])

        # ============================================================
        # 2. 内联 _get_gen_batch(self, batch: DataProto) 的完整逻辑
        # ============================================================

        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid"})
            & test_batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

        non_tensor_batch_keys_to_pop = (
            set(test_batch.non_tensor_batch.keys()) - reward_model_keys
        )

        gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # ============================================================
        # 3. 设置生成 meta_info
        # ============================================================

        if gen_batch.meta_info is None:
            gen_batch.meta_info = {}

        gen_batch.meta_info.update({
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
        })

        # 4. pad 到 world_size 的倍数
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch,
            self.world_size,
        )

        # 5. 生成
        with torch.no_grad():
            output_padded = self.generate_sequences(gen_batch_padded)

        # 6. unpad
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        # 7. 保持原 validate 逻辑：把剩余 test_batch 和 output 合并
        #
        # 注意：
        # test_batch 已经被 pop 掉 input_ids / attention_mask / position_ids
        # 以及非 reward_model_keys 的 non_tensor 字段。
        #
        # 因此这里 union(output) 通常不会和 output 冲突。
        test_batch_with_output = test_batch.union(output)

        if test_batch_with_output.meta_info is None:
            test_batch_with_output.meta_info = {}

        test_batch_with_output.meta_info["validate"] = True

        # 8. 调 reward function
        result = val_reward_fn(test_batch_with_output, return_dict=True)

        # 9. 提取 score
        if (
            isinstance(result, dict)
            and "reward_extra_info" in result
            and "acc" in result["reward_extra_info"]
        ):
            scores = [
                1.0 if x else 0.0
                for x in result["reward_extra_info"]["acc"]
            ]
        else:
            scores = result["reward_tensor"].sum(-1).cpu().tolist()

        return scores

    def _run_iterative_test(self, test_batches: list, val_reward_fn) -> dict:  # ← 新增参数
        import torch.distributed as dist

        all_scores = []
        for test_batch in test_batches:
            batch_scores = self._run_iterative_test_on_batch(test_batch, val_reward_fn)  # ← 传进去
            all_scores.extend(batch_scores)

        local_sum = float(sum(all_scores))
        local_count = float(len(all_scores))

        # ── 全局同步，防止各 rank 决策不一致导致 NCCL 卡死 ──
        if dist.is_initialized():
            tensor = torch.tensor(
                [local_sum, local_count],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            global_sum = tensor[0].item()
            global_count = tensor[1].item()
        else:
            global_sum = local_sum
            global_count = local_count

        avg_score = global_sum / global_count if global_count > 0 else 0.0
        return {
            "score": avg_score,
            "scores": all_scores,
            "num_samples": int(global_count),
        }
    
    def _should_modify_param(self, param_name: str) -> bool:
        """
        只修改 attention 和 mlp 相关参数
        兼容常见 transformer 命名
        """
        name = param_name.lower()

        keywords = [
            # attention
            "attn",
            "self_attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",

            # mlp / ffn
            "mlp",
            "feed_forward",
            "ffn",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        return any(k in name for k in keywords)


    def _apply_parameter_modification(self, config: dict):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        amplification_factor = config.get("amplification_factor", 5)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            load_fsdp_model_to_gpu(self.base_module_fsdp)

        modified_param_count = 0
        skipped_param_count = 0
        shape_mismatch_count = 0
        missing_param_count = 0

        try:
            with torch.no_grad():
                with FSDP.summon_full_params(
                    self.actor_module_fsdp,
                    writeback=True,
                    recurse=True,
                ):
                    with FSDP.summon_full_params(
                        self.base_module_fsdp,
                        writeback=False,
                        recurse=True,
                    ):
                        actor_named_params = dict(self.actor_module_fsdp.named_parameters())
                        base_named_params = dict(self.base_module_fsdp.named_parameters())

                        for name, actor_param in actor_named_params.items():
                            if not self._should_modify_param(name):
                                skipped_param_count += 1
                                continue

                            if name not in base_named_params:
                                skipped_param_count += 1
                                missing_param_count += 1
                                continue

                            base_param = base_named_params[name]

                            if actor_param.shape != base_param.shape:
                                skipped_param_count += 1
                                shape_mismatch_count += 1
                                continue

                            # 为了数值稳定，先在 float32 中计算 diff，再转回 actor_param.dtype
                            actor_data_fp32 = actor_param.data.float()
                            base_data_fp32 = base_param.data.to(
                                device=actor_param.device,
                                dtype=torch.float32,
                            )

                            diff_fp32 = actor_data_fp32 - base_data_fp32
                            updated = actor_data_fp32 + amplification_factor * diff_fp32

                            actor_param.data.copy_(updated.to(dtype=actor_param.dtype))
                            modified_param_count += 1

            if self.rank == 0:
                print(
                    f"[Iterative Test] Applied parameter modification: "
                    f"factor={amplification_factor}, "
                    f"modified={modified_param_count}, "
                    f"skipped={skipped_param_count}, "
                    f"missing={missing_param_count}, "
                    f"shape_mismatch={shape_mismatch_count}"
                )

        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.base_module_fsdp)
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    def _backup_params(self):
        backup = {}
        backup_count = 0
        total_elements = 0
        total_bytes = 0

        with torch.no_grad():
            with FSDP.summon_full_params(
                self.actor_module_fsdp,
                writeback=False,
                recurse=True,
            ):
                for name, param in self.actor_module_fsdp.named_parameters():
                    if param is None:
                        continue

                    # 只备份需要修改的参数
                    if not self._should_modify_param(name):
                        continue

                    # 直接在 CPU 上 clone
                    backup_tensor = param.detach().cpu().clone()

                    backup[name] = backup_tensor
                    backup_count += 1
                    total_elements += backup_tensor.numel()
                    total_bytes += backup_tensor.numel() * backup_tensor.element_size()

        if self.rank == 0:
            print(
                f"[Iterative Test] Backed up {backup_count} parameters "
                f"({total_elements:,} elements, ~{total_bytes / 1024**3:.2f} GB)"
            )

        return backup


    def _restore_params(self, backup):
        restore_count = 0
        skip_count = 0
        missing_count = 0
        shape_mismatch_count = 0
        total_elements = 0
        total_bytes = 0

        with torch.no_grad():
            with FSDP.summon_full_params(
                self.actor_module_fsdp,
                writeback=True,
                recurse=True,
            ):
                for name, param in self.actor_module_fsdp.named_parameters():
                    if param is None:
                        skip_count += 1
                        continue

                    # 只恢复需要修改的参数
                    if name not in backup:
                        skip_count += 1
                        missing_count += 1
                        continue

                    restored = backup[name]

                    if tuple(param.shape) != tuple(restored.shape):
                        skip_count += 1
                        shape_mismatch_count += 1
                        continue

                    # 将备份的 tensor 移回当前参数的设备
                    restored = restored.to(
                        device=param.device,
                        dtype=param.dtype,
                    )

                    # 关键：写回真实 live parameter
                    param.copy_(restored)

                    restore_count += 1
                    total_elements += param.numel()
                    total_bytes += param.numel() * param.element_size()

        if self.rank == 0:
            print(
                f"[Iterative Test] Restored {restore_count} parameters "
                f"({total_elements:,} elements, ~{total_bytes / 1024**3:.2f} GB), "
                f"skipped={skip_count}, "
                f"missing={missing_count}, "
                f"shape_mismatch={shape_mismatch_count}"
            )


    def _iterative_test_and_modify(
        self,
        test_data,
        max_test_iterations: int = 5,
        val_reward_fn=None,
        global_step: int = 1,
        total_training_steps: int = 1,
    ):

        # 统一在函数开头 load 到 GPU
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            load_fsdp_model_to_gpu(self.base_module_fsdp)


        best_results = self._run_iterative_test(test_data, val_reward_fn) 
        best_score = best_results["score"]

        best_backup = self._backup_params()

        denom = max(1, total_training_steps - 1)
        progress = float(global_step - 1) / float(denom)
        progress = max(0.0, min(1.0, progress))

        amplification_factor = 2.0 * (1.0 - progress) * self.acc_rate
        history = [{
            "iteration": 0,
            "score": best_score,
            "accepted": True,
            "next_amplification_factor": amplification_factor,
            "info": "initial evaluation"
        }]

        for i in range(1, max_test_iterations + 1):
            current_factor = amplification_factor
            modification_config = {"amplification_factor": current_factor}

            if self.rank == 0:
                print(f"[Iterative Test] Iteration {i}: applying modification with factor={current_factor:.6f}")

            self._apply_parameter_modification(modification_config)

            new_results = self._run_iterative_test(test_data, val_reward_fn) 
            new_score = new_results["score"]

            if new_score > best_score:
                if self.rank == 0:
                    print(
                        f"[Iterative Test] Iteration {i}: score improved "
                        f"{best_score:.4f} -> {new_score:.4f} (Δ={new_score - best_score:.4f}), accepted"
                    )
                
                best_score = new_score
                self.acc_rate = 1 - best_score
                best_results = new_results
                best_backup = self._backup_params()
                amplification_factor = current_factor / (1.0 + current_factor)

                history.append({
                    "iteration": i,
                    "score": new_score,
                    "accepted": True,
                    "modification_config": modification_config,
                    "next_amplification_factor": amplification_factor,
                })
            elif new_score == best_score:
                if self.rank == 0:
                    print(
                        f"[Iterative Test] Iteration {i}: score equal "
                        f"{best_score:.4f} -> {new_score:.4f} (Δ={new_score - best_score:.4f}), accepted"
                    )
                
                best_score = new_score
                best_results = new_results
                amplification_factor = current_factor / (1.0 + current_factor)

                history.append({
                    "iteration": i,
                    "score": new_score,
                    "accepted": True,
                    "modification_config": modification_config,
                    "next_amplification_factor": amplification_factor,
                })
            else:
                if self.rank == 0:
                    print(
                        f"[Iterative Test] Iteration {i}: score did not improve "
                        f"{best_score:.4f} -> {new_score:.4f} (Δ={new_score - best_score:.4f}), rejected, rolling back"
                    )
                
                self._restore_params(best_backup)

                history.append({
                    "iteration": i,
                    "score": new_score,
                    "accepted": False,
                    "modification_config": modification_config,
                    "next_amplification_factor": amplification_factor,
                    "info": "rollback to previous best state"
                })
                break

        if self.rank == 0:
            accepted_count = sum(1 for h in history if h.get("accepted", False))
            print(
                f"[Iterative Test] Completed: {len(history)} iterations, "
                f"{accepted_count} accepted, final score={best_score:.4f}"
            )

        # 统一在函数结尾 offload 回 CPU
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.base_module_fsdp)
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return {
            "best_score": best_score,
            "best_results": best_results,
            "history": history,
        }


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        self._current_global_step = int(prompts.meta_info.get("global_steps", 0))
        if self._is_actor:
            self.actor.set_trainable_vector_global_step(self._current_global_step)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        self._current_global_step = int(data.meta_info.get("global_step", self._current_global_step))
        self.actor.set_trainable_vector_global_step(self._current_global_step)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora and self._is_actor:
            # In LoRA actor mode, use actor with adapter disabled as reference policy.
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data

        assert self._is_ref
        # Otherwise, use standalone ref model (including LoRA runs with explicit ref model path).

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="green", role="base_compute_log_prob")
    def compute_base_log_prob(self, data: DataProto):
        """Compute log probabilities using actor's base model.
        
        This is used for corrected reward computation:
        corrected_reward = old_log_prob - ref_log_prob - (base_log_prob - base_ref_log_prob)
        
        Args:
            data: DataProto containing input_ids, attention_mask, position_ids, responses
            
        Returns:
            DataProto with base_log_prob tensor
        """
        if not self._has_base_model:
            raise ValueError("Base model not initialized. Please set actor_rollout_ref.model.base_model_path in config.")
        
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch
            output, _ = self.base_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"base_log_prob": output})

        output = output.to("cpu")

        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.base_policy.actor_module) == 1:
                self.base_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.base_policy.actor_module) == 2:
                self.base_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="base_ref_compute_log_prob")
    def compute_base_ref_log_prob(self, data: DataProto):
        """Compute log probabilities using ref's base model.
        
        This is used for corrected reward computation:
        corrected_reward = old_log_prob - ref_log_prob - (base_log_prob - base_ref_log_prob)
        
        Args:
            data: DataProto containing ref_input_ids, ref_attention_mask, ref_position_ids, responses
                  (uses ref model inputs if different tokenization is needed)
            
        Returns:
            DataProto with base_ref_log_prob tensor
        """
        if not self._has_base_ref_model:
            raise ValueError("Base ref model not initialized. Please set actor_rollout_ref.ref.model.base_model_path in config.")
        
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch
            output, _ = self.base_ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"base_ref_log_prob": output})

        output = output.to("cpu")

        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.base_ref_policy.actor_module) == 1:
                self.base_ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.base_ref_policy.actor_module) == 2:
                self.base_ref_policy.actor_module.reshard()

        return output

    def has_base_models(self):
        """Check if base models are available for corrected reward computation."""
        return self._has_base_model and self._has_base_ref_model


    def _gather_full_trainable_vector_param(self, param: torch.Tensor, *, expected_shape: tuple[int, ...], name: str):
        vector = param.detach()

        if hasattr(vector, "full_tensor"):
            vector = vector.full_tensor().reshape(expected_shape)
            return vector.cpu() if self.rank == 0 else None

        vector = vector.reshape(-1).contiguous()
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return vector.reshape(expected_shape).cpu() if self.rank == 0 else None

        local_len = torch.tensor([vector.numel()], device=vector.device, dtype=torch.long)
        all_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
        dist.all_gather(all_lens, local_len)
        lens = [int(t.item()) for t in all_lens]

        max_len = max(lens) if lens else 0
        padded = torch.zeros(max_len, dtype=vector.dtype, device=vector.device)
        if vector.numel() > 0:
            padded[: vector.numel()] = vector

        gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)

        if self.rank != 0:
            return None

        chunks = [g[:l].clone() for g, l in zip(gathered, lens)]
        expected_numel = int(np.prod(expected_shape))
        if len(chunks) > 0 and all(c.numel() == expected_numel for c in chunks):
            full_vector = chunks[0]
        elif sum(c.numel() for c in chunks) == expected_numel:
            full_vector = torch.cat(chunks, dim=0)
        else:
            logger.warning(
                "Unexpected gathered steer_vector shapes=%s, expected_numel=%s, name=%s. Fallback to rank-0 local chunk.",
                [c.numel() for c in chunks],
                expected_numel,
                name,
            )
            full_vector = chunks[0]

        return full_vector.reshape(expected_shape).cpu()

    def _collect_trainable_vector_for_save(self):
        # Trainable-vector hook modules are already marked as FSDP ignored states,
        # so each rank keeps a full local copy. Reuse the existing hook payload
        # collection path and avoid extra distributed gathers during save.
        local_payload = _collect_trainable_vector_payload_by_layer(self.actor_module_fsdp)
        if local_payload is not None:
            if self.rank != 0:
                return None

            normalized_payload: dict[int, Any] = {}
            for layer_idx, payload in sorted(local_payload.items()):
                vector = payload.get("vector")
                if vector is None:
                    continue

                vector = vector.detach().cpu() if torch.is_tensor(vector) else torch.as_tensor(vector)
                alpha = payload.get("alpha")
                if alpha is None:
                    normalized_payload[layer_idx] = vector
                    continue

                if torch.is_tensor(alpha):
                    alpha = alpha.reshape(()).detach().cpu()
                else:
                    alpha = torch.tensor(float(alpha))

                normalized_payload[layer_idx] = {
                    "vector": vector,
                    "alpha": alpha,
                    "is_multi": bool(payload.get("is_multi", vector.dim() == 2)),
                }

            return normalized_payload or None

        wrapped = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        actor_module = getattr(wrapped, "module", wrapped)
        trainable_params = [
            (name, p)
            for name, p in actor_module.named_parameters()
            if _is_trainable_vector_param_name(name)
        ]

        if len(trainable_params) == 0:
            return None

        vectors: dict[int, dict[str, Any]] = {}
        fallback_layer_idx = int(self.config.model.get("trainable_token_vector_layer_idx", -1))
        wrapped = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        actor_module = getattr(wrapped, "module", wrapped)
        try:
            hidden_size = get_hidden_size(actor_module)
        except Exception:
            hidden_size = None
        multi_vector_num = int(self.config.model.get("trainable_token_vector_num", 1))
        alpha_by_layer = _collect_trainable_alpha_by_layer(self.actor_module_fsdp)

        for name, vector_param in trainable_params:
            layer_idx = _infer_layer_idx_from_single_vector_key(name)
            if layer_idx is None:
                layer_idx = fallback_layer_idx
            if name.endswith(".alpha"):
                continue
            if layer_idx in vectors:
                raise RuntimeError(f"Duplicated steer_vector on layer {layer_idx}: {name}")
            vectors[layer_idx] = {"name": name, "param": vector_param, "is_multi": _MULTI_VECTOR_HOOK_KEY in name}

        saved: dict[int, torch.Tensor] = {}

        for layer_idx, payload in sorted(vectors.items()):
            if payload["is_multi"]:
                if hidden_size is None:
                    expected_shape = tuple(payload["param"].shape)
                else:
                    expected_shape = (multi_vector_num, hidden_size)
                gathered = self._gather_full_trainable_vector_param(
                    payload["param"],
                    expected_shape=expected_shape,
                    name=payload["name"],
                )
                if self.rank == 0 and gathered is not None:
                    saved[layer_idx] = gathered.reshape(expected_shape)
                continue

            if hidden_size is None:
                expected_shape = (payload["param"].numel(),)
            else:
                expected_shape = (hidden_size,)
            full_vector = self._gather_full_trainable_vector_param(
                payload["param"],
                expected_shape=expected_shape,
                name=payload["name"],
            )
            if self.rank == 0 and full_vector is not None:
                saved[layer_idx] = full_vector.reshape(-1)

        if self.rank != 0:
            return None

        if not saved:
            return None

        payload_by_layer: dict[int, Any] = {}
        for layer_idx, tensor in saved.items():
            alpha = alpha_by_layer.get(layer_idx)
            if alpha is None:
                payload_by_layer[layer_idx] = tensor
                continue

            payload_by_layer[layer_idx] = {
                "vector": tensor,
                "alpha": alpha,
                "is_multi": bool(vectors[layer_idx]["is_multi"]),
            }

        return payload_by_layer

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_trainable_vector(self, save_dir, global_step: int):
        if not self._is_actor:
            return
        if not bool(self.config.model.get("enable_trainable_token_vector", False)):
            return
        if save_dir is None:
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        try:
            vectors_by_layer = self._collect_trainable_vector_for_save()
            if self.rank == 0 and vectors_by_layer:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"vectors_step_{int(global_step):07d}.pt")
                torch.save(vectors_by_layer, save_path)
                logger.warning(
                    "Saved trainable vectors: step=%s, layers=%s, path=%s",
                    int(global_step),
                    sorted(vectors_by_layer.keys()),
                    save_path,
                )
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
