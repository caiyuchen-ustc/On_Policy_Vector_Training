# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig"]


@dataclass
class HFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # specify the base model path for the model. This is useful when we want to load a model with a different base model in on-policy distillation.
    base_model_path: Optional[str] = None

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = False

    # lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[str] = "all-linear"

    exclude_modules: Optional[str] = None

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    architectures: Optional[list[str]] = None
    enable_trainable_token_vector: bool = False
    trainable_token_vector_mode: str = "single"
    trainable_token_vector_num: int = 1
    trainable_token_vector_sampling_method: str = "hypersphere"
    trainable_token_vector_scale: float = 1.0
    trainable_token_vector_learnable_alpha: bool = False
    trainable_token_vector_alpha_init: float = 0.0
    trainable_token_vector_curriculum: str = "none"
    trainable_token_vector_warmup_steps: int = 0
    trainable_token_vector_warmup_end_step: Optional[int] = None
    trainable_token_vector_secondary_freeze_steps: int = 0
    trainable_token_vector_secondary_end_step: Optional[int] = None
    trainable_token_vector_primary_scale: float = 1.0
    trainable_token_vector_secondary_scale: float = 1.05
    trainable_token_vector_freeze_primary_after_warmup: bool = False
    trainable_token_vector_layer_idx: int = 16
    trainable_token_vector_layer_start: Optional[int] = None
    trainable_token_vector_layer_end: Optional[int] = None
    trainable_token_vector_all_layers: bool = False
    trainable_token_vector_force_all_tokens: bool = False
    train_q_bias_only: bool = False
    q_bias_layer_start: Optional[int] = None
    q_bias_layer_end: Optional[int] = None
    train_single_layer_only: bool = False
    train_layer_idx: Optional[int] = None
    train_single_mlp_only: bool = False
    train_mlp_layer_idx: Optional[int] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # constuct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        if self.custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = self.custom_chat_template
            else:
                self.tokenizer.chat_template = self.custom_chat_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # constuct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        if self.trainable_token_vector_mode not in {"single", "multi"}:
            raise ValueError(
                f"trainable_token_vector_mode must be 'single' or 'multi', got {self.trainable_token_vector_mode!r}"
            )
        if self.trainable_token_vector_num < 1:
            raise ValueError(
                f"trainable_token_vector_num must be >= 1, got {self.trainable_token_vector_num}"
            )
        if self.trainable_token_vector_sampling_method not in {"hypersphere", "interpolation"}:
            raise ValueError(
                "trainable_token_vector_sampling_method must be 'hypersphere' or 'interpolation', "
                f"got {self.trainable_token_vector_sampling_method!r}"
            )
        if self.trainable_token_vector_scale <= 0:
            raise ValueError(
                f"trainable_token_vector_scale must be > 0, got {self.trainable_token_vector_scale}"
            )
        if self.trainable_token_vector_curriculum not in {"none", "warmup_expand"}:
            raise ValueError(
                "trainable_token_vector_curriculum must be 'none' or 'warmup_expand', "
                f"got {self.trainable_token_vector_curriculum!r}"
            )
        if self.trainable_token_vector_warmup_steps < 0:
            raise ValueError(
                f"trainable_token_vector_warmup_steps must be >= 0, got {self.trainable_token_vector_warmup_steps}"
            )
        if self.trainable_token_vector_warmup_end_step is not None and self.trainable_token_vector_warmup_end_step < 0:
            raise ValueError(
                "trainable_token_vector_warmup_end_step must be >= 0 when set, "
                f"got {self.trainable_token_vector_warmup_end_step}"
            )
        if self.trainable_token_vector_secondary_freeze_steps < 0:
            raise ValueError(
                "trainable_token_vector_secondary_freeze_steps must be >= 0, "
                f"got {self.trainable_token_vector_secondary_freeze_steps}"
            )
        if self.trainable_token_vector_secondary_end_step is not None and self.trainable_token_vector_secondary_end_step < 0:
            raise ValueError(
                "trainable_token_vector_secondary_end_step must be >= 0 when set, "
                f"got {self.trainable_token_vector_secondary_end_step}"
            )
        if (
            self.trainable_token_vector_warmup_end_step is not None
            and self.trainable_token_vector_secondary_end_step is not None
            and self.trainable_token_vector_secondary_end_step < self.trainable_token_vector_warmup_end_step
        ):
            raise ValueError(
                "trainable_token_vector_secondary_end_step must be >= trainable_token_vector_warmup_end_step, "
                f"got {self.trainable_token_vector_secondary_end_step} < {self.trainable_token_vector_warmup_end_step}"
            )
        if self.trainable_token_vector_primary_scale <= 0:
            raise ValueError(
                f"trainable_token_vector_primary_scale must be > 0, got {self.trainable_token_vector_primary_scale}"
            )
        if self.trainable_token_vector_secondary_scale <= 0:
            raise ValueError(
                f"trainable_token_vector_secondary_scale must be > 0, got {self.trainable_token_vector_secondary_scale}"
            )
        if self.trainable_token_vector_mode == "single":
            if self.trainable_token_vector_num != 1:
                raise ValueError(
                    "trainable_token_vector_mode='single' requires trainable_token_vector_num == 1"
                )
        elif self.trainable_token_vector_num < 2:
            raise ValueError(
                "trainable_token_vector_mode='multi' requires trainable_token_vector_num >= 2"
            )

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # get model architectures
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )

        # per model patch
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
