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

import logging
import os

import torch
import torch.nn.functional as F


logger = logging.getLogger(__file__)

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ == "Qwen3MoeLLMForCausalLM":
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader


# vLLM-side single-vector steering support -------------------------------------------------

def _get_vllm_model_layers(model):
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        return None
    if hasattr(inner_model, "model") and hasattr(inner_model.model, "layers"):
        return inner_model.model.layers
    if hasattr(inner_model, "layers"):
        return inner_model.layers
    return None


def _build_vllm_response_token_mask(num_tokens: int, attn_metadata) -> torch.Tensor:
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    if num_tokens <= 0 or attn_metadata is None:
        return mask

    # In vLLM mixed batches, hidden_states are ordered as [prefill_tokens, decode_tokens].
    # Generated tokens correspond to decode_tokens; prefill tokens are prompt-side computation.
    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    if isinstance(num_prefill_tokens, int):
        start = min(max(num_prefill_tokens, 0), num_tokens)
        mask[start:] = True
        return mask

    # vLLM 0.10 may expose (num_prefills, query_start_loc) instead of num_prefill_tokens.
    num_prefills = getattr(attn_metadata, "num_prefills", None)
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if isinstance(num_prefills, int) and query_start_loc is not None:
        if torch.is_tensor(query_start_loc):
            qsl = query_start_loc.tolist()
        else:
            qsl = list(query_start_loc)

        if 0 <= num_prefills < len(qsl):
            start = int(qsl[num_prefills])
            start = min(max(start, 0), num_tokens)
            mask[start:] = True
            return mask

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None:
        # Fallback: if we cannot infer prompt/response boundary, apply to all tokens
        # to avoid silently disabling steer vectors.
        mask[:] = True
        return mask

    if torch.is_tensor(query_start_loc):
        qsl = query_start_loc.tolist()
    else:
        qsl = list(query_start_loc)

    # Best-effort fallback:
    # For query_start_loc=[0, prefill_tokens, ...], boundary is qsl[1].
    if len(qsl) >= 2:
        start = int(qsl[1])
        start = min(max(start, 0), num_tokens)
        mask[start:] = True
        return mask

    # Last resort: apply all tokens rather than disabling the steer vector path.
    mask[:] = True
    return mask


def _sample_positive_simplex_coeffs(num_vectors: int, *, device: torch.device, dtype: torch.dtype, eps: float) -> torch.Tensor:
    coeffs = torch.rand(num_vectors, device=device, dtype=dtype) + eps
    coeff_sum = coeffs.sum()
    if not torch.isfinite(coeff_sum) or coeff_sum <= eps:
        coeffs = torch.ones_like(coeffs)
        coeff_sum = coeffs.sum()
    return coeffs / coeff_sum.clamp_min(eps)


def _sample_positive_hypersphere_coeffs(
    num_vectors: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    coeffs = torch.rand(num_vectors, device=device, dtype=dtype) + eps
    coeff_norm = torch.linalg.vector_norm(coeffs)
    if not torch.isfinite(coeff_norm) or coeff_norm <= eps:
        coeffs = torch.ones_like(coeffs)
        coeff_norm = torch.linalg.vector_norm(coeffs)
    return coeffs / coeff_norm.clamp_min(eps)


def _renormalize_positive_coeffs(coeffs: torch.Tensor, sampling_method: str, *, eps: float) -> torch.Tensor:
    if sampling_method == "interpolation":
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


def _sample_vllm_steer_vector(layer) -> torch.Tensor | None:
    alpha_value = getattr(layer, "_single_trainable_vector_alpha", None)
    vector_scale = float(getattr(layer, "_single_trainable_vector_scale", 1.0))
    curriculum = str(getattr(layer, "_multi_trainable_vector_curriculum", "none"))
    warmup_steps = int(getattr(layer, "_multi_trainable_vector_warmup_steps", 0))
    warmup_end_step = getattr(layer, "_multi_trainable_vector_warmup_end_step", None)
    secondary_freeze_steps = int(getattr(layer, "_multi_trainable_vector_secondary_freeze_steps", 0))
    secondary_end_step = getattr(layer, "_multi_trainable_vector_secondary_end_step", None)
    primary_scale = float(getattr(layer, "_multi_trainable_vector_primary_scale", 1.0))
    secondary_scale = float(getattr(layer, "_multi_trainable_vector_secondary_scale", 1.0))
    global_step = int(getattr(layer, "_single_trainable_vector_global_step", 0))

    if warmup_end_step is not None:
        warmup_end_step = int(warmup_end_step)
    else:
        warmup_end_step = warmup_steps

    if secondary_end_step is not None:
        secondary_end_step = int(secondary_end_step)
    else:
        secondary_end_step = warmup_end_step + secondary_freeze_steps

    steer_vector = getattr(layer, "_single_trainable_vector_steer_vector", None)
    if steer_vector is not None:
        if alpha_value is None:
            return steer_vector

        vector_norm = torch.linalg.vector_norm(steer_vector.float())
        eps = torch.finfo(torch.float32).eps
        if not torch.isfinite(vector_norm) or vector_norm <= eps:
            steer_vector = torch.zeros_like(steer_vector)
            if steer_vector.numel() > 0:
                steer_vector.reshape(-1)[0] = 1
        else:
            steer_vector = steer_vector / vector_norm.clamp_min(eps).to(dtype=steer_vector.dtype)

        alpha = alpha_value.to(device=steer_vector.device, dtype=steer_vector.dtype)
        return steer_vector * alpha

    steer_basis = getattr(layer, "_multi_trainable_vector_steer_basis", None)
    if steer_basis is None:
        return None

    if steer_basis.dim() != 2:
        steer_basis = steer_basis.reshape(steer_basis.size(0), -1)

    sampling_method = getattr(layer, "_multi_trainable_vector_sampling_method", "hypersphere")
    work_dtype = torch.float32
    eps = torch.finfo(work_dtype).eps
    steer_basis = F.normalize(steer_basis.float(), dim=-1).to(dtype=steer_basis.dtype)

    in_warmup = curriculum == "warmup_expand" and steer_basis.size(0) > 1 and global_step <= warmup_end_step

    second_warmup = (
        curriculum == "warmup_expand"
        and steer_basis.size(0) > 1
        and global_step > warmup_end_step
        and secondary_end_step > warmup_end_step
        and global_step <= secondary_end_step
    )

    bias_weights = None
    if curriculum == "warmup_expand" and steer_basis.size(0) > 1 and global_step > warmup_end_step:
        exponents = torch.arange(steer_basis.size(0), device=steer_basis.device, dtype=work_dtype)
        bias_weights = primary_scale / torch.pow(
            torch.tensor(secondary_scale, device=steer_basis.device, dtype=work_dtype),
            exponents,
        )

    if in_warmup:
        coeffs = torch.ones(steer_basis.size(0), device=steer_basis.device, dtype=work_dtype)
    elif second_warmup:
        coeffs = bias_weights
        if coeffs is None or not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
            coeffs = torch.ones(steer_basis.size(0), device=steer_basis.device, dtype=work_dtype)
    elif sampling_method == "interpolation":
        coeffs = _sample_positive_simplex_coeffs(
            steer_basis.size(0),
            device=steer_basis.device,
            dtype=work_dtype,
            eps=eps,
        )
    else:
        coeffs = _sample_positive_hypersphere_coeffs(
            steer_basis.size(0),
            device=steer_basis.device,
            dtype=work_dtype,
            eps=eps,
        )

    if bias_weights is not None and not in_warmup and not second_warmup:
        coeffs = coeffs * bias_weights
        if not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
            coeffs = bias_weights.clone()
        coeffs = _renormalize_positive_coeffs(coeffs, sampling_method, eps=eps)

    coeffs = coeffs.to(dtype=steer_basis.dtype)
    vector = torch.sum(steer_basis * coeffs.unsqueeze(-1), dim=0)
    if curriculum == "warmup_expand" and global_step > warmup_end_step and global_step <= warmup_end_step + 1:
        logger.warning(
            "vLLM steer curriculum activated: step=%s, uniform_mix_end_step=%s, second_phase_end_step=%s, "
            "primary_scale=%.6f, secondary_scale=%.6f",
            global_step,
            warmup_end_step,
            secondary_end_step,
            primary_scale,
            secondary_scale,
        )
    if not in_warmup:
        vector_norm = torch.linalg.vector_norm(vector.float())
        if not torch.isfinite(vector_norm) or vector_norm <= eps:
            vector = steer_basis[0]
            vector_norm = torch.linalg.vector_norm(vector.float())
        vector = vector / vector_norm.clamp_min(eps).to(dtype=vector.dtype)
    if alpha_value is None:
        alpha = torch.tensor(vector_scale, device=vector.device, dtype=vector.dtype)
    else:
        alpha = alpha_value.to(device=vector.device, dtype=vector.dtype)
    return vector * alpha


def _single_vector_layer_forward(self, positions, hidden_states, residual):
    hidden_states, residual = self._verl_original_forward(positions, hidden_states, residual)

    steer_vector = _sample_vllm_steer_vector(self)
    if steer_vector is None:
        return hidden_states, residual

    delta = steer_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
    view_shape = [1] * hidden_states.dim()
    view_shape[-1] = -1
    delta_view = delta.view(*view_shape)

    debug_once = bool(getattr(self, "_single_trainable_vector_debug_once", False)) and not bool(
        getattr(self, "_single_trainable_vector_debug_printed", False)
    )
    hidden_states_before = None
    if debug_once:
        hidden_states_before = hidden_states.detach()

    token_mask = None
    if self._single_trainable_vector_response_only:
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        attn_metadata = forward_ctx.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = next((v for v in attn_metadata.values() if v is not None), None)

        token_mask = _build_vllm_response_token_mask(hidden_states.size(0), attn_metadata)
        token_mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)

    if token_mask is None:
        hidden_states = hidden_states + delta_view
    else:
        hidden_states = hidden_states + delta_view * token_mask

    if debug_once and hidden_states_before is not None:
        with torch.no_grad():
            before = hidden_states_before.reshape(-1, hidden_states_before.size(-1)).float()
            after = hidden_states.detach().reshape(-1, hidden_states.size(-1)).float()
            diff = after - before
            sample_dim = min(8, before.size(-1))

            delta_flat = delta.float().reshape(-1)
            delta_sample = delta_flat[:sample_dim].detach().cpu().tolist()
            before_sample = before[0, :sample_dim].detach().cpu().tolist()
            after_sample = after[0, :sample_dim].detach().cpu().tolist()
            diff_sample = diff[0, :sample_dim].detach().cpu().tolist()

            if token_mask is None:
                mask_ratio = 1.0
            else:
                mask_ratio = float((token_mask > 0).float().mean().item())

            logger.warning(
                "vLLM steer forward debug: layer=%s, response_only=%s, token_mask_ratio=%.6f, "
                "delta_mean=%.6f, delta_std=%.6f, before0[:%d]=%s, after0[:%d]=%s, diff0[:%d]=%s, delta[:%d]=%s",
                getattr(self, "_single_trainable_vector_layer_idx", None),
                bool(getattr(self, "_single_trainable_vector_response_only", True)),
                mask_ratio,
                float(delta_flat.mean().item()),
                float(delta_flat.std().item()),
                sample_dim,
                before_sample,
                sample_dim,
                after_sample,
                sample_dim,
                diff_sample,
                sample_dim,
                delta_sample,
            )

        self._single_trainable_vector_debug_printed = True

    return hidden_states, residual


def patch_vllm_single_vector_hook(
    model,
    *,
    steer_vector: torch.Tensor,
    layer_idx: int,
    response_only: bool = True,
    sampling_method: str = "hypersphere",
    vector_scale: float = 1.0,
    is_multi: bool = False,
    alpha_value: torch.Tensor | None = None,
    curriculum: str = "none",
    warmup_steps: int = 0,
    warmup_end_step: int | None = None,
    secondary_freeze_steps: int = 0,
    secondary_end_step: int | None = None,
    primary_scale: float = 1.0,
    secondary_scale: float = 1.0,
    global_step: int = 0,
    debug_once: bool | None = None,
):
    layers = _get_vllm_model_layers(model)
    if layers is None:
        raise RuntimeError("Cannot find decoder layers in vLLM model for single-vector hook patching.")

    num_layers = len(layers)
    if layer_idx < 0:
        layer_idx += num_layers
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx={layer_idx}, model has {num_layers} layers")

    layer = layers[layer_idx]
    if not hasattr(layer, "_verl_original_forward"):
        layer._verl_original_forward = layer.forward
        layer.forward = _single_vector_layer_forward.__get__(layer, layer.__class__)

    vector = steer_vector.detach().clone()
    if vector.dim() == 1:
        vector = vector.reshape(-1)
    elif vector.dim() == 2:
        vector = vector.reshape(vector.size(0), -1)
    else:
        raise ValueError(f"steer_vector must be 1D or 2D, got shape={tuple(vector.shape)}")

    ref_param = next(layer.parameters(), None)
    if ref_param is not None:
        vector = vector.to(device=ref_param.device, dtype=ref_param.dtype)
        if alpha_value is not None:
            alpha_value = alpha_value.to(device=ref_param.device, dtype=ref_param.dtype)

    if vector.dim() == 1 and not is_multi:
        layer._single_trainable_vector_steer_vector = vector
        if hasattr(layer, "_multi_trainable_vector_steer_basis"):
            delattr(layer, "_multi_trainable_vector_steer_basis")
        if hasattr(layer, "_multi_trainable_vector_sampling_method"):
            delattr(layer, "_multi_trainable_vector_sampling_method")
    else:
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        layer._multi_trainable_vector_steer_basis = vector
        if hasattr(layer, "_single_trainable_vector_steer_vector"):
            delattr(layer, "_single_trainable_vector_steer_vector")
        layer._multi_trainable_vector_sampling_method = sampling_method
        layer._multi_trainable_vector_curriculum = curriculum
        layer._multi_trainable_vector_warmup_steps = int(warmup_steps)
        layer._multi_trainable_vector_warmup_end_step = None if warmup_end_step is None else int(warmup_end_step)
        layer._multi_trainable_vector_secondary_freeze_steps = int(secondary_freeze_steps)
        layer._multi_trainable_vector_secondary_end_step = None if secondary_end_step is None else int(secondary_end_step)
        layer._multi_trainable_vector_primary_scale = float(primary_scale)
        layer._multi_trainable_vector_secondary_scale = float(secondary_scale)
    layer._single_trainable_vector_response_only = bool(response_only)
    layer._single_trainable_vector_layer_idx = int(layer_idx)
    layer._single_trainable_vector_scale = float(vector_scale)
    layer._single_trainable_vector_alpha = alpha_value
    layer._single_trainable_vector_global_step = int(global_step)

    if debug_once is None:
        debug_once = str(os.getenv("VERL_VLLM_STEER_DEBUG_ONCE", "0")).lower() in ("1", "true", "yes", "y", "on")
    layer._single_trainable_vector_debug_once = bool(debug_once)
    layer._single_trainable_vector_debug_printed = False


def get_vllm_single_vector_debug_info(model) -> dict:
    layers = _get_vllm_model_layers(model)
    if layers is None:
        return {"num_layers": 0, "patched_layers": [], "vector_stats": {}}

    patched_layers: list[int] = []
    vector_stats: dict[int, dict[str, float]] = {}
    for idx, layer in enumerate(layers):
        vec = getattr(layer, "_single_trainable_vector_steer_vector", None)
        basis = getattr(layer, "_multi_trainable_vector_steer_basis", None)
        if vec is None and basis is None:
            continue
        patched_layers.append(idx)
        payload = vec if vec is not None else basis
        stats = {
            "mean": float(payload.float().mean().item()),
            "std": float(payload.float().std().item()),
            "min": float(payload.float().min().item()),
            "max": float(payload.float().max().item()),
        }
        alpha = getattr(layer, "_single_trainable_vector_alpha", None)
        if alpha is not None:
            stats["alpha"] = float(alpha.float().item())
        if basis is not None:
            stats["num_vectors"] = int(basis.size(0))
        vector_stats[idx] = stats

    return {
        "num_layers": len(layers),
        "patched_layers": patched_layers,
        "vector_stats": vector_stats,
    }
