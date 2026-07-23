# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.fsdp_workers import (
    _cache_trainable_alpha_from_scalar,
    _get_trainable_alpha_sync_group,
    _materialize_trainable_alpha,
)
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_STEER_VECTOR_PARAM_SUFFIXES = (
    "_single_trainable_vector_hook.steer_vector",
    "_single_trainable_vector_hook.alpha",
    "_multi_trainable_vector_hook.basis_vectors",
    "_multi_trainable_vector_hook.alpha",
)


def _is_trainable_vector_param_name(name: str) -> bool:
    return any(suffix in name for suffix in _STEER_VECTOR_PARAM_SUFFIXES)


def _unwrap_model(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _get_single_vector_hooks(module: nn.Module):
    layers = None
    if hasattr(module, "model") and hasattr(module.model, "layers"):
        layers = module.model.layers
    elif hasattr(module, "gpt_neox") and hasattr(module.gpt_neox, "layers"):
        layers = module.gpt_neox.layers
    elif hasattr(module, "transformer") and hasattr(module.transformer, "h"):
        layers = module.transformer.h

    if layers is None:
        return []

    hooks = []
    for layer in layers:
        single_hook = getattr(layer, "_single_trainable_vector_hook", None)
        multi_hook = getattr(layer, "_multi_trainable_vector_hook", None)
        if single_hook is not None:
            hooks.append(single_hook)
        if multi_hook is not None:
            hooks.append(multi_hook)
    return hooks


def _orthogonalize_trainable_vector_hooks(actor_module: nn.Module) -> None:
    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        orth_fn = getattr(hook, "orthogonalize_basis", None)
        if orth_fn is not None:
            orth_fn()


def _resample_trainable_vector_hooks(actor_module: nn.Module) -> None:
    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        resample_fn = getattr(hook, "resample_cached_coefficients", None)
        if resample_fn is not None:
            resample_fn()


def _project_trainable_vector_grads(actor_module: nn.Module) -> None:
    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        project_fn = getattr(hook, "project_gradients_to_tangent", None)
        if project_fn is not None:
            project_fn()


def _allreduce_trainable_vector_grads(actor_module: nn.Module) -> None:
    """Average trainable-vector gradients across data-parallel ranks.

    Trainable-vector hooks are passed to FSDP via ``ignored_states`` so they stay
    as ordinary (unsharded) parameters. The flip side is that FSDP never reduces
    their gradients: each rank only holds the local gradient computed on its own
    micro-batch shard. Without this all-reduce the subsequent
    ``_broadcast_trainable_vector_state`` overwrites every rank with rank-0's
    parameters, so 1/world_size of the data effectively drives the update and the
    vector barely moves. Averaging the gradients here restores the full-batch
    signal and keeps every rank consistent before the optimizer step.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    group = _get_trainable_alpha_sync_group()
    world_size = dist.get_world_size()

    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        for pname in ("steer_vector", "basis_vectors", "alpha"):
            param = getattr(hook, pname, None)
            if param is None or param.grad is None:
                continue

            grad = param.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()

            # gloo group operates on CPU tensors (mirrors _broadcast_trainable_vector_state)
            payload = grad.detach().float().cpu().contiguous()
            dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=group)
            payload.div_(world_size)
            grad.data.copy_(payload.to(device=grad.device, dtype=grad.dtype))


def _broadcast_trainable_vector_state(actor_module: nn.Module) -> None:
    group = _get_trainable_alpha_sync_group() if dist.is_initialized() else None
    rank = dist.get_rank() if dist.is_initialized() else 0

    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        if getattr(hook, "num_vectors", 1) == 1:
            steer_vector = getattr(hook, "steer_vector", None)
            if steer_vector is not None:
                if rank == 0:
                    payload = steer_vector.detach().float().cpu()
                else:
                    payload = torch.empty_like(steer_vector.detach().float().cpu())
                if dist.is_initialized():
                    dist.broadcast(payload, src=0, group=group)
                steer_vector.data.copy_(payload.to(device=steer_vector.device, dtype=steer_vector.dtype))
        else:
            basis_vectors = getattr(hook, "basis_vectors", None)
            if basis_vectors is not None:
                if rank == 0:
                    payload = basis_vectors.detach().float().cpu()
                else:
                    payload = torch.empty_like(basis_vectors.detach().float().cpu())
                if dist.is_initialized():
                    dist.broadcast(payload, src=0, group=group)
                basis_vectors.data.copy_(payload.to(device=basis_vectors.device, dtype=basis_vectors.dtype))

        alpha_param = getattr(hook, "alpha", None)
        if alpha_param is None:
            continue
        if rank == 0:
            alpha_payload = _materialize_trainable_alpha(alpha_param).detach().reshape(1).float().cpu()
        else:
            alpha_payload = torch.zeros(1, dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(alpha_payload, src=0, group=group)
        alpha_param.data = alpha_payload.to(device=alpha_param.device, dtype=alpha_param.dtype)
        _cache_trainable_alpha_from_scalar(alpha_param, alpha_payload[0])


def _set_trainable_vector_global_step(actor_module: nn.Module, global_step: int | None) -> None:
    for hook in _get_single_vector_hooks(_unwrap_model(actor_module)):
        set_step_fn = getattr(hook, "set_current_global_step", None)
        if set_step_fn is not None:
            set_step_fn(global_step)
        activate_fn = getattr(hook, "maybe_activate_expanded_basis", None)
        if activate_fn is not None:
            activate_fn()


@contextmanager
def _response_only_steer_vector_context(actor_module: nn.Module, response_token_mask: torch.Tensor):
    hooks = _get_single_vector_hooks(_unwrap_model(actor_module))
    if not hooks:
        yield
        return

    module = _unwrap_model(actor_module)
    force_all_tokens = bool(getattr(module, "_single_trainable_vector_force_all_tokens", False))
    _resample_trainable_vector_hooks(actor_module)
    if force_all_tokens:
        for hook in hooks:
            hook.set_response_token_mask(None)
        yield
        return

    mask = response_token_mask
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)

    # Keep mask on module after forward so torch checkpoint recomputation in backward
    # sees identical hook state and graph structure.
    for hook in hooks:
        hook.set_response_token_mask(mask)
    yield


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        self._trainable_vector_global_step = 0
        _set_trainable_vector_global_step(self.actor_module, self._trainable_vector_global_step)

    def set_trainable_vector_global_step(self, global_step: int | None) -> None:
        self._trainable_vector_global_step = 0 if global_step is None else max(int(global_step), 0)
        _set_trainable_vector_global_step(self.actor_module, self._trainable_vector_global_step)

    def _build_full_response_logprob_token_mask(
        self, attention_mask: torch.Tensor, response_length: int
    ) -> torch.Tensor:
        """Mask the positions whose logits are used to score response tokens.

        For causal LM training, response token log-probs come from the previous
        positions: logits[:, -response_length-1:-1]. The steer hook needs to be
        applied on those predictor states instead of the response token states
        themselves, otherwise the hook is shifted by one token and directly
        perturbs an extra trailing position that is not optimized.
        """

        response_logprob_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if response_length <= 0:
            return response_logprob_token_mask

        seq_len = attention_mask.size(-1)
        start = max(seq_len - response_length - 1, 0)
        end = max(seq_len - 1, 0)
        if end > start:
            response_logprob_token_mask[:, start:end] = attention_mask[:, start:end].to(dtype=torch.bool)

        return response_logprob_token_mask

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            full_response_token_mask = self._build_full_response_logprob_token_mask(attention_mask, response_length)

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                response_token_mask_rmpad = index_first_axis(
                    rearrange(full_response_token_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
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

                    response_token_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        response_token_mask_rmpad.to(dtype=input_ids_rmpad.dtype),
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    response_token_mask_rmpad = response_token_mask_rmpad.to(dtype=torch.bool)

                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                with _response_only_steer_vector_context(
                    self.actor_module,
                    response_token_mask_rmpad,
                ):
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
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
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
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
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                with _response_only_steer_vector_context(
                    self.actor_module,
                    full_response_token_mask,
                ):
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _collect_steer_grad_metrics(self) -> dict[str, float]:
        grad_norms: list[float] = []
        vec_norms: list[float] = []
        nonzero_count = 0
        none_count = 0
        param_count = 0

        for name, param in self.actor_module.named_parameters():
            if not _is_trainable_vector_param_name(name):
                continue

            param_count += 1

            # Track the parameter's own L2 norm (direction/basis vectors, not alpha),
            # so training progress is visible even when grads are averaged elsewhere.
            if not name.endswith(".alpha"):
                p_local = param.detach()
                if isinstance(p_local, DTensor):
                    p_local = p_local.to_local()
                vec_norms.append(float(p_local.float().norm().item()))

            grad = param.grad
            if grad is None:
                none_count += 1
                continue

            if isinstance(grad, DTensor):
                grad = grad.to_local()

            grad_norm = float(grad.detach().float().norm().item())
            grad_norms.append(grad_norm)
            if grad_norm > 0:
                nonzero_count += 1

        if param_count == 0:
            return {}

        if grad_norms:
            mean_norm = float(sum(grad_norms) / len(grad_norms))
            max_norm = float(max(grad_norms))
            min_norm = float(min(grad_norms))
        else:
            mean_norm = 0.0
            max_norm = 0.0
            min_norm = 0.0

        metrics = {
            "actor/steer_grad_norm_mean": mean_norm,
            "actor/steer_grad_norm_max": max_norm,
            "actor/steer_grad_norm_min": min_norm,
            "actor/steer_grad_nonzero_frac": float(nonzero_count / max(param_count, 1)),
            "actor/steer_grad_none_count": float(none_count),
            "actor/steer_grad_param_count": float(param_count),
        }

        if vec_norms:
            metrics["actor/steer_vec_norm_mean"] = float(sum(vec_norms) / len(vec_norms))
            metrics["actor/steer_vec_norm_max"] = float(max(vec_norms))
            metrics["actor/steer_vec_norm_min"] = float(min(vec_norms))

        return metrics

    def _optimizer_step(self):
        _set_trainable_vector_global_step(self.actor_module, self._trainable_vector_global_step)
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)

        steer_grad_metrics = self._collect_steer_grad_metrics()
        # FSDP-ignored trainable-vector grads are not reduced by FSDP; average them
        # across DP ranks before projection/clipping so the update uses the full batch.
        _allreduce_trainable_vector_grads(self.actor_module)
        _project_trainable_vector_grads(self.actor_module)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        stepped = False
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            stepped = True
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
                stepped = True

        if stepped:
            _orthogonalize_trainable_vector_hooks(self.actor_module)
            _broadcast_trainable_vector_state(self.actor_module)
        return grad_norm, steer_grad_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        self.set_trainable_vector_global_step(data.meta_info.get("global_step", self._trainable_vector_global_step))
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        self.set_trainable_vector_global_step(data.meta_info.get("global_step", self._trainable_vector_global_step))

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = self.config.policy_loss.lambda_vals

                            if self.config.policy_loss.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                        advantages = (- (reverse_kl))
                    advantages = torch.clamp(advantages, min=-2, max=2)
                    # if model_inputs["advantages"].sum() < 0:
                    #     advantages[:,:] = 0 
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov

                    # breakpoint()
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, steer_grad_metrics = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                mini_batch_metrics.update(steer_grad_metrics)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
