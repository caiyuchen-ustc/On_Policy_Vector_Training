"""Shared curriculum-stage computation and deterministic coefficient sampling for
trainable token-vector (OPD) steering.

This module is the SINGLE SOURCE OF TRUTH used by both:
  - the FSDP training side (verl/workers/fsdp_workers.py, TrainableTokenVectorHook)
  - the vLLM rollout side (verl/utils/vllm/patch.py)

Why it exists
-------------
Multi-vector steering samples a random combination of basis vectors each forward.
Previously every forward pass drew *fresh* unseeded ``torch.rand`` coefficients, so
the vector injected during rollout, during old-logprob computation, and during each
new-logprob update all differed. That poisons the PPO importance ratio (the ratio is
supposed to compare the same conditional policy) and breaks the on-policy assumption.

To fix it we make coefficient sampling a deterministic function of
``(global_step, layer_idx, num_vectors)`` via a seeded ``torch.Generator``. Because
all three phases (rollout / old-logp / new-logp) share the same ``global_step`` for a
given training iteration and the same ``layer_idx`` / ``num_vectors``, they now draw
byte-identical coefficients. The next iteration uses a new ``global_step`` and thus a
new coefficient set. Each hooked layer gets an independent coefficient set (seed
depends on ``layer_idx``).

All sampling happens on CPU in float32 to guarantee bit-identical results across the
FSDP process and the vLLM process regardless of GPU/dtype; callers move the result to
their device/dtype.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

# Fixed primes for combining (global_step, layer_idx, num_vectors) into a stable seed.
# NOT Python's built-in hash() (which is salted per-process and would break cross-process
# determinism). manual_seed accepts a 64-bit value; we keep it well within range.
_SEED_PRIME_STEP = 1000003
_SEED_PRIME_LAYER = 19349663
_SEED_PRIME_NUM = 83492791
_SEED_MOD = (1 << 63) - 1

# curricula that use the progressive-doubling stage machinery (1 -> 2 -> 4 -> ... -> N)
_PD_CURRICULA = ("progressive_double", "raw_then_double")


class StageInfo(NamedTuple):
    stage: int          # 0-based doubling stage, capped at final stage
    active: int         # number of basis vectors used for sampling = min(2**stage, N)
    prev_active: int    # start of the "newly added" block for this stage (0 in stage 0)
    ramp_g: float       # in-stage ramp coefficient g: 1 -> 0 (old subspace dominance fades)
    in_raw_phase: bool  # raw_then_double stage-0 raw phase (basis[0] used raw, no norm/alpha)


def make_generator(global_step: int, layer_idx: int, num_vectors: int) -> torch.Generator:
    """Deterministic CPU generator keyed by (global_step, layer_idx, num_vectors)."""
    gs = max(int(global_step), 0)
    li = max(int(layer_idx), 0)
    nv = max(int(num_vectors), 1)
    seed = (gs * _SEED_PRIME_STEP + li * _SEED_PRIME_LAYER + nv * _SEED_PRIME_NUM) % _SEED_MOD
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _pd_stage_steps(warmup_steps: int) -> int:
    return max(1, int(warmup_steps))


def _pd_num_stages(num_vectors: int) -> int:
    # how many doubling stages to cover N: active = 1, 2, 4, ..., >= N
    n, stages = 1, 1
    while n < num_vectors:
        n *= 2
        stages += 1
    return stages


def stage_info(
    global_step: int,
    *,
    curriculum: str,
    warmup_steps: int,
    num_vectors: int,
    secondary_freeze_steps: int = 0,
) -> StageInfo:
    """Replicates the FSDP-side _pd_stage/_pd_active/_pd_prev_active/_pd_ramp_g/
    _rtd_in_raw_phase logic (fsdp_workers.py:807-850) so both sides agree on the stage.
    """
    pd_enabled = curriculum in _PD_CURRICULA and num_vectors > 1
    rtd_enabled = curriculum == "raw_then_double" and num_vectors > 1
    step = max(int(global_step), 0)
    stage_steps = _pd_stage_steps(warmup_steps)

    if not pd_enabled:
        return StageInfo(stage=0, active=num_vectors, prev_active=0, ramp_g=0.0, in_raw_phase=False)

    num_stages = _pd_num_stages(num_vectors)
    stage = int(min(step // stage_steps, num_stages - 1))
    active = int(min(2 ** stage, num_vectors))
    prev_active = 0 if stage == 0 else int(min(2 ** (stage - 1), num_vectors))
    in_raw_phase = bool(rtd_enabled and step < stage_steps)

    # in-stage ramp g: for stage>=1, first S_ramp steps ramp g from 1 -> 0; else 0.
    ramp_g = 0.0
    if stage >= 1:
        s_ramp = int(secondary_freeze_steps)
        if s_ramp > 0:
            step_in_stage = step - stage * stage_steps
            if step_in_stage < s_ramp:
                ramp_g = 1.0 - float(step_in_stage) / float(s_ramp)

    return StageInfo(stage=stage, active=active, prev_active=prev_active, ramp_g=ramp_g, in_raw_phase=in_raw_phase)


# ---- primitive coefficient samplers (seeded, CPU float32) ----

def _sample_simplex(n: int, g: torch.Generator, eps: float) -> torch.Tensor:
    coeffs = torch.rand(n, generator=g, dtype=torch.float32) + eps
    coeff_sum = coeffs.sum()
    if not torch.isfinite(coeff_sum) or coeff_sum <= eps:
        coeffs = torch.ones_like(coeffs)
        coeff_sum = coeffs.sum()
    return coeffs / coeff_sum.clamp_min(eps)


def _sample_hypersphere(n: int, g: torch.Generator, eps: float) -> torch.Tensor:
    coeffs = torch.rand(n, generator=g, dtype=torch.float32) + eps
    coeff_norm = torch.linalg.vector_norm(coeffs)
    if not torch.isfinite(coeff_norm) or coeff_norm <= eps:
        coeffs = torch.ones_like(coeffs)
        coeff_norm = torch.linalg.vector_norm(coeffs)
    return coeffs / coeff_norm.clamp_min(eps)


def _renormalize(coeffs: torch.Tensor, sampling_method: str, eps: float) -> torch.Tensor:
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


def _rand(n: int, sampling_method: str, g: torch.Generator, eps: float) -> torch.Tensor:
    if sampling_method == "interpolation":
        return _sample_simplex(n, g, eps)
    return _sample_hypersphere(n, g, eps)


def _effective_warmup_end(warmup_steps: int, warmup_end_step) -> int:
    return int(warmup_steps) if warmup_end_step is None else int(warmup_end_step)


def _effective_secondary_end(warmup_steps: int, warmup_end_step, secondary_freeze_steps: int, secondary_end_step) -> int:
    if secondary_end_step is not None:
        return int(secondary_end_step)
    return _effective_warmup_end(warmup_steps, warmup_end_step) + int(secondary_freeze_steps)


def _bias_weights(n: int, primary_scale: float, secondary_scale: float) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    return float(primary_scale) / torch.pow(torch.tensor(float(secondary_scale), dtype=torch.float32), idx)


def _atb_ramp_g(global_step: int, warmup_steps: int, secondary_freeze_steps: int) -> float:
    # alpha_then_basis: phase1 (step<t1) g=1; phase2 ramp g:1->0 over t2; phase3 g=0
    t1, t2 = int(warmup_steps), int(secondary_freeze_steps)
    step = max(int(global_step), 0)
    if step < t1:
        return 1.0
    if t2 <= 0 or step >= t1 + t2:
        return 0.0
    return 1.0 - float(step - t1) / float(t2)


def sample_coefficients(
    global_step: int,
    layer_idx: int,
    *,
    num_vectors: int,
    curriculum: str,
    sampling_method: str = "hypersphere",
    warmup_steps: int = 0,
    warmup_end_step=None,
    secondary_freeze_steps: int = 0,
    secondary_end_step=None,
    primary_scale: float = 1.0,
    secondary_scale: float = 1.0,
) -> torch.Tensor:
    """Deterministically sample the basis-combination coefficients for one hook layer.

    Returns a CPU float32 tensor of shape [num_vectors]. Callers move it to their
    device/dtype. Bit-identical for identical arguments — in particular identical
    (global_step, layer_idx, num_vectors) — so the vector injected during rollout
    (vLLM), during old-logprob, and during each new-logprob update match exactly.

    This is the SINGLE implementation of ALL multi-vector curricula
    (none / warmup_expand / alpha_then_basis / progressive_double / raw_then_double);
    both the FSDP side (fsdp_workers.py _sample_coefficients) and the vLLM side
    (patch.py _sample_vllm_steer_vector) delegate here so they never diverge.
    """
    eps = torch.finfo(torch.float32).eps
    n = int(num_vectors)
    g = make_generator(global_step, layer_idx, n)
    step = max(int(global_step), 0)

    # ---- progressive_double / raw_then_double ----
    if curriculum in _PD_CURRICULA and n > 1:
        info = stage_info(
            global_step,
            curriculum=curriculum,
            warmup_steps=warmup_steps,
            num_vectors=n,
            secondary_freeze_steps=secondary_freeze_steps,
        )
        coeffs = torch.zeros(n, dtype=torch.float32)
        active = info.active
        if active <= 1:
            coeffs[0] = 1.0  # stage 0: one-hot -> only basis[0], all others * 0
            return coeffs
        ramp = info.ramp_g
        if ramp <= 0.0:
            # 随机采样期：在整个当前子空间 [0, active) 上随机取一个单位方向。
            sub = _rand(active, sampling_method, g, eps)
        else:
            # ramp 淡入期（每个倍增阶段前 S_ramp 步）：向量空间干净凸组合——
            #   old_unit = 旧子空间 [0, prev) 的随机组合、归一化成 norm=1；
            #   new_unit = 新增块 [prev, active) 的随机组合、归一化成 norm=1（不含旧方向分量）；
            #   mix = ramp*old_unit + (1-ramp)*new_unit，再整体归一化。
            # 例：stage1 ramp=0.9 -> 0.9*basis[0] + 0.1*(新增块单位向量)，归一化后 *alpha。
            # 因基正交，系数空间的该组合等价于向量空间组合。
            prev = info.prev_active
            old_unit = torch.zeros(active, dtype=torch.float32)
            old_unit[:prev] = _rand(prev, sampling_method, g, eps)
            new_unit = torch.zeros(active, dtype=torch.float32)
            new_unit[prev:active] = _rand(active - prev, sampling_method, g, eps)
            mix = ramp * old_unit + (1.0 - ramp) * new_unit
            sub = _renormalize(mix, sampling_method, eps)
        coeffs[:active] = sub
        return coeffs

    # ---- alpha_then_basis ----
    if curriculum == "alpha_then_basis" and n > 1:
        onehot = torch.zeros(n, dtype=torch.float32)
        onehot[0] = 1.0
        if step < int(warmup_steps):
            return onehot  # phase 1: only basis[0]
        ramp = _atb_ramp_g(step, warmup_steps, secondary_freeze_steps)
        rand_c = _rand(n, sampling_method, g, eps)
        if ramp <= 0.0:
            return rand_c  # phase 3: pure random
        coeffs = ramp * onehot + (1.0 - ramp) * rand_c  # phase 2: smooth transition
        return _renormalize(coeffs, sampling_method, eps)

    # ---- warmup_expand ----
    if curriculum == "warmup_expand" and n > 1:
        w_end = _effective_warmup_end(warmup_steps, warmup_end_step)
        s_end = _effective_secondary_end(warmup_steps, warmup_end_step, secondary_freeze_steps, secondary_end_step)
        if step <= w_end:
            return torch.ones(n, dtype=torch.float32)  # uniform warmup
        bias = _bias_weights(n, primary_scale, secondary_scale)
        if w_end < step <= s_end and s_end > w_end:
            coeffs = bias
            if not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
                coeffs = torch.ones(n, dtype=torch.float32)
            return coeffs
        coeffs = _rand(n, sampling_method, g, eps)
        coeffs = coeffs * bias
        if not torch.isfinite(coeffs).all() or torch.all(coeffs <= eps):
            coeffs = bias.clone()
        return _renormalize(coeffs, sampling_method, eps)

    # ---- none (and any multi with no curriculum) ----
    return _rand(n, sampling_method, g, eps)

