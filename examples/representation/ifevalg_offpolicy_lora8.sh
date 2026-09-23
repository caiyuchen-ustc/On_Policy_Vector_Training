#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# LoRA(rank=8) OFF-POLICY distillation, IFEvalG-style.
#
# The frozen teacher generates the rollouts (vLLM serves the teacher checkpoint,
# the per-step student->vLLM sync is skipped). The student's LoRA adapter still
# trains via the base-corrected reverse-KL-to-teacher advantage + rollout IS
# correction, i.e. SFT-style LoRA distillation on teacher-generated data.
#
# NOTE: off-policy serves the teacher in vLLM directly, so the student LoRA is NOT
# injected into the rollout engine (unlike on-policy LoRA rollout). ROLLOUT_LOAD_FORMAT
# is left as the on-policy default; vLLM loads real teacher weights regardless because
# off_policy_rollout forces a non-dummy load_format.
#
# Any env var set by the caller still overrides these defaults.
# ============================================================================

# LoRA student training.
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Align with official FSDP+vLLM LoRA recommendations for the student side.
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"

# Off-policy: vLLM serves the frozen teacher.
export OFF_POLICY_ROLLOUT="${OFF_POLICY_ROLLOUT:-true}"

# Keep the OPD objective: base-corrected reverse-KL advantage + rollout IS correction.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}"
export ROLLOUT_IS="${ROLLOUT_IS:-token}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"

# LoRA training is typically stable with a lower LR than vector-only training.
export ACTOR_LR="${ACTOR_LR:-1e-5}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-IFEvalG-Qwen3-4B-IF-OffPolicy_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/ifevalg_offpolicy/ifevalg_qwen3-4b-offpolicy-lora_lr${ACTOR_LR}}"

# nohup sh ${OPV_ROOT}/examples/representation/ifevalg_offpolicy_lora8.sh >ifevalg_offpolicy_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@"
