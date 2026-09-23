#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter OFFLINE teacher-rollout reverse-KL distillation.
# The student's full weights are trained to match the teacher (reverse-KL) on the
# pre-generated teacher rollouts. Any caller env still overrides these defaults.

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-2e-5}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-IFEvalG-Qwen3-4B-IF-OfflineDistill_fullparam_lr${ACTOR_LR}}"
exec bash "${SCRIPT_DIR}/ifevalg_distill_offline.sh" "$@"
