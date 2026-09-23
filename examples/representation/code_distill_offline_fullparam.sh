#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter OFFLINE teacher-rollout reverse-KL distillation, Qwen3-4B on code
# (teacher = Qwen3-4B code-RL DAPO). Reuses code_distill_offline.sh.

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-4B-Code-OfflineDistill_fullparam_lr${ACTOR_LR}}"

# nohup bash ${OPV_ROOT}/examples/representation/code_distill_offline_fullparam.sh >code_distill_offline_fullparam.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_distill_offline.sh" "$@"
