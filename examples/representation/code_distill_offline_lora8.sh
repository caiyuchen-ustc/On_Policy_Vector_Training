#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) OFFLINE teacher-rollout reverse-KL distillation, Qwen3-4B on code. Reuses code_distill_offline.sh.

export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-4B-Code-OfflineDistill_lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# nohup bash ${OPV_ROOT}/examples/representation/code_distill_offline_lora8.sh >code_distill_offline_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_distill_offline.sh" "$@"
