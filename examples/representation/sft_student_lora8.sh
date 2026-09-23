#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) student SFT on teacher rollouts. Any caller env still overrides these.

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
export STRATEGY=${STRATEGY:-fsdp2}
export LR=${LR:-1e-4}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ifevalg-student-sft-lora_r${LORA_RANK}_lr${LR}}

# nohup bash ${OPV_ROOT}/examples/representation/sft_student_lora8.sh >sft_student_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/sft_student.sh" "$@"
