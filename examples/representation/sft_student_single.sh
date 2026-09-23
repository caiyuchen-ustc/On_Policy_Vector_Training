#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single trainable-token-vector student SFT on teacher rollouts (base frozen).
# Requires the fsdp1 strategy (use_orig_params) — set by sft_student.sh when the vector is on.
# Any caller env still overrides these.

export LORA_RANK=${LORA_RANK:-0}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export STRATEGY=${STRATEGY:-fsdp}
export TRAINABLE_TOKEN_VECTOR_LAYER_IDX=${TRAINABLE_TOKEN_VECTOR_LAYER_IDX:-8}
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# Vector-only training uses a high LR (matches the RL single-vector setup).
export LR=${LR:-1e-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ifevalg-student-sft-singlev_layer${TRAINABLE_TOKEN_VECTOR_LAYER_IDX}_lr${LR}}

# nohup bash ${OPV_ROOT}/examples/representation/sft_student_single.sh >sft_student_single.log 2>&1 &
exec bash "${SCRIPT_DIR}/sft_student.sh" "$@"
