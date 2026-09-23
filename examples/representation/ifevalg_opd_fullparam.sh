#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter IFEvalG instruction-following OPD wrapper around ifevalg_opd.sh.
# Any env var set by the caller still overrides these defaults.
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Full-parameter distillation is typically less stable with the vector-training LR.
export ACTOR_LR="${ACTOR_LR:-1e-6}"

# Vector artifacts are not needed for full-parameter training.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Optional convenience defaults for naming/outputs.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-IFEvalG-Qwen3-4B-IF-OPD_fullparam_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/ifevalg_opd/ifevalg_qwen3-4b-opd-fullparam_lr${ACTOR_LR}}"

# nohup sh ${OPV_ROOT}/examples/representation/ifevalg_opd_fullparam.sh >ifevalg_opd_fullparam1e-6.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@"
