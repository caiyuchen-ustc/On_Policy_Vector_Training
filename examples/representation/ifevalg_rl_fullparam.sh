#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Full-parameter IFEvalG instruction-following RL wrapper around ifevalg_rl.sh.
# Any env var set by the caller still overrides these defaults.
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Keep baseline RL setup close to the representation reference scripts.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

# No vector artifact saving for full-parameter by default.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Save dir + exp name carry the learning rate (lr in the path).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-IFEvalG-Qwen3-8B-IF-RL_fullparam_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${OPV_ROOT}/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr${ACTOR_LR}}"
exec bash "${SCRIPT_DIR}/ifevalg_rl.sh" "$@"
