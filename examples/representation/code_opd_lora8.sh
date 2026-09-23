#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) code OPD wrapper around code_opd.sh.
# Any env var set by the caller still overrides these defaults.

# Student/teacher model defaults (same 4B backbone family as code_rl wrappers).
export MODEL_PATH="${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/Qwen3-4B}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH}}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/Qwen3-4B-Instruct-2507}"

export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Follow LoRA rollout recommendations.
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-false}"

# Keep code-RL aligned batch defaults in wrappers.
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-OPD-Qwen3-4B-Code_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${OPV_ROOT}/examples/code_opd/code_qwen3-4b-lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# Auto-start sandbox server unless managed externally.
export SANDBOX_PORT="${SANDBOX_PORT:-8080}"
SANDBOXFUSION_DIR="${SANDBOXFUSION_DIR:-${OPV_ROOT}/../SandboxFusion}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
AUTO_START_SANDBOX="${AUTO_START_SANDBOX:-true}"
if [ "${AUTO_START_SANDBOX}" = "true" ]; then
    bash "${SANDBOXFUSION_DIR}/run_sandboxfusion.sh" start
fi
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://127.0.0.1:${SANDBOX_PORT}/run_code}"

exec bash "${SCRIPT_DIR}/code_opd.sh" "$@"
