#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# LoRA(rank=8) OFFLINE teacher-rollout reverse-KL distillation,
# DeepSeek-R1-Distill-Qwen-1.5B on SciKnowEval (teacher = BroRL-1.5B).
# Reuses the shared ifevalg_distill_offline.sh engine.
# ============================================================================

# --- deepseek1.5b / sciknoweval specifics ---
export MODEL_PATH="${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/BroRL-1.5B}"
export OFFLINE_TEACHER_DATA_PATH="${OFFLINE_TEACHER_DATA_PATH:-${OPV_ROOT}/examples/deepseek1p5b_offpolicy/teacher_sft/teacher_sft_all.parquet}"
export DATA_ROOT="${DATA_ROOT:-${OPV_LEGACY_DATA_ROOT}/sciknoweval}"
export VAL_FILE="${VAL_FILE:-${DATA_ROOT}/sciknoweval_validation.parquet}"
export VERIFY_SCRIPT_PATH="${VERIFY_SCRIPT_PATH:-${OPV_ROOT}/verl/utils/reward_score/sciknoweval_verify.py}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export GEN_TP="${GEN_TP:-1}"

export PROJECT_NAME="${PROJECT_NAME:-DeepSeek1p5B_SciKnowEval_OfflineDistill}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-DeepSeek-1.5B-SciKnowEval-OfflineDistill}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/deepseek1p5b_offline_distill}"

# --- LoRA training ---
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepSeek-1.5B-SciKnowEval-OfflineDistill_lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# nohup bash ${OPV_ROOT}/examples/representation/deepseek1p5b_distill_offline_lora8.sh >deepseek1p5b_distill_offline_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline.sh" "$@"
