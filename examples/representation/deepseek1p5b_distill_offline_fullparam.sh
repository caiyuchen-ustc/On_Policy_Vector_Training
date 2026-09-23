#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Full-parameter OFFLINE teacher-rollout reverse-KL distillation, DeepSeek-R1-Distill-Qwen-1.5B
# on SciKnowEval. Uses the pre-generated teacher rollouts from
# gen_teacher_rollouts_deepseek1p5b.sh (teacher = BroRL-1.5B).
#
# Reuses the shared ifevalg_distill_offline.sh engine; only the model / data / reward env differ.
# ============================================================================

# --- deepseek1.5b / sciknoweval specifics ---
export MODEL_PATH="${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${OPV_ROOT}/deepseek1p5b_brorl_rl/checkpoints/fullparam_lr1e-6/global_step_820/hf_model}"
export OFFLINE_TEACHER_DATA_PATH="${OFFLINE_TEACHER_DATA_PATH:-${OPV_ROOT}/examples/deepseek1p5b_offpolicy/teacher_sft/teacher_sft_all.parquet}"
export DATA_ROOT="${DATA_ROOT:-${OPV_LEGACY_DATA_ROOT}/sciknoweval}"
export VAL_FILE="${VAL_FILE:-${DATA_ROOT}/sciknoweval_validation.parquet}"
export VERIFY_SCRIPT_PATH="${VERIFY_SCRIPT_PATH:-${OPV_ROOT}/verl/utils/reward_score/sciknoweval_verify.py}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"     # DeepSeek-R1-Distill is a thinking model
export GEN_TP="${GEN_TP:-1}"

export PROJECT_NAME="${PROJECT_NAME:-DeepSeek1p5B_SciKnowEval_OfflineDistill}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-DeepSeek-1.5B-SciKnowEval-OfflineDistill}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/deepseek1p5b_offline_distill}"

# --- full-parameter training ---
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepSeek-1.5B-SciKnowEval-OfflineDistill_fullparam_lr${ACTOR_LR}}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/deepseek1p5b_distill_offline_fullparam.sh >deepseek1p5b_distill_offline_fullparam.log 2>&1 &
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline.sh" "$@"
