#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"
export NCCL_P2P_DISABLE=0
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Sequential orthogonal basis distillation (offline teacher rollouts),
# DeepSeek-R1-Distill-Qwen-1.5B on SciKnowEval (teacher = BroRL-1.5B, base frozen),
# steer layers 5..15. Reuses the shared deepseek1p5b_distill_offline.sh engine.
#
# Trains a set of orthogonal, free-magnitude steer vectors ONE AT A TIME on the pre-generated
# teacher data:
#   - v0 trains a FIXED SEQ_RAW_STEPS steps; its per-layer norm ||v0|| becomes that layer's
#     target norm.
#   - each subsequent v_k (per layer) is trained projected onto the orthogonal complement of the
#     already-frozen v_0..v_{k-1} (free magnitude — NOT unit-normalized), injecting ONLY v_k.
#   - a layer freezes its v_k once its (post-projection) norm reaches that layer's target ||v0||;
#     when EVERY steer layer has reached its target, all layers switch to the next vector together
#     and one eval runs (injecting the just-finished vector). Then the next vector starts.
#   - stop after NUM_VECTORS vectors -> a per-layer orthogonal basis (magnitudes ~||v0||).
#
# Any caller env still overrides these defaults.
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
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export GEN_TP="${GEN_TP:-1}"

export PROJECT_NAME="${PROJECT_NAME:-DeepSeek1p5B_SciKnowEval_OfflineDistill}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-DeepSeek-1.5B-SciKnowEval-OfflineDistill}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/deepseek1p5b_offline_distill}"

# --- multi-vector, sequential_orthogonal curriculum (free magnitude + orthogonal) ---
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-multi}"
##############
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-1:27}"
##############
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-sequential_orthogonal}"
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}"

export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}"

# Sequential switch knobs (norm-target).
# vector 0: trained a FIXED SEQ_RAW_STEPS steps; per-layer ||v0|| becomes that layer's target.
# vectors 1..: each layer freezes its vector once its norm reaches that layer's target ||v0||;
#              all layers switch together only after EVERY layer has reached its target (one eval
#              per switch). SEQ_MAX_ITERS is a per-vector safety cap.
export TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS="${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS:-100}"
export TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS="${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS:-500}"
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD="${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD:-0.0}"
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE="${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE:-10}"

# Vector-only training defaults.
export LORA_RANK="${LORA_RANK:-0}"
export ACTOR_LR="${ACTOR_LR:-5e-2}"
export SAVE_VECTOR="${SAVE_VECTOR:-true}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export OFFLOAD="${OFFLOAD:-false}"
# steer-vector Adam beta1 (0 = no first-moment momentum; avoids constraint-beating ripple).
export STEER_ADAM_BETA1="${STEER_ADAM_BETA1:-0}"

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepSeek-1.5B-SciKnowEval-OfflineDistill_seqbasis${TRAINABLE_TOKEN_VECTOR_NUM}_layers${LAYER_TAG}_lr${ACTOR_LR}}"
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline.sh" "$@"
