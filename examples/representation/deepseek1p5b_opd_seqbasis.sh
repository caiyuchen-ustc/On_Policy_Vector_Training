#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"
export NCCL_P2P_DISABLE=0
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# DeepSeek-R1-Distill-Qwen-1.5B ON-POLICY multi-vector OPD training,
# sequential_orthogonal curriculum, steer layers 5..15.
#
# Same model / teacher / data / vLLM-rollout OPD loop as deepseek1p5b_opd_multi_raw.sh, but
# swaps the raw_then_double curriculum for sequential_orthogonal (one orthogonal, free-magnitude
# vector at a time):
#   - v0 trains a FIXED SEQ_RAW_STEPS steps; per-layer ||v0|| becomes that layer's target norm.
#   - each subsequent v_k (per layer) is trained projected onto the orthogonal complement of the
#     already-frozen v_0..v_{k-1} (free magnitude — NOT unit-normalized), injecting ONLY v_k.
#   - a layer freezes its v_k once its (post-projection) norm reaches that layer's target ||v0||;
#     when EVERY steer layer has reached its target, all layers switch to the next vector together
#     and one eval runs (injecting the just-finished vector). Then the next vector starts.
#   - stop after NUM_VECTORS vectors -> a per-layer orthogonal basis (magnitudes ~||v0||).
#
# eval fires ONLY on a vector switch (or the last step), not every test_freq (handled in the
# trainer for sequential_orthogonal). Any caller env still overrides these defaults.
# ============================================================================

export MODEL_PATH="${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export Teacher_MODEL_PATH="${Teacher_MODEL_PATH:-${OPV_ROOT}/deepseek1p5b_brorl_rl/checkpoints/fullparam_lr1e-6/global_step_820/hf_model}"

RUN_ROOT="${RUN_ROOT:-${OPV_ROOT}/deepseek1p5b_rl_opd}"

export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-multi}"

# ----- core: vectors per layer + steer layers 5..15 -----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-1:27}"

export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}"

# ----- sequential_orthogonal curriculum (free magnitude + orthogonal, norm-target switch) -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-sequential_orthogonal}"
# vector 0: FIXED SEQ_RAW_STEPS steps; per-layer ||v0|| becomes that layer's target norm.
# vectors 1..: each layer freezes its vector once its norm reaches that layer's target ||v0||;
#              all layers switch together only after EVERY layer has reached its target (one eval
#              per switch). SEQ_MAX_ITERS is a per-vector safety cap.
export TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS="${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS:-100}"
export TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS="${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS:-500}"
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD="${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD:-0.0}"
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE="${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE:-10}"

# ----- vector-only OPD defaults -----
export ACTOR_LR="${ACTOR_LR:-5e-2}"
export SAVE_VECTOR="${SAVE_VECTOR:-true}"
# steer-vector Adam beta1 (0 = no first-moment momentum; avoids constraint-beating ripple).
export STEER_ADAM_BETA1="${STEER_ADAM_BETA1:-0}"

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek1p5b_rl_opd_seqbasis${TRAINABLE_TOKEN_VECTOR_NUM}_layers${LAYER_TAG}_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${RUN_ROOT}/checkpoints}"
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${RUN_ROOT}/trainable_vectors_seqbasis}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/deepseek1p5b_opd_seqbasis.sh >1p5b_opd_seqbasis5e-2.log 2>&1 &
exec bash "${SCRIPT_DIR}/1p5bopd.sh" "$@"
