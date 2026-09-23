#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"
export RAY_DISABLE_MEMORY_MONITOR=1
export USED_MODEL="no_api"

# Attention backend for VLLM (commented out if not needed)
# export VLLM_ATTENTION_BACKEND=XFORMERS

export NCCL_P2P_DISABLE=0
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Sequential orthogonal basis distillation (offline teacher rollouts).
#
# Trains a set of orthonormal steer vectors ONE AT A TIME on the pre-generated teacher data:
#   - v0 learns raw for SEQ_RAW_STEPS steps -> alpha := ||v0|| (fixed for the whole run), v0 normalized.
#   - then each subsequent v_k is trained projected onto the orthogonal complement of all frozen
#     v_0..v_{k-1} (unit norm), injecting ONLY the current v_k * alpha.
#   - switch to the next vector when it has trained SEQ_MAX_ITERS steps OR the policy loss stays
#     below SEQ_LOSS_THRESHOLD for SEQ_LOSS_PATIENCE consecutive steps.
#   - stop after NUM_VECTORS vectors -> an orthonormal basis + one fixed alpha.
#
# Any caller env still overrides these defaults.
# ============================================================================

# Multi-vector, sequential_orthogonal curriculum. Learnable alpha so v0's raw magnitude is captured.
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-multi}
##############
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-64}
# Hard cap per vector (only hit if it never converges). Convergence-based switch (below)
# normally triggers first; later vectors in the residual space take more steps and get them.
export TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS=${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS:-500}
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-1:27}
##############
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-sequential_orthogonal}
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}

export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# Sequential switch knobs.
# vector 0: trained a FIXED SEQ_RAW_STEPS steps, then switch (no convergence test) — it fits
#           the largest reverse-KL component, give it the full fixed budget.
# vectors 1..: CONVERGENCE-BASED early-stopping on the EMA-smoothed loss — switch when the EMA
#           loss stops reaching a new low for PATIENCE consecutive steps (a plateau). THRESHOLD
#           is a small relative noise-floor: an improvement only counts as a "new low" if it
#           beats the best-so-far by >= THRESHOLD (0 = any new low counts). Slow-but-steady
#           descent keeps setting new lows -> keeps training; only a true plateau switches. So
#           later (residual-space) vectors automatically get more steps, capped by SEQ_MAX_ITERS.
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD:-0.005}
export TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE:-20}
# v0 fixed training steps (was the old raw-phase length; now = vector-0 budget).
export TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS=${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS:-100}

# Vector-only training defaults.
export LORA_RANK=${LORA_RANK:-0}
export ACTOR_LR=${ACTOR_LR:-5e-2}
export SAVE_VECTOR=${SAVE_VECTOR:-true}
export SAVE_FREQ=${SAVE_FREQ:--1}
export OFFLOAD=${OFFLOAD:-false}

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-IFEvalG-Qwen3-4B-IF-loraOfflineDistill_seqbasis${TRAINABLE_TOKEN_VECTOR_NUM}_layers${LAYER_TAG}_lr${ACTOR_LR}}
RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/ifevalg_distill_offline_seqbasis.sh >ifevalg_distill_offline_seqbasismulti5e-2.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_distill_offline.sh" "$@"
