#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GATED single-vector OPD (on-policy rep. distillation) wrapper around 4bopd.sh.
#
# Ablation for the "missing non-linearity" hypothesis:
#   h' = h + g(w·h + b) * (alpha * v)
# One extra learnable vector `w` (gate_vector) makes the additive steering
# INPUT-DEPENDENT (a per-token non-linear gate). If a HIGH layer that fails with
# plain vector steering RECOVERS once the gate is added, the bottleneck is the
# missing input-dependent non-linearity -- not layer depth or parameter count.
#
# Defaults inject at a HIGH layer range (32:33), where plain single-vector
# steering fails. Compare against 4bopd_single.sh at the SAME layers.

export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-single}"
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-1}"
# high layers: plain steering fails here, the gate should help
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-32:33}"
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"

# raw single vector from zero, no separate alpha scalar (match single-vector setup)
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}"

# no multi-vector curriculum for single-vector training
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}"

# --- gate (the ablation switch) ---
export TRAINABLE_TOKEN_VECTOR_GATED="${TRAINABLE_TOKEN_VECTOR_GATED:-true}"
export TRAINABLE_TOKEN_VECTOR_GATE_ACTIVATION="${TRAINABLE_TOKEN_VECTOR_GATE_ACTIVATION:-sigmoid}"
# gate_rank>0 -> low-rank non-linear adapter h + B*g(A*h) (B zero-init); 0 -> scalar-dot gate
export TRAINABLE_TOKEN_VECTOR_GATE_RANK="${TRAINABLE_TOKEN_VECTOR_GATE_RANK:-16}"

SAVE_VECTOR=${SAVE_VECTOR:-true}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"${OPV_ROOT}/repre/opd/repre_qwen3-4b-opd_gated/trainable_vectors"}

exec bash "${SCRIPT_DIR}/4bopd.sh" "$@"
