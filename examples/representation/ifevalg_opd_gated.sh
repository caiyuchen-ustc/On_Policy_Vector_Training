#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GATED single-vector IFEvalG instruction-following OPD wrapper around ifevalg_opd.sh.
#
# Ablation for the "missing non-linearity" hypothesis:
#   h' = h + g(h · v1) * v2
# v2 = the usual single steering vector (the injected direction). v1 = a second
# learnable vector; h·v1 is a per-token scalar, g a non-linearity, so the amount
# of v2 injected is input-dependent and non-linear. If a HIGH layer that fails
# with plain vector steering RECOVERS with the gate, the bottleneck is the missing
# input-dependent non-linearity -- not layer depth or parameter count.
#
# Compare against ifevalg_opd_single.sh at the SAME layers (GATED=false).

export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-single}
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
# high layers: plain steering is weakest here, the gate should help
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-20:27}
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}

# raw single vector from zero, no separate alpha scalar (match single-vector setup)
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}

# no multi-vector curriculum for single-vector training
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# --- gate (the ablation switch) ---
export TRAINABLE_TOKEN_VECTOR_GATED=${TRAINABLE_TOKEN_VECTOR_GATED:-true}
export TRAINABLE_TOKEN_VECTOR_GATE_ACTIVATION=${TRAINABLE_TOKEN_VECTOR_GATE_ACTIVATION:-sigmoid}
# gate_rank>0 -> low-rank non-linear adapter h + B*g(A*h) (B zero-init); 0 -> scalar-dot gate
export TRAINABLE_TOKEN_VECTOR_GATE_RANK=${TRAINABLE_TOKEN_VECTOR_GATE_RANK:-16}

# Vector-only OPD defaults.
export LORA_RANK=${LORA_RANK:-0}
export ACTOR_LR=${ACTOR_LR:-1e-1}
export SAVE_VECTOR=${SAVE_VECTOR:-true}
export SAVE_FREQ=${SAVE_FREQ:--1}
export OFFLOAD=${OFFLOAD:-false}

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
GATE_TAG=$([ "${TRAINABLE_TOKEN_VECTOR_GATED}" = "true" ] && echo "gated-${TRAINABLE_TOKEN_VECTOR_GATE_ACTIVATION}" || echo "plain")
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-IFEvalG-Qwen2.5-7B-IF-OPD_singlev_${GATE_TAG}_layers${LAYER_TAG}_lr${ACTOR_LR}}
export DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${OPV_ROOT}/examples/ifevalg_opd/ifevalg_qwen3-4b-if-opd-singlev_${GATE_TAG}_layers${LAYER_TAG}_lr${ACTOR_LR}}
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/ifevalg_opd_gated.sh >ifevalg_opd_gated.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@"
