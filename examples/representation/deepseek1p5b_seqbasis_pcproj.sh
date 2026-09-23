#!/bin/bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"
# Ablation wrapper: sequential-orthogonal steering with activation principal-subspace
# gradient projection. Controls WHERE the steering gradient is allowed to live:
#
#   PROJ=none        baseline — free training (identical to the base seqbasis script)
#   PROJ=complement  steer only in the ORTHOGONAL COMPLEMENT of the top-r activation PCs
#                    (removes the massive-activation principal directions)
#   PROJ=principal   steer only ALONG the top-r activation PCs (forced into the主方向)
#
# top-r = top-10% of hidden dim (1536 -> 154), precomputed & frozen from base-model
# activations on ALL FOUR SciKnowEval domains (physics/chemistry/biology/material,
# balanced sampling). Set PC_PROJECTION_FILE to the precomputed basis file.
#
# Usage:
#   PROJ=complement bash deepseek1p5b_seqbasis_pcproj.sh
#   PROJ=principal  bash deepseek1p5b_seqbasis_pcproj.sh
#   PROJ=none       bash deepseek1p5b_seqbasis_pcproj.sh
#
# Run all three (one per set of GPUs / sequentially) to compare:
#   free vs complement-only vs principal-only.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJ="${PROJ:-complement}"
export PC_PROJECTION_MODE="${PROJ}"
export PC_PROJECTION_FILE="${PC_PROJECTION_FILE:-}"

# Distinct experiment name / save dir per ablation arm so the three runs don't collide.
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-5:15}"
LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export ACTOR_LR="${ACTOR_LR:-5e-2}"
# Train the FIRST vector (v0) for 1000 steps, and eval every 100 steps during it.
# SEQ_RAW_STEPS = v0 raw-phase length; TEST_FREQ = periodic eval interval (seq curriculum
# now evals every test_freq steps in addition to at each vector switch).
export TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS="${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS:-1000}"
export TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS="${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS:-1000}"
export TEST_FREQ="${TEST_FREQ:-100}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepSeek-1.5B-SciKnowEval-seqbasis_pcproj-${PROJ}_layers${LAYER_TAG}_lr${ACTOR_LR}}"

echo "=================================================================="
echo "[PC-PROJ ablation] mode=${PC_PROJECTION_MODE}"
echo "  PC file : ${PC_PROJECTION_FILE}"
echo "  layers  : ${TRAINABLE_TOKEN_VECTOR_LAYERS}"
echo "  exp     : ${EXPERIMENT_NAME}"
echo "=================================================================="

if [ "${PC_PROJECTION_MODE}" != "none" ] && [ ! -f "${PC_PROJECTION_FILE}" ]; then
  echo "ERROR: Set PC_PROJECTION_FILE to an existing precomputed basis file." >&2
  exit 1
fi
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline_seqbasis.sh" "$@"
