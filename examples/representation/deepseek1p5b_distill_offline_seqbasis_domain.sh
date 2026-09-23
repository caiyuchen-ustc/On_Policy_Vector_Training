#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Per-DOMAIN sequential-orthogonal offline distillation, DeepSeek-1.5B / SciKnowEval.
#
# The SciKnowEval data was 4 domains (physics/chemistry/biology/material) merged together.
# split_sciknoweval_by_domain.py has split train / validation / teacher-rollout parquet into
# data/sciknoweval/by_domain/<DOMAIN>/ and .../teacher_sft/by_domain/<DOMAIN>/.
# This wrapper points a seqbasis run at ONE domain so each topic is trained + eval'd separately.
#
# Usage:
#   DOMAIN=physics   bash deepseek1p5b_distill_offline_seqbasis_domain.sh
#   DOMAIN=chemistry bash deepseek1p5b_distill_offline_seqbasis_domain.sh
#   DOMAIN=biology   bash deepseek1p5b_distill_offline_seqbasis_domain.sh
#   DOMAIN=material  bash deepseek1p5b_distill_offline_seqbasis_domain.sh
# ============================================================================

# DOMAIN="${DOMAIN:-physics}"
# DOMAIN="${DOMAIN:-chemistry}"
# DOMAIN="${DOMAIN:-biology}"
DOMAIN="${DOMAIN:-material}"
case "$DOMAIN" in
    physics|chemistry|biology|material) ;;
    *) echo "Invalid DOMAIN=$DOMAIN (expected physics|chemistry|biology|material)" >&2; exit 1 ;;
esac

DATA_ROOT_BASE="${DATA_ROOT_BASE:-${OPV_LEGACY_DATA_ROOT}/sciknoweval}"
TEACHER_BASE="${TEACHER_BASE:-${OPV_ROOT}/examples/deepseek1p5b_offpolicy/teacher_sft}"

# Point train / val / teacher-rollout at this domain's split.
export DATA_ROOT="${DATA_ROOT_BASE}/by_domain/${DOMAIN}"
export VAL_FILE="${DATA_ROOT}/sciknoweval_validation.parquet"
export OFFLINE_TEACHER_DATA_PATH="${TEACHER_BASE}/by_domain/${DOMAIN}/teacher_sft_all.parquet"

# Tag the run / save dirs with the domain so they don't collide across domains.
export PROJECT_NAME="${PROJECT_NAME:-DeepSeek1p5B_SciKnowEval_OfflineDistill_perdomain}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-DeepSeek-1.5B-SciKnowEval-${DOMAIN}-OfflineDistill}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepSeek-1.5B-SciKnowEval-${DOMAIN}-seqbasis}"

# The engine builds SAVE_VECTOR_DIR / DEFAULT_LOCAL_DIR from a layer/lr tag that has NO domain,
# so all four domains would overwrite the same dir. Pin domain-tagged output dirs here.
# (layer 5:15 -> tag "5-15", lr 5e-2 match the seqbasis wrapper defaults; override via env if changed.)
LAYERS_FOR_TAG="${TRAINABLE_TOKEN_VECTOR_LAYERS:-5:15}"
LAYER_TAG="${LAYERS_FOR_TAG//:/-}"
LR_FOR_TAG="${ACTOR_LR:-5e-2}"
_LOCAL_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/deepseek1p5b_offline_distill}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${_LOCAL_BASE}/seqbasis_${DOMAIN}_layers${LAYER_TAG}_lr${LR_FOR_TAG}}"
export SAVE_VECTOR_DIR="${SAVE_VECTOR_DIR:-${_LOCAL_BASE}/trainable_vectors/seqbasis_${DOMAIN}_layers${LAYER_TAG}_lr${LR_FOR_TAG}}"

echo "[domain] DOMAIN=${DOMAIN}"
echo "[domain] TRAIN/TEACHER = ${OFFLINE_TEACHER_DATA_PATH}"
echo "[domain] VAL           = ${VAL_FILE}"
echo "[domain] SAVE_VECTOR_DIR = ${SAVE_VECTOR_DIR}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/deepseek1p5b_distill_offline_seqbasis_domain.sh >deepseek1p5b_distill_offline_domain2seqbasis5e-4.log 2>&1 &
# Delegate to the standard 1.5b seqbasis wrapper (layers 5:15, sequential_orthogonal, etc.).
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline_seqbasis.sh" "$@"
