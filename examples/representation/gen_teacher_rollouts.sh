#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"



if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================================
# Teacher rollout dump for off-policy distillation SFT (IFEvalG-style).
#
# Runs the frozen teacher over the training prompts with vLLM (verl.trainer.main_generation),
# ROUNDS times, each round a fresh independent sample of the WHOLE prompt set (different vLLM
# seed). Each round is written to round_${i}.parquet (prompts + a `responses` column). After
# all rounds, merge_rollouts_to_sft.py explodes them into one SFT-ready parquet with columns
# `prompt` (chat list) + `response` (teacher text), consumable by SFTDataset.
#
# "batch=1024" refers to the generation micro-batch (data.batch_size); the teacher still
# sweeps the entire prompt file each round. Set ROUNDS to how many times you want to resample
# every prompt.
#
# Usage:
#   ROUNDS=4 bash gen_teacher_rollouts.sh
#   ROUNDS=2 N_SAMPLES=1 TEACHER_MODEL_PATH=/path/to/teacher bash gen_teacher_rollouts.sh
# ============================================================================


export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -----------------------------
# Teacher / data / output
# -----------------------------
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${OPV_ROOT}/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr1e-6/global_step_825/hf_model"}

DATA_ROOT=${DATA_ROOT:-"${OPV_LEGACY_DATA_ROOT}/ifevalg"}
PROMPT_FILE=${PROMPT_FILE:-"${DATA_ROOT}/ifevalg_train.parquet"}
PROMPT_KEY=${PROMPT_KEY:-prompt}

OUTPUT_BASE=${OUTPUT_BASE:-"${OPV_ROOT}/examples/ifevalg_offpolicy/teacher_sft"}
MERGED_OUTPUT=${MERGED_OUTPUT:-"${OUTPUT_BASE}/teacher_sft_all.parquet"}

# -----------------------------
# Rollout knobs
# -----------------------------
ROUNDS=${ROUNDS:-1}                       # how many independent full-dataset sampling passes
N_SAMPLES=${N_SAMPLES:-1}                 # samples per prompt within a single round
BATCH_SIZE=${BATCH_SIZE:-1024}            # generation micro-batch
# MAX_STEPS: cap the number of generation batches per round (0 = whole dataset). The teacher
# then only sweeps the first MAX_STEPS*BATCH_SIZE prompts each round. Useful for quick tests.
MAX_STEPS=${MAX_STEPS:-0}
if [ "${MAX_STEPS}" -gt 0 ]; then
    MAX_PROMPTS=$(( MAX_STEPS * BATCH_SIZE ))
else
    MAX_PROMPTS=0
fi
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
PROMPT_LENGTH=${PROMPT_LENGTH:-$((1024*3))}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-$((1024*16))}
GEN_TP=${GEN_TP:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
ENABLE_THINKING=${ENABLE_THINKING:-false}
SEED_BASE=${SEED_BASE:-1000}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

mkdir -p "${OUTPUT_BASE}"

ROUND_FILES=()
for (( i=0; i<ROUNDS; i++ )); do
    ROUND_OUT="${OUTPUT_BASE}/round_${i}.parquet"
    ROUND_SEED=$(( SEED_BASE + i ))
    echo "=============================================================="
    echo "[gen_teacher_rollouts] round $((i+1))/${ROUNDS} -> ${ROUND_OUT} (seed=${ROUND_SEED})"
    echo "=============================================================="

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
        model.path="${TEACHER_MODEL_PATH}" \
        data.path="${PROMPT_FILE}" \
        data.prompt_key=${PROMPT_KEY} \
        data.n_samples=${N_SAMPLES} \
        data.batch_size=${BATCH_SIZE} \
        data.max_prompts=${MAX_PROMPTS} \
        data.output_path="${ROUND_OUT}" \
        +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
        rollout.name=vllm \
        rollout.temperature=${TEMPERATURE} \
        rollout.top_p=${TOP_P} \
        rollout.top_k=${TOP_K} \
        rollout.seed=${ROUND_SEED} \
        rollout.prompt_length=${PROMPT_LENGTH} \
        rollout.response_length=${RESPONSE_LENGTH} \
        rollout.tensor_model_parallel_size=${GEN_TP} \
        rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
        rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
        rollout.enforce_eager=True \
        rollout.free_cache_engine=True \
        "$@"

    ROUND_FILES+=("${ROUND_OUT}")
done

echo "=============================================================="
echo "[gen_teacher_rollouts] merging ${#ROUND_FILES[@]} round(s) -> ${MERGED_OUTPUT}"
echo "=============================================================="
python3 "${SCRIPT_DIR}/merge_rollouts_to_sft.py" \
    --inputs "${ROUND_FILES[@]}" \
    --output "${MERGED_OUTPUT}" \
    --prompt_key "${PROMPT_KEY}" \
    --gen_key responses \
    --response_key response

echo "[gen_teacher_rollouts] done. SFT data at: ${MERGED_OUTPUT}"

