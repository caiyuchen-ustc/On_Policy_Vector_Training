#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

# ============================================================================
# Teacher rollout dump for Qwen3-4B off-policy distillation (math).
#
# Model / data mirror examples/representation/4bopd_fullparam.sh + 4bopd.sh:
#   teacher = Qwen3-4B-Non-Thinking-RL-Math-Step500,
#   prompts = DeepMath-103K/train_filtered_level6.parquet.
#
# Set ROUNDS to the number of full-dataset sampling passes (epochs), and MAX_STEPS to cap the
# number of generation batches per round (0 = whole dataset). Each round -> round_${i}.parquet,
# then merged into teacher_sft_all.parquet (prompt + response) for the distill trainer.
#
# Usage:
#   ROUNDS=4 bash gen_teacher_rollouts_4b.sh              # 4 epochs over full data
#   ROUNDS=1 MAX_STEPS=10 bash gen_teacher_rollouts_4b.sh # 1 epoch, first 10*BATCH prompts
# nohup bash examples/representation/gen_teacher_rollouts_4b.sh >gen_teacher_4b.log 2>&1 &
# ============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Teacher + prompt data (from 4bopd_fullparam.sh / 4bopd.sh).
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
export DATA_ROOT="${DATA_ROOT:-${OPV_LEGACY_DATA_ROOT}/G-OPD-Training-Data/DeepMath-103K}"
export PROMPT_FILE="${PROMPT_FILE:-${DATA_ROOT}/train_filtered_level6.parquet}"
export PROMPT_KEY="${PROMPT_KEY:-prompt}"

# Output.
export OUTPUT_BASE="${OUTPUT_BASE:-${OPV_ROOT}/examples/4b_offpolicy/teacher_sft}"
export MERGED_OUTPUT="${MERGED_OUTPUT:-${OUTPUT_BASE}/teacher_sft_all.parquet}"

# Lengths mirror 4bopd.sh (prompt 3072 / response 16384). Teacher is a non-thinking model.
export PROMPT_LENGTH="${PROMPT_LENGTH:-3072}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-16384}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export GEN_TP="${GEN_TP:-2}"

# Rollout passes / step cap (override on the command line).
export ROUNDS="${ROUNDS:-2}"
export MAX_STEPS="${MAX_STEPS:-50}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/gen_teacher_rollouts_4b.sh >gen_teacher_rollouts_4b.log 2>&1 &
exec bash "${SCRIPT_DIR}/gen_teacher_rollouts.sh" "$@"
