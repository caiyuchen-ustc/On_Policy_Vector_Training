#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

# ============================================================================
# Teacher rollout dump for code off-policy distillation.
#
# Model / data mirror examples/representation/code_opd_fullparam.sh + code_opd.sh:
#   teacher = Qwen3-4B-Instruct-2507, prompts = Eurus/code_train.parquet.
#
# Set ROUNDS to the number of full-dataset sampling passes (epochs), and MAX_STEPS to cap the
# number of generation batches per round (0 = whole dataset). Each round -> round_${i}.parquet,
# then merged into teacher_sft_all.parquet (prompt + response) for the distill trainer.
#
# Usage:
#   ROUNDS=4 bash gen_teacher_rollouts_code.sh              # 4 epochs over full data
#   ROUNDS=1 MAX_STEPS=10 bash gen_teacher_rollouts_code.sh # 1 epoch, first 10*BATCH prompts
# nohup bash examples/representation/gen_teacher_rollouts_code.sh >gen_teacher_code.log 2>&1 &
# ============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Teacher + prompt data (from code_opd_fullparam.sh / code_opd.sh).
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${OPV_ROOT}/examples/code_rl/code_qwen3-4b-fullparam_lr1e-6_DAPO/global_step_300/hf_model}"
export DATA_ROOT="${DATA_ROOT:-${OPV_LEGACY_DATA_ROOT}/G-OPD-Training-Data/Eurus}"
export PROMPT_FILE="${PROMPT_FILE:-${DATA_ROOT}/code_train.parquet}"
export PROMPT_KEY="${PROMPT_KEY:-prompt}"

# Output.
export OUTPUT_BASE="${OUTPUT_BASE:-${OPV_ROOT}/examples/code_offpolicy/teacher_sft}"
export MERGED_OUTPUT="${MERGED_OUTPUT:-${OUTPUT_BASE}/teacher_sft_all.parquet}"

# Lengths mirror code_opd.sh (prompt 1024*2 / response 1024*16). Qwen3-4B-Instruct-2507 is non-thinking.
export PROMPT_LENGTH="${PROMPT_LENGTH:-$((1024*3))}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-$((1024*16))}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export GEN_TP="${GEN_TP:-2}"

# Rollout passes / step cap (override on the command line).
export ROUNDS="${ROUNDS:-2}"
export MAX_STEPS="${MAX_STEPS:-50}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/gen_teacher_rollouts_code.sh >gen_teacher_rollouts_code.log 2>&1 &
exec bash "${SCRIPT_DIR}/gen_teacher_rollouts.sh" "$@"
