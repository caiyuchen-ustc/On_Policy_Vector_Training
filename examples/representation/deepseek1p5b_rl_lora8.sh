#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) RL wrapper around deepseek1p5b_rl.sh.
export MODEL_PATH=${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}

# LoRA rollout/training defaults
export MODEL_USE_SHM=${MODEL_USE_SHM:-false}
export ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
export ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-true}
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}

export ACTOR_LR=${ACTOR_LR:-3e-5}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-128}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}

export SAVE_VECTOR=${SAVE_VECTOR:-false}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek1p5b_brorl_rl_lora_r${LORA_RANK}_lr${ACTOR_LR}}
exec bash "${SCRIPT_DIR}/deepseek1p5b_rl.sh" "$@"
