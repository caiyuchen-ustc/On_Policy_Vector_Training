#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH=${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}

# ---- 训练模式: 只训练 attention q_proj 的 bias (模型原生参数), 冻结 k/v ----
# 关闭 LoRA 和 steer token vector, 打开 q-bias-only
export LORA_RANK=${LORA_RANK:-0}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
export TRAIN_Q_BIAS_ONLY=${TRAIN_Q_BIAS_ONLY:-true}
# 只训练 0-20 层 (闭区间, 共 21 层) 的 q_proj bias, 其余层冻结
export Q_BIAS_LAYER_START=${Q_BIAS_LAYER_START:-0}
export Q_BIAS_LAYER_END=${Q_BIAS_LAYER_END:-20}

# ---- 算法 ----
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-false}

# ---- 优化: bias 直接作用于 Q, 杠杆大 -> 小 lr + 大 mini_bsz, 不加 weight decay ----
export ACTOR_LR=${ACTOR_LR:-5e-4}
export ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.0}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-128}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}

# q-bias 训练需要 use_orig_params=True (只有部分参数可训练)
# deepseek1p5b_rl.sh 默认已设 use_orig_params=True

export SAVE_VECTOR=${SAVE_VECTOR:-false}

LAYER_TAG="${Q_BIAS_LAYER_START}-${Q_BIAS_LAYER_END}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek1p5b_rl_qbias_layers${LAYER_TAG}_lr${ACTOR_LR}}
exec bash "${SCRIPT_DIR}/deepseek1p5b_rl.sh" "$@"
