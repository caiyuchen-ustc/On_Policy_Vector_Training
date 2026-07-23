#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}
export MODEL_PATH=${MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B}

# ---- 训练模式: 只训练单个 decoder layer 的全部参数, 其余所有层与非层参数冻结 ----
export LORA_RANK=${LORA_RANK:-0}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
export TRAIN_Q_BIAS_ONLY=${TRAIN_Q_BIAS_ONLY:-false}
export TRAIN_SINGLE_LAYER_ONLY=${TRAIN_SINGLE_LAYER_ONLY:-true}
# 定位第 15 层
export TRAIN_LAYER_IDX=${TRAIN_LAYER_IDX:-15}

# ---- 算法 ----
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-false}

# ---- 优化: 单层全参 (~2000万参数), 用比全参略大的小 lr ----
export ACTOR_LR=${ACTOR_LR:-1e-5}
export ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.0}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-64}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}

# 单层训练需要 use_orig_params=True (只有部分参数可训练)
# deepseek1p5b_rl.sh 默认已设 use_orig_params=True

export SAVE_VECTOR=${SAVE_VECTOR:-false}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek1p5b_rl_singlelayer${TRAIN_LAYER_IDX}_lr${ACTOR_LR}}
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/deepseek1p5b_rl_singlelayer.sh >1p5b_rl_singlelayer.log 2>&1 &
exec bash "${SCRIPT_DIR}/deepseek1p5b_rl.sh" "$@"
