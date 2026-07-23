#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B}"
export Teacher_MODEL_PATH="${Teacher_MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/BroRL-1.5B}"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
RUN_ROOT="${RUN_ROOT:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/deepseek1p5b_brorl_opd}"

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek1p5b_brorl_opd_fullparam_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${RUN_ROOT}/checkpoints}"
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${RUN_ROOT}/trainable_vectors}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/deepseek1p5b_brorl_opd_fullparam.sh >1p5b_rl_fullparam1e-6.log 2>&1 &
exec bash "${SCRIPT_DIR}/1p5bopd.sh" "$@"
