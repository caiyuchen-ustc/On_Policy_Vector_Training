#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
# Full-parameter IFEvalG instruction-following RL wrapper around ifevalg_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Keep baseline RL setup close to the representation reference scripts.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

# No vector artifact saving for full-parameter by default.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Save dir + exp name carry the learning rate (lr in the path).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-IFEvalG-Qwen3-8B-IF-RL_fullparam_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr${ACTOR_LR}}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/ifevalg_rl_fullparam.sh >ifevalg_rl_fullparam_1e-6.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_rl.sh" "$@"
