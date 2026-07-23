#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter RL wrapper around 4b_rl.sh.
# Any env var set by the caller still overrides these defaults.

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Keep baseline RL setup close to your reference script.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

# No vector artifact saving for full-parameter by default.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DAPO-Qwen3-4B-RL_fullparam_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/rl/science_repre_qwen3-4b-fullparam_lr${ACTOR_LR}}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/4b_rl_fullparam.sh >4b_rl_fullparam.log 2>&1 &
exec bash "${SCRIPT_DIR}/4b_rl_science.sh" "$@"
