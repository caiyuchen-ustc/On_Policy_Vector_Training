#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) RL wrapper around 4b_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
# Follow recommended LoRA rollout settings.
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-false}"

# For RL LoRA runs with use_kl_loss=false, keep this off;
# otherwise update_policy may require missing ref_log_prob and crash.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}"

# LoRA LR default.
export ACTOR_LR="${ACTOR_LR:-3e-5}"

# LoRA does not need vector artifacts.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
# export WANDB_PROXY=http://star-proxy.oa.com:3128
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DAPO-Qwen3-4B-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/4b_rl_lora8.sh >4b_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/4b_rl.sh" "$@"
