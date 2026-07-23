#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) K&K logic-RL wrapper around logic_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Follow recommended LoRA rollout settings.
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-false}"

# For RL LoRA runs with use_kl_loss=false, keep this off.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}"

export ACTOR_LR="${ACTOR_LR:-1e-5}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-KK-Qwen3-4B-Logic-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/logic_rl/kk_qwen3-4b-lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/logic_rl_lora8.sh >logic_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/logic_rl.sh" "$@"
