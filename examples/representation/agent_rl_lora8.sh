#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) workbench agent-RL wrapper around agent_rl.sh.
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

export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Agent-Qwen3-8B-Workbench-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/agent_rl/agent_qwen3-8b-lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/agent_rl_lora8.sh >agent_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/agent_rl.sh" "$@"
