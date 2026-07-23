#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}

# LoRA(rank=8) IFEvalG RL wrapper around ifevalg_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}
export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}

# LoRA rollout defaults.
export MODEL_USE_SHM=${MODEL_USE_SHM:-false}
export ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
# layered_summon: sync LoRA weights layer-by-layer to reduce peak memory and sync time.
export ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-true}
# fully_sharded_loras: shard LoRA computation fully across TP ranks (faster at high TP/seq-len).
export LORA_FULLY_SHARDED=${LORA_FULLY_SHARDED:-1}

# For RL LoRA runs with use_kl_loss=false, keep this off.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}
export ACTOR_LR="${ACTOR_LR:-3e-5}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"


# LoRA does not need vector artifacts.
export SAVE_VECTOR=${SAVE_VECTOR:-false}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-IFEvalG-Qwen3-8B-IF-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}
export DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-lora_r${LORA_RANK}_lr${ACTOR_LR}}
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/ifevalg_rl_lora8.sh >ifevalg_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_rl.sh" "$@"
