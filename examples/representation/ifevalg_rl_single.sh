#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}

# Single-vector IFEvalG RL wrapper around ifevalg_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}

export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-single}
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-0:20}
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}

# Match single-vector setup: no separate learnable alpha.
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}

# Single-vector training does not need multi-vector curriculum.
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}

# Vector-only RL defaults.
export LORA_RANK=${LORA_RANK:-0}
export ACTOR_LR=${ACTOR_LR:-5e-2}
export SAVE_VECTOR=${SAVE_VECTOR:-true}
export SAVE_FREQ=${SAVE_FREQ:--1}
# Trainable token vectors do not work with FSDP param/optimizer offload here;
# keep them on GPU to avoid optimizer state / grad device mismatch at step().
export OFFLOAD=${OFFLOAD:-false}

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-IFEvalG-Qwen3-8B-IF-RL_singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}
export DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/ifevalg_rl_single.sh >ifevalg_rl_single.log 2>&1 &
exec bash "${SCRIPT_DIR}/ifevalg_rl.sh" "$@"
