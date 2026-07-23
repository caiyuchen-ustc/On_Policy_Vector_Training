#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-vector code OPD wrapper around code_opd.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}

# Student/teacher model defaults (same 4B backbone family as code_rl wrappers).
export MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-8B}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH}}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-fullparam_lr1e-6_DAPO/global_step_300/hf_model}"

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-single}"
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-1}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-19:20}"
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"

# Train one raw vector directly, no separate alpha scalar.
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}"

# Single-vector training does not need multi-vector curriculum.
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}"

# Keep code-RL aligned batch defaults in wrappers.
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-1024}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-1024}"

export ACTOR_LR="${ACTOR_LR:-1e-2}"
export SAVE_VECTOR="${SAVE_VECTOR:-true}"

LAYER_TAG="${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-OPD-Qwen3-4B-Code_singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_opd/code_qwen3-4b-singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}"
export SAVE_VECTOR_DIR="${SAVE_VECTOR_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/code_opd/trainable_vectors/singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}"

# Auto-start sandbox server unless managed externally.
export SANDBOX_PORT="${SANDBOX_PORT:-8081}"
SANDBOXFUSION_DIR="${SANDBOXFUSION_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/SandboxFusion}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
AUTO_START_SANDBOX="${AUTO_START_SANDBOX:-true}"
if [ "${AUTO_START_SANDBOX}" = "true" ]; then
    bash "${SANDBOXFUSION_DIR}/run_sandboxfusion.sh" start
fi
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://127.0.0.1:${SANDBOX_PORT}/run_code}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/code_opd_single.sh >code_opd_single_layers10-25.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_opd.sh" "$@"
