#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-vector DAPO code-RL wrapper around code_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f

export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-single}"
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-1}"
# Layer range the single vector is injected into, e.g. 0:20 == layers 0-20.
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-10:25}"
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"

# Match the original single-vector setup: directly train one raw vector from zero,
# without a separate alpha scalar.
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}"

# Single-vector training does not need the multi-vector curriculum.
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}"

# Vector-only RL keeps the OPD-style higher LR and saves vector artifacts.
export LORA_RANK="${LORA_RANK:-0}"
export ACTOR_LR="${ACTOR_LR:-1e-4}"
export SAVE_VECTOR="${SAVE_VECTOR:-true}"
# Vector-only training keeps steer params on GPU; optimizer offload can cause
# AdamW state/device mismatch (cpu vs cuda) during step.
export OFFLOAD="${OFFLOAD:-false}"

# Layer range goes into both the save dir and the experiment name (lr also in path).
LAYER_TAG="${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DAPO-Qwen3-4B-Code-RL_singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-singlev_layers${LAYER_TAG}_lr${ACTOR_LR}_DAPO}"
export SAVE_VECTOR_DIR="${SAVE_VECTOR_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/code_rl/trainable_vectors/singlev_layers${LAYER_TAG}_lr${ACTOR_LR}}"

# --- Code sandbox: auto-start the official SandboxFusion server and wire its URL ---
# run_sandboxfusion.sh start is idempotent (re-uses an already-running server) and serves
# POST /run_code on 127.0.0.1:${SANDBOX_PORT}. One-time env setup: setup_sandboxfusion.sh.
# Set AUTO_START_SANDBOX=false to skip (sandbox managed separately / on another host).
export SANDBOX_PORT="${SANDBOX_PORT:-8080}"
SANDBOXFUSION_DIR="${SANDBOXFUSION_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/SandboxFusion}"
# Loopback traffic must bypass the corporate proxy, else POSTs to 127.0.0.1 fail.
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
AUTO_START_SANDBOX="${AUTO_START_SANDBOX:-true}"
if [ "${AUTO_START_SANDBOX}" = "true" ]; then
    bash "${SANDBOXFUSION_DIR}/run_sandboxfusion.sh" start
fi
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://127.0.0.1:${SANDBOX_PORT}/run_code}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/code_rl_single.sh >code_rl_single_layers0-20.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_rl.sh" "$@"
