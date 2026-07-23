#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) DAPO code-RL wrapper around code_rl.sh.
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

# For RL LoRA runs with use_kl_loss=false, keep this off;
# otherwise update_policy may require missing ref_log_prob and crash.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}"

# LoRA LR default.
export ACTOR_LR="${ACTOR_LR:-2e-5}"

# LoRA does not need vector artifacts.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Save dir + exp name carry the LoRA rank and learning rate (lr in the path).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DAPO-Qwen3-4B-Code-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-lora_r${LORA_RANK}_lr${ACTOR_LR}_DAPO}"

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
# export WANDB_PROXY=http://star-proxy.oa.com:3128

# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/code_rl_lora8.sh >code_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_rl.sh" "$@"
