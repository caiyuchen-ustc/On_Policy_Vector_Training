#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
# Full-parameter DAPO code-RL wrapper around code_rl.sh.
# Any env var set by the caller still overrides these defaults.
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Keep baseline RL setup close to the representation reference script.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-256}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

# No vector artifact saving for full-parameter by default.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Client-side sandbox concurrency (global semaphore over all in-flight test cases).
# Keep this aligned with SandboxFusion's server cap (sandbox/configs/local.yaml: max_concurrency).
export SANDBOX_MAX_CONCURRENT="${SANDBOX_MAX_CONCURRENT:-32}"

# Save dir + exp name carry the learning rate (lr in the path).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DAPO-Qwen3-4B-Code-RL_fullparam_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-fullparam_lr${ACTOR_LR}_DAPO}"

# Kill any stale main_ppo process running the same experiment to prevent GPU deadlock.
# (Running two identical experiments simultaneously causes both to hang indefinitely.)
STALE=$(pgrep -f "main_ppo.*${EXPERIMENT_NAME}" 2>/dev/null)
if [ -n "${STALE}" ]; then
    echo "[code_rl_fullparam] WARNING: killing stale main_ppo processes: ${STALE}"
    echo "${STALE}" | xargs kill -9 2>/dev/null || true
    sleep 3
fi

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
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/code_rl_fullparam.sh >code_rl_fullparam_1e-6.log 2>&1 &
exec bash "${SCRIPT_DIR}/code_rl.sh" "$@"
