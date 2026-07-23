#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) VerIF instruction-following RL wrapper around verif_rl.sh.
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
export ACTOR_LR="${ACTOR_LR:-1e-5}"

# LoRA does not need vector artifacts.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# VerIF reward LLM judge is OFF by default (rule-only). To enable:
#   export VERIF_LLM_ENABLE=1 IF_LLM_VERIFIER_URL=http://127.0.0.1:8000/v1 IF_LLM_VERIFIER_MODEL=QwQ-32B

# Save dir + exp name carry the LoRA rank and learning rate (lr in the path).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-VerIF-Qwen3-4B-IF-RL_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/verif_rl/verif_qwen3-4b-lora_r${LORA_RANK}_lr${ACTOR_LR}}"

# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/verif_rl_lora8.sh >verif_rl_lora8.log 2>&1 &
exec bash "${SCRIPT_DIR}/verif_rl.sh" "$@"
