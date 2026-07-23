#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA(rank=8) wrapper around 4bopd.sh.
# Any env var set by the caller still overrides these defaults.
source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Align with official FSDP+vLLM LoRA recommendations.
export MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-false}"

# Rollout speed: pure LoRA training doesn't use the vector steering hook, so the
# eager-mode default from 4bopd.sh (needed only for trainable token vectors) just
# disables CUDA graph and slows generation ~2x. Re-enable CUDA graph here.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
# 4B (bf16 ~8GB) fits on a single GPU at gpu_memory_utilization=0.8, so TP=1
# avoids per-layer all-reduce and doubles rollout data parallelism vs the
# hard-coded TP=2 in 4bopd.sh. Override via GEN_TP if needed.
export GEN_TP="${GEN_TP:-1}"

# Keep loss settings aligned with your full-parameter/vector setup.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}"

# LoRA only trains a small adapter (scaled by alpha/rank), so it needs a much
# higher LR than full-param/vector training. 1e-6 barely moved the weights
# (grad_norm/pg_loss/rollout_probs_diff all flat); 1e-5 matches the working
# 1p5b_rl_lora32 baseline.
export ACTOR_LR="${ACTOR_LR:-5e-6}"

# Vector artifacts are not needed for pure LoRA training.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Optional convenience defaults for naming/outputs.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-repre_qwen3-4b_lora_r${LORA_RANK}_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/g_opd/repre_qwen3-4b-opd-lora_lr${ACTOR_LR}}"
RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/4bopd_lora8.sh >4bopd_lora8.log 2>&1 &
# tensor_model_parallel_size is hard-coded in 4bopd.sh, so override it on the CLI
# (caller args still win since "$@" comes last).
exec bash "${SCRIPT_DIR}/4bopd.sh" \
     actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" "$@"
