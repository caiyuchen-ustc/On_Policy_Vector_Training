#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-vector RL wrapper around deepseek1p5b_rl.sh, using REINFORCE++ instead
# of GRPO.
#
# Why this variant exists:
#   The GRPO single-vector run (see 1p5b_rl_single.log) does not improve: the
#   binary sciknoweval MCQ reward is normalized *within each prompt group*, so
#   for groups that are all-correct or all-wrong the advantage collapses to ~0,
#   and overall advantages/mean sits at ~0 while entropy craters (0.94 -> 8e-4).
#   REINFORCE++ (arXiv:2501.03262) uses a *global* (whole-batch) baseline via
#   masked_whiten instead of a per-group baseline, so every sample keeps a
#   non-zero advantage relative to the batch-average accuracy (~0.40). This is
#   the one estimator in verl that actually changes *which samples carry signal*
#   (RLOO/OPO are still group-relative; PPO/gae needs a critic and degenerates on
#   single-step outcome rewards).
#
# Caveats (read before trusting the result):
#   * This does NOT raise the expressivity of the single input-agnostic vector.
#     If the bottleneck is capacity (as the LoRA rank=1 comparison suggests), the
#     likely outcome is: entropy stabilizes / grad_norm smooths out, but val
#     reward still stays flat. That outcome is itself diagnostic -> go multi-vector
#     or LoRA.
#   * Global whitening of a binary (0/1) reward produces a two-spike advantage
#     distribution; combined with the original 5e-3 LR this is unstable. We drop
#     ACTOR_LR to 5e-4 and add a small entropy bonus to guard against the entropy
#     collapse seen in the GRPO run.
#
# Read-out after ~30-50 steps:
#   val-core/sciknoweval/reward/mean@4 starts trending up  -> signal-density was
#     the problem, REINFORCE++ helps, keep going.
#   entropy stabilizes but reward stays flat                -> expressivity is the
#     problem, switch to multi-vector (TRAINABLE_TOKEN_VECTOR_NUM>1) or LORA_RANK=1.

export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}
export MODEL_PATH=${MODEL_PATH:-/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B}

export LORA_RANK=${LORA_RANK:-0}
export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-single}
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-0:20}
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# --- REINFORCE++ specifics -------------------------------------------------
# Global-baseline estimator (not group-relative like GRPO/RLOO).
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-reinforce_plus_plus}
# Global whitening + binary reward is spiky; the GRPO run's 5e-3 was already
# collapsing entropy, so default an order of magnitude lower here.
export ACTOR_LR=${ACTOR_LR:-5e-4}
# REINFORCE++ still benefits from batch spread for a stable whitening baseline,
# so keep multiple samples per prompt by default.
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
# IMPORTANT: the base script defaults ENABLE_FILTER_GROUPS=true (DAPO dynamic
# sampling), which DROPS prompt groups that are all-correct/all-wrong by group
# accuracy. That is a GRPO-oriented trick and directly cancels REINFORCE++'s main
# advantage: those very groups still carry non-zero signal against the *global*
# baseline. Turn it off so REINFORCE++ sees the full batch.
export ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-false}
# ---------------------------------------------------------------------------

export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2048}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-256}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}

export SAVE_VECTOR=${SAVE_VECTOR:-true}

# entropy_coeff is hard-coded to 0 in deepseek1p5b_rl.sh (line 286); override it
# via a hydra arg appended after "$@" (main_ppo passes "$@" last, so this wins).
# Guard against the entropy collapse seen in the GRPO single-vector run.
ENTROPY_COEFF=${ENTROPY_COEFF:-0.005}

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek1p5b_brorl_rl_singlev_rpp_layers${LAYER_TAG}_lr${ACTOR_LR}}
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup bash /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/deepseek1p5b_rl_single_rpp.sh >1p5b_rl_single_rpp.log 2>&1 &
exec bash "${SCRIPT_DIR}/deepseek1p5b_rl.sh" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    "$@"
