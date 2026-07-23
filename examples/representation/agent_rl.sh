#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================================
# Workplace-assistant (workbench) agent RL for G-OPD.
#
# Trains a model to call the 27 workplace-assistant tools in the correct order
# to satisfy user requests (email / calendar / analytics / CRM / PM tasks).
# Reward = 1.0 iff the final system state (DataFrames) matches the ground
# truth after all tool calls, else 0.0  — computed by M2RL's is_correct().
#
# Uses verl's SGLang multi-turn rollout (isolation=none, no docker needed):
#   - model generates one tool call per turn
#   - WorkbenchTool executes it against a fresh per-trajectory environment
#   - reward computed at trajectory end
#
# Data:  data/agent/agent_train.parquet + agent_validation.parquet
#        (built by data/agent/convert_agent.py from M2RL's agent.parquet)
#
# Tool config: verl/examples/representation/workbench_tool_config.yaml
#              (27 tools, all backed by WorkbenchTool, isolation=none)
#
# Env-driven three training modes, selected by thin wrappers:
#   * fullparam : LORA_RANK=0, no vector
#   * lora      : LORA_RANK>0
#   * vector    : ENABLE_TRAINABLE_TOKEN_VECTOR=true
# ============================================================================

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

# Cluster/runtime env
export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}
export RAY_DISABLE_MEMORY_MONITOR=${RAY_DISABLE_MEMORY_MONITOR:-1}
export RAY_DEBUGGER_ENABLED=${RAY_DEBUGGER_ENABLED:-True}
export USED_MODEL=${USED_MODEL:-no_api}

export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# M2RL's workbench utils: add the Gym root to PYTHONPATH so WorkbenchTool
# can import resources_servers.* at runtime.
M2RL_GYM=${M2RL_GYM:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/M2RL/M2RL/Gym"}
export PYTHONPATH="${M2RL_GYM}:${PYTHONPATH:-}"

# -----------------------------
# Experiment defaults
# -----------------------------
PROJECT_NAME=${PROJECT_NAME:-Agent_Workbench}
EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-Agent-Qwen3-8B-Workbench-RL}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-8B"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${SCRIPT_DIR}/workbench_tool_config.yaml}"

DATA_ROOT=${DATA_ROOT:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/agent"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/agent_train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_ROOT}/agent_validation.parquet"}
TEST_FILES=${TEST_FILES:-"['${VAL_FILE}']"}

# Cap validation at 100 rows at load time.
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-100}

MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}

# -----------------------------
# Training mode (fullparam / lora / vector)
# -----------------------------
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-10:30}
TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-multi}
TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-10}
TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}
TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}
TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-warmup_expand}
TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-20}
TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-20}
TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-20}
TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-40}
TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-true}
TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}

MODEL_USE_SHM=${MODEL_USE_SHM:-false}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-false}

VECTOR_ONLY_TRAINING=false
if [ "${LORA_RANK}" -le 0 ] && [ "${ENABLE_TRAINABLE_TOKEN_VECTOR}" = "true" ]; then
    VECTOR_ONLY_TRAINING=true
fi

if [ "${LORA_RANK}" -gt 0 ]; then
    TRAIN_TAG="lora_r${LORA_RANK}"
elif [ "${ENABLE_TRAINABLE_TOKEN_VECTOR}" = "true" ]; then
    LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
    TRAIN_TAG="vector_${TRAINABLE_TOKEN_VECTOR_MODE}${TRAINABLE_TOKEN_VECTOR_NUM}_layers${LAYER_TAG}"
else
    TRAIN_TAG="fullparam"
fi

# -----------------------------
# Algorithm / optimization
# -----------------------------
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-false}
KL_COEF=${KL_COEF:-0.0}
USE_KL_LOSS=${USE_KL_LOSS:-false}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}

# Agent prompts are short; responses can be long (multi-turn tool calls).
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 16))}

# Multi-turn rollout settings.
# max_assistant_turns: up to 10 tool calls per trajectory (workbench tasks
# typically need 1-6 tool calls; 10 gives ample room).
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
MAX_USER_TURNS=${MAX_USER_TURNS:-0}      # 0 = no user feedback turns (pure tool loop)
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}    # workbench is sequential
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-512}

TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-16}

TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
ACTOR_LR=${ACTOR_LR:-1e-6}

USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}
INFER_PPO_MAX_TOKEN_LEN_PER_GPU=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$((ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU))
INFER_PPO_MAX_TOKEN_LEN_PER_GPU=$((INFER_PPO_MAX_TOKEN_LEN_PER_GPU))

OFFLOAD=${OFFLOAD:-true}
SP_SIZE=${SP_SIZE:-1}       # sglang multi-turn doesn't use ulysses SP
GEN_TP=${GEN_TP:-4}
FSDP_SIZE=${FSDP_SIZE:-8}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}

REF_MODEL_PATH=${REF_MODEL_PATH:-}

# -----------------------------
# Logging / checkpoint
# -----------------------------
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}

DEFAULT_LOCAL_DIR_BASE=${DEFAULT_LOCAL_DIR_BASE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/agent_rl/repre_qwen3-8b-agent-rl"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${DEFAULT_LOCAL_DIR_BASE}/${TRAIN_TAG}_lr${ACTOR_LR}"}

DEFAULT_SAVE_VECTOR=false
if [ "${VECTOR_ONLY_TRAINING}" = "true" ]; then
    DEFAULT_SAVE_VECTOR=true
fi
SAVE_VECTOR=${SAVE_VECTOR:-${DEFAULT_SAVE_VECTOR}}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/agent_rl/trainable_vectors"}
SAVE_VECTOR_DIR=${SAVE_VECTOR_DIR:-"${SAVE_VECTOR_BASE_DIR}/${TRAIN_TAG}_lr${ACTOR_LR}"}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${EXP_NAME_PREFIX}_${TRAIN_TAG}_lr${ACTOR_LR}"}

LAYER_ARGS=()
case "${TRAINABLE_TOKEN_VECTOR_LAYERS}" in
    all)
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=true
            actor_rollout_ref.model.trainable_token_vector_layer_start=null
            actor_rollout_ref.model.trainable_token_vector_layer_end=null
        )
        ;;
    *:*)
        IFS=':' read -r VECTOR_LAYER_START VECTOR_LAYER_END <<< "${TRAINABLE_TOKEN_VECTOR_LAYERS}"
        if [ -z "${VECTOR_LAYER_START}" ] || [ -z "${VECTOR_LAYER_END}" ]; then
            echo "Invalid TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS}" >&2
            exit 1
        fi
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_start=${VECTOR_LAYER_START}
            actor_rollout_ref.model.trainable_token_vector_layer_end=${VECTOR_LAYER_END}
        )
        ;;
    *)
        VECTOR_LAYER_IDX="${TRAINABLE_TOKEN_VECTOR_LAYERS}"
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_idx=${VECTOR_LAYER_IDX}
            actor_rollout_ref.model.trainable_token_vector_layer_start=null
            actor_rollout_ref.model.trainable_token_vector_layer_end=null
        )
        ;;
esac

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.max_position_embeddings=${MAX_POSITION_EMBEDDINGS}
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    actor_rollout_ref.model.enable_trainable_token_vector=${ENABLE_TRAINABLE_TOKEN_VECTOR}
    actor_rollout_ref.model.trainable_token_vector_mode=${TRAINABLE_TOKEN_VECTOR_MODE}
    actor_rollout_ref.model.trainable_token_vector_num=${TRAINABLE_TOKEN_VECTOR_NUM}
    actor_rollout_ref.model.trainable_token_vector_sampling_method=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD}
    actor_rollout_ref.model.trainable_token_vector_scale=${TRAINABLE_TOKEN_VECTOR_SCALE}
    actor_rollout_ref.model.trainable_token_vector_learnable_alpha=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA}
    actor_rollout_ref.model.trainable_token_vector_alpha_init=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT}
    actor_rollout_ref.model.trainable_token_vector_curriculum=${TRAINABLE_TOKEN_VECTOR_CURRICULUM}
    actor_rollout_ref.model.trainable_token_vector_warmup_steps=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS}
    actor_rollout_ref.model.trainable_token_vector_warmup_end_step=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP}
    actor_rollout_ref.model.trainable_token_vector_secondary_freeze_steps=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS}
    actor_rollout_ref.model.trainable_token_vector_secondary_end_step=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP}
    actor_rollout_ref.model.trainable_token_vector_primary_scale=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE}
    actor_rollout_ref.model.trainable_token_vector_secondary_scale=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE}
    actor_rollout_ref.model.trainable_token_vector_freeze_primary_after_warmup=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP}
    actor_rollout_ref.model.trainable_token_vector_force_all_tokens=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS}
    actor_rollout_ref.model.use_shm=${MODEL_USE_SHM}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "${LAYER_ARGS[@]}"
)

if [ -n "${REF_MODEL_PATH}" ]; then
    MODEL_ARGS+=(+actor_rollout_ref.ref.model.path="${REF_MODEL_PATH}")
fi

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILES}"
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.prompt_key=prompt
    data.truncation=left
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_PROMPT_BSZ}
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=True
)

ALGO_ARGS=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES}
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.fsdp_config.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}
)

# Multi-turn tool-calling rollout via SGLang.
ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}
    actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON}
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=4
    # Multi-turn tool-calling configuration
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}"
    actor_rollout_ref.rollout.multi_turn.format=hermes
)

REF_ARGS=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD}
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True
    actor_rollout_ref.ref.fsdp_config.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}
)

TRAINER_ARGS=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.test_freq=${TEST_FREQ}
    trainer.save_freq=${SAVE_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.enable_iterative_test=False
    trainer.save_vector=${SAVE_VECTOR}
    trainer.save_vector_dir="${SAVE_VECTOR_DIR}"
)

# Reward: the WorkbenchTool computes reward inside the multi-turn rollout
# (calc_reward() called after all turns complete). Use naive reward manager.
REWARD_ARGS=(
    reward_model.reward_manager=naive
)

export PYTHONUNBUFFERED=1
python3 -m verl.trainer.main_ppo \
    "${DATA_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" "$@"
