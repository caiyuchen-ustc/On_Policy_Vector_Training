#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

# Cluster/runtime env
export WANDB_API_KEY=${WANDB_API_KEY:-ce312b01106d2ee09f038f0768661e2dc5f0872f}
export RAY_DISABLE_MEMORY_MONITOR=${RAY_DISABLE_MEMORY_MONITOR:-1}
export RAY_DEBUGGER_ENABLED=${RAY_DEBUGGER_ENABLED:-True}
export USED_MODEL=${USED_MODEL:-no_api}

# Network env
export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}

# Common hardware/env knobs
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# -----------------------------
# Experiment defaults
# -----------------------------
PROJECT_NAME=${PROJECT_NAME:-on-policy-rep-distillation-1.5B}
EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-deepseek1p5b-brorl-rl}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B"}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}

# -----------------------------
# Data/validation config (aligned with 1p5bopd.sh)
# -----------------------------
SCIKNOWEVAL_TRAIN_FILE=${SCIKNOWEVAL_TRAIN_FILE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/sciknoweval/sciknoweval_train.parquet"}
SCIKNOWEVAL_VAL_FILE=${SCIKNOWEVAL_VAL_FILE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/sciknoweval/sciknoweval_validation.parquet"}

TRAIN_FILE=${TRAIN_FILE:-"${SCIKNOWEVAL_TRAIN_FILE}"}
VAL_FILE=${VAL_FILE:-"${SCIKNOWEVAL_VAL_FILE}"}
TEST_FILES=${TEST_FILES:-"['${VAL_FILE}']"}
ITERATIVE_TEST_FILE=${ITERATIVE_TEST_FILE:-"${VAL_FILE}"}

VERIFY_SCRIPT_PATH=${VERIFY_SCRIPT_PATH:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/verl/utils/reward_score/sciknoweval_verify.py"}

# -----------------------------
# Training mode (fullparam / lora / vector)
# -----------------------------
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-0:20}
TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-single}
TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-false}
TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.0}
TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}
TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}
TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}
TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}

# Train only attention q_proj bias (native model params; mutually exclusive with token vector)
TRAIN_Q_BIAS_ONLY=${TRAIN_Q_BIAS_ONLY:-false}
Q_BIAS_LAYER_START=${Q_BIAS_LAYER_START:-null}
Q_BIAS_LAYER_END=${Q_BIAS_LAYER_END:-null}

# Train all params of a single decoder layer only (freeze all other layers)
TRAIN_SINGLE_LAYER_ONLY=${TRAIN_SINGLE_LAYER_ONLY:-false}
TRAIN_LAYER_IDX=${TRAIN_LAYER_IDX:-null}

# Train only the MLP block of a single decoder layer (freeze everything else)
TRAIN_SINGLE_MLP_ONLY=${TRAIN_SINGLE_MLP_ONLY:-false}
TRAIN_MLP_LAYER_IDX=${TRAIN_MLP_LAYER_IDX:-null}

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
ALTERNATE_ADV_SIGN=${ALTERNATE_ADV_SIGN:-false}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-false}
KL_COEF=${KL_COEF:-0.0}
USE_KL_LOSS=${USE_KL_LOSS:-false}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-false}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_PROMPT_LENGTH=$((MAX_PROMPT_LENGTH))
MAX_RESPONSE_LENGTH=$((MAX_RESPONSE_LENGTH))

ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-true}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}
MAX_NUM_GEN_BATCHES=$((MAX_NUM_GEN_BATCHES))

# Keep 1.5B RL defaults aligned with your expected setup.
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
TRAIN_PROMPT_BSZ=$((TRAIN_PROMPT_BSZ))
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-$((TRAIN_PROMPT_BSZ * 1))}
GEN_PROMPT_BSZ=$((GEN_PROMPT_BSZ))
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-32}

TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.1}

USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
INFER_PPO_MAX_TOKEN_LEN_PER_GPU=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$((ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU))
INFER_PPO_MAX_TOKEN_LEN_PER_GPU=$((INFER_PPO_MAX_TOKEN_LEN_PER_GPU))
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

OFFLOAD=${OFFLOAD:-false}
SP_SIZE=${SP_SIZE:-1}
GEN_TP=${GEN_TP:-1}
FSDP_SIZE=${FSDP_SIZE:-8}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-1.0}

REF_MODEL_PATH=${REF_MODEL_PATH:-}

# -----------------------------
# Logging / checkpoint
# -----------------------------
RUN_ROOT=${RUN_ROOT:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/deepseek1p5b_brorl_rl"}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-3}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}
ENABLE_ITERATIVE_TEST=${ENABLE_ITERATIVE_TEST:-false}
MAX_TEST_ITERATIONS=${MAX_TEST_ITERATIONS:-5}
DATA_SEED=${DATA_SEED:-42}

DEFAULT_LOCAL_DIR_BASE=${DEFAULT_LOCAL_DIR_BASE:-"${RUN_ROOT}/checkpoints"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${DEFAULT_LOCAL_DIR_BASE}/${TRAIN_TAG}_lr${ACTOR_LR}"}

DEFAULT_SAVE_VECTOR=false
if [ "${VECTOR_ONLY_TRAINING}" = "true" ]; then
    DEFAULT_SAVE_VECTOR=true
fi
SAVE_VECTOR=${SAVE_VECTOR:-${DEFAULT_SAVE_VECTOR}}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"${RUN_ROOT}/trainable_vectors"}
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
            echo "Invalid TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS}, expected start:end or single idx or all" >&2
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
    actor_rollout_ref.model.train_q_bias_only=${TRAIN_Q_BIAS_ONLY}
    actor_rollout_ref.model.q_bias_layer_start=${Q_BIAS_LAYER_START}
    actor_rollout_ref.model.q_bias_layer_end=${Q_BIAS_LAYER_END}
    actor_rollout_ref.model.train_single_layer_only=${TRAIN_SINGLE_LAYER_ONLY}
    actor_rollout_ref.model.train_layer_idx=${TRAIN_LAYER_IDX}
    actor_rollout_ref.model.train_single_mlp_only=${TRAIN_SINGLE_MLP_ONLY}
    actor_rollout_ref.model.train_mlp_layer_idx=${TRAIN_MLP_LAYER_IDX}
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
    data.prompt_key=prompt
    data.truncation=error
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_PROMPT_BSZ}
    +data.gen_batch_size=${GEN_PROMPT_BSZ}
    data.filter_overlong_prompts=True
    data.shuffle=True
    data.seed=${DATA_SEED}
    data.return_raw_chat=True
    data.iterative_test_files="${ITERATIVE_TEST_FILE}"
    data.iterative_test_batch_size=null
    data.iterative_test_max_samples=-1
)

ALGO_ARGS=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.alternate_adv_sign_by_batch=${ALTERNATE_ADV_SIGN}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    +algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS}
    +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}
    +algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY}
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

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
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

REWARD_ARGS=(
    reward_model.reward_manager=naive
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
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.enable_iterative_test=${ENABLE_ITERATIVE_TEST}
    trainer.max_test_iterations=${MAX_TEST_ITERATIONS}
    trainer.save_vector=${SAVE_VECTOR}
    trainer.save_vector_dir="${SAVE_VECTOR_DIR}"
)

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    TRAINER_ARGS+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

export PYTHONUNBUFFERED=1
python3 -m verl.trainer.main_ppo \
    "${DATA_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    custom_reward_function.path="${VERIFY_SCRIPT_PATH}" \
    custom_reward_function.name=compute_score \
    "${REWARD_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" "$@"
