#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================================
# OFFLINE teacher-rollout reverse-KL distillation, IFEvalG-style.
#
# Trains the student to match a frozen teacher on PRE-GENERATED teacher rollouts
# (teacher_sft_all.parquet from gen_teacher_rollouts.sh). No vLLM sampling happens:
# generate_sequences is replaced by an offline loader that reads the teacher text and
# tokenizes it as the rollout batch. Then the standard OPD loop runs:
#   student forward -> old_log_prob (π_student)
#   teacher forward -> ref_log_prob (π_teacher)     [teacher = ref model]
#   advantage = -(old_log_prob - ref_log_prob) = teacher_logp - student_logp   (pure reverse-KL)
#   -> policy loss -> update student
#
# This is "SFT-style distillation" done in log-prob space (reverse KL to the teacher) on the
# offline teacher data, rather than cross-entropy on the teacher tokens.
#
# Pure teacher-student KL (lambda=1.0): NO student base_model_path is set, so dp_actor takes the
# `reverse_kl = old_log_prob - ref_log_prob` branch. Rollout IS correction is OFF (offline data
# has no rollout_log_probs).
#
# Three modes via the wrappers (ifevalg_distill_offline_{fullparam,lora8,single}.sh):
#   * fullparam : LORA_RANK=0, no vector
#   * lora      : LORA_RANK>0
#   * single    : trainable token vector
# ============================================================================


export RAY_DISABLE_MEMORY_MONITOR=${RAY_DISABLE_MEMORY_MONITOR:-1}
export USED_MODEL=${USED_MODEL:-no_api}

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
# Multi-node: some IB drivers reject NCCL's ibv_set_ece (errno 22 Invalid argument),
# which breaks cross-node connect. Disable ECE. If IB still fails, set
# NCCL_IB_DISABLE=1 to fall back to TCP over NCCL_SOCKET_IFNAME (slower but works).

# IFEvalG reward knobs (validation uses the IFEvalG scorer; training signal is reverse-KL).
export IFEVALG_VERBOSE=${IFEVALG_VERBOSE:-0}
export IFEVALG_LOOSE=${IFEVALG_LOOSE:-0}

# steer-vector Adam beta1 override. Unit vectors are re-orthonormalized every step, so the
# flat-space Adam first-moment can beat against that constraint (periodic loss/grad ripple).
# Set STEER_ADAM_BETA1=0 to disable first-moment momentum (keeps second-moment adaptivity).
export STEER_ADAM_BETA1=${STEER_ADAM_BETA1:-0}


PROJECT_NAME=${PROJECT_NAME:-IF_IFEvalG_OfflineDistill}
EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-IFEvalG-Qwen3-4B-IF-OfflineDistill}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Student (initialized from base) + teacher to distill from (= ref model).
MODEL_PATH=${MODEL_PATH:-"${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B"}
# TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${OPV_ROOT}/examples/ifevalg_rl/ifevalg_qwen2.5-7b-lora_r8_lr5e-5/global_step_450/hf_model"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${OPV_ROOT}/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr1e-6/global_step_825/hf_model"}
# Offline teacher rollouts produced by gen_teacher_rollouts.sh + merge_rollouts_to_sft.py.
# This parquet keeps all original columns of the train set (prompt / reward_model / extra_info /
# ...) plus a `response` column, so it doubles as the TRAIN file: the dataloader reads its
# `prompt` for rollout inputs and OfflineTeacherRollout reads its `prompt`->`response` map.
# Using the same file guarantees every training prompt has a matching teacher response.
OFFLINE_TEACHER_DATA_PATH=${OFFLINE_TEACHER_DATA_PATH:-"${OPV_ROOT}/examples/ifevalg_offpolicy/teacher_sft/teacher_sft_all.parquet"}
OFFLINE_TEACHER_RESPONSE_KEY=${OFFLINE_TEACHER_RESPONSE_KEY:-response}

# Train on the teacher-rollout parquet itself.
TRAIN_FILE=${TRAIN_FILE:-"${OFFLINE_TEACHER_DATA_PATH}"}

# Validation set (still needed for val_before_train + end-of-epoch evaluation).
DATA_ROOT=${DATA_ROOT:-"${OPV_LEGACY_DATA_ROOT}/ifevalg"}
VAL_FILE=${VAL_FILE:-"${DATA_ROOT}/ifevalg_validation.parquet"}
TEST_FILES=${TEST_FILES:-"['${VAL_FILE}']"}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-100}

VERIFY_SCRIPT_PATH=${VERIFY_SCRIPT_PATH:-"${OPV_ROOT}/verl/utils/reward_score/ifevalg_verify.py"}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}
ENABLE_THINKING=${ENABLE_THINKING:-false}

# -----------------------------
# Training mode (fullparam / lora / vector)
# -----------------------------
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-8:}
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
TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# sequential_orthogonal curriculum knobs (train one basis vector at a time).
TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS=${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS:-100}
TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD:-0.0}
TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE:-5}
TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS=${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS:-20}

MODEL_USE_SHM=${MODEL_USE_SHM:-false}

VECTOR_ONLY_TRAINING=false
if [ "${LORA_RANK}" -le 0 ] && [ "${ENABLE_TRAINABLE_TOKEN_VECTOR}" = "true" ]; then
    VECTOR_ONLY_TRAINING=true
fi

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}
if [ "${LORA_RANK}" -gt 0 ]; then
    TRAIN_TAG="lora_r${LORA_RANK}"
elif [ "${ENABLE_TRAINABLE_TOKEN_VECTOR}" = "true" ]; then
    TRAIN_TAG="singlev_layers${LAYER_TAG}"
else
    TRAIN_TAG="fullparam"
fi

# -----------------------------
# Algorithm: pure reverse-KL to the teacher, offline rollouts, IS off.
# -----------------------------
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024*3}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024*16}
MAX_PROMPT_LENGTH=$((MAX_PROMPT_LENGTH))
MAX_RESPONSE_LENGTH=$((MAX_RESPONSE_LENGTH))

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
# Offline data: one teacher response per prompt is consumed per step.
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-32768}

TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
ACTOR_LR=${ACTOR_LR:-1e-6}
OFFLOAD=${OFFLOAD:-true}
GEN_TP=${GEN_TP:-2}

# vLLM is not used for sampling in offline mode, but the rollout worker still builds an engine.
# Keep its memory footprint small.
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-true}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
ROLLOUT_VAL_SAMPLES=${ROLLOUT_VAL_SAMPLES:-4}

TEST_FREQ=${TEST_FREQ:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}

DEFAULT_SAVE_FREQ=10
if [ "${VECTOR_ONLY_TRAINING}" = "true" ]; then
    DEFAULT_SAVE_FREQ=-1
fi
SAVE_FREQ=${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}

DEFAULT_SAVE_VECTOR=false
if [ "${VECTOR_ONLY_TRAINING}" = "true" ]; then
    DEFAULT_SAVE_VECTOR=true
fi
SAVE_VECTOR=${SAVE_VECTOR:-${DEFAULT_SAVE_VECTOR}}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${EXP_NAME_PREFIX}_${TRAIN_TAG}_lr${ACTOR_LR}"}
LOCAL_DIR_BASE=${LOCAL_DIR_BASE:-"${OPV_ROOT}/examples/ifevalg_offline_distill"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${LOCAL_DIR_BASE}/${TRAIN_TAG}_lr${ACTOR_LR}"}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"${LOCAL_DIR_BASE}/trainable_vectors"}
SAVE_VECTOR_DIR=${SAVE_VECTOR_DIR:-"${SAVE_VECTOR_BASE_DIR}/${TRAIN_TAG}_lr${ACTOR_LR}full"}

# Layer args for the token-vector mode.
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
        # Open-ended range: "8:" -> layers [8, last]; ":20" -> layers [0, 20]. The token-vector
        # hook installer requires both start and end (it does not accept one-sided None), so map
        # an empty side to 0 (start) / -1 (end, normalized to the last layer).
        VECTOR_LAYER_START=${VECTOR_LAYER_START:-0}
        VECTOR_LAYER_END=${VECTOR_LAYER_END:--1}
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_start=${VECTOR_LAYER_START}
            actor_rollout_ref.model.trainable_token_vector_layer_end=${VECTOR_LAYER_END}
        )
        ;;
    *)
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_idx=${TRAINABLE_TOKEN_VECTOR_LAYERS}
            actor_rollout_ref.model.trainable_token_vector_layer_start=null
            actor_rollout_ref.model.trainable_token_vector_layer_end=null
        )
        ;;
esac

# NOTE: NO base_model_path is set (student or ref), so dp_actor uses the pure reverse-KL branch
# advantage = old_log_prob - ref_log_prob  (teacher_logp - student_logp).
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
    actor_rollout_ref.model.trainable_token_vector_seq_max_iters=${TRAINABLE_TOKEN_VECTOR_SEQ_MAX_ITERS}
    actor_rollout_ref.model.trainable_token_vector_seq_loss_threshold=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_THRESHOLD}
    actor_rollout_ref.model.trainable_token_vector_seq_loss_patience=${TRAINABLE_TOKEN_VECTOR_SEQ_LOSS_PATIENCE}
    actor_rollout_ref.model.trainable_token_vector_seq_raw_steps=${TRAINABLE_TOKEN_VECTOR_SEQ_RAW_STEPS}
    actor_rollout_ref.model.use_shm=${MODEL_USE_SHM}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL_PATH}"
    "${LAYER_ARGS[@]}"
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.calculate_log_probs=false
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.rollout.enforce_eager=${VLLM_ENFORCE_EAGER}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_SAMPLES}
    actor_rollout_ref.rollout.offline_teacher_rollout=true
    actor_rollout_ref.rollout.offline_teacher_data_path="${OFFLINE_TEACHER_DATA_PATH}"
    actor_rollout_ref.rollout.offline_teacher_response_key=${OFFLINE_TEACHER_RESPONSE_KEY}
)

REF_ARGS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True
    actor_rollout_ref.ref.fsdp_config.dtype=${TRAIN_DTYPE}
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
)

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILES}"
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.prompt_key=prompt
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.seed=42
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}
)

# Offline data has no rollout_log_probs, so disable rollout importance-sampling correction.
ALGO_ARGS=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.bypass_mode=false
    algorithm.use_kl_in_reward=False
)

TRAINER_ARGS=(
    trainer.enable_drop_wrong_generations=False
    trainer.save_vector=${SAVE_VECTOR}
    trainer.save_vector_dir="${SAVE_VECTOR_DIR}"
    trainer.enable_iterative_test=False
    trainer.critic_warmup=0
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.logger='["console","wandb"]'
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

REWARD_ARGS=(
    reward_model.reward_manager=naive
    custom_reward_function.path="${VERIFY_SCRIPT_PATH}"
    custom_reward_function.name=compute_score
)

export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
    "${ALGO_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" "$@"
