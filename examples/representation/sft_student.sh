#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================================
# Student SFT on teacher-generated rollouts (off-policy distillation, step 2).
#
# Consumes the merged parquet produced by gen_teacher_rollouts.sh + merge_rollouts_to_sft.py
# (columns: prompt [chat list] + response [teacher text]) and fine-tunes the student with
# verl.trainer.fsdp_sft_trainer. Three modes are selected by the wrapper scripts via env:
#   * fullparam : LORA_RANK=0, no vector  (STRATEGY=fsdp2)
#   * lora      : LORA_RANK>0             (STRATEGY=fsdp2)
#   * single    : trainable token vector  (STRATEGY=fsdp, base frozen)
#
# Any env var set by the caller still overrides these defaults.
# ============================================================================

export PYTHONUNBUFFERED=1

# -----------------------------
# Cluster / distributed
# -----------------------------
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}

# -----------------------------
# Student model + data
# -----------------------------
MODEL_PATH=${MODEL_PATH:-"${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B"}

DATA_DIR=${DATA_DIR:-"${OPV_ROOT}/examples/ifevalg_offpolicy/teacher_sft"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/teacher_sft_all.parquet"}
VAL_FILE=${VAL_FILE:-"${TRAIN_FILE}"}
PROMPT_KEY=${PROMPT_KEY:-prompt}
RESPONSE_KEY=${RESPONSE_KEY:-response}

MAX_LENGTH=${MAX_LENGTH:-$((1024*11))}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-200}
ENABLE_THINKING=${ENABLE_THINKING:-false}

# -----------------------------
# Training mode knobs (set by wrappers)
# -----------------------------
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}
TRAINABLE_TOKEN_VECTOR_LAYER_IDX=${TRAINABLE_TOKEN_VECTOR_LAYER_IDX:-16}
TRAINABLE_TOKEN_VECTOR_ALL_LAYERS=${TRAINABLE_TOKEN_VECTOR_ALL_LAYERS:-false}
TRAINABLE_TOKEN_VECTOR_LAYER_START=${TRAINABLE_TOKEN_VECTOR_LAYER_START:-null}
TRAINABLE_TOKEN_VECTOR_LAYER_END=${TRAINABLE_TOKEN_VECTOR_LAYER_END:-null}
TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-1}
TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}
TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# fullparam/lora default to fsdp2; single-vector needs fsdp1 (use_orig_params).
if [ "${ENABLE_TRAINABLE_TOKEN_VECTOR}" = "true" ]; then
    STRATEGY=${STRATEGY:-fsdp}
else
    STRATEGY=${STRATEGY:-fsdp2}
fi

# -----------------------------
# Optim / logging / checkpoint
# -----------------------------
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:--1}
# test_freq<=0 disables step-based validation; validation still runs on the base model before
# training (VAL_BEFORE_TRAIN) and at the end of every epoch.
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
ULYSSES_SP=${ULYSSES_SP:-1}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}

PROJECT_NAME=${PROJECT_NAME:-IF_IFEvalG_OffPolicy_SFT}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ifevalg-student-sft}
SAVE_PATH=${SAVE_PATH:-"${OPV_ROOT}/examples/ifevalg_offpolicy/sft/${EXPERIMENT_NAME}"}

torchrun --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --node-rank=${NODE_RANK} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=${PROMPT_KEY} \
    data.response_key=${RESPONSE_KEY} \
    data.max_length=${MAX_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.truncation=right \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    model.partial_pretrain="${MODEL_PATH}" \
    model.strategy=${STRATEGY} \
    model.lora_rank=${LORA_RANK} \
    model.lora_alpha=${LORA_ALPHA} \
    model.target_modules=${LORA_TARGET_MODULES} \
    model.enable_trainable_token_vector=${ENABLE_TRAINABLE_TOKEN_VECTOR} \
    model.trainable_token_vector_layer_idx=${TRAINABLE_TOKEN_VECTOR_LAYER_IDX} \
    model.trainable_token_vector_all_layers=${TRAINABLE_TOKEN_VECTOR_ALL_LAYERS} \
    model.trainable_token_vector_layer_start=${TRAINABLE_TOKEN_VECTOR_LAYER_START} \
    model.trainable_token_vector_layer_end=${TRAINABLE_TOKEN_VECTOR_LAYER_END} \
    model.trainable_token_vector_num=${TRAINABLE_TOKEN_VECTOR_NUM} \
    model.trainable_token_vector_sampling_method=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD} \
    model.trainable_token_vector_scale=${TRAINABLE_TOKEN_VECTOR_SCALE} \
    model.trainable_token_vector_force_all_tokens=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS} \
    optim.lr=${LR} \
    ulysses_sequence_parallel_size=${ULYSSES_SP} \
    use_remove_padding=${USE_REMOVE_PADDING} \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} "$@"
