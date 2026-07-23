#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# set -x
# export PYTHONUNBUFFERED=1
source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd
# # export TMPDIR="/apdcephfs_zwfy3/share_302867165/ewencai/c"
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DEBUGGER_ENABLED=True
export USED_MODEL="no_api"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG="INFO"
# export RAY_TMPDIR=/apdcephfs_zwfy3/share_302867165/ewencai/ray_tmp
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
#nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/4bopd.sh >opd4bre.log 2>&1 &
AIME24_TEST_FILE=${AIME24_TEST_FILE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/AIME2024/test.parquet"}
AIME25_TEST_FILE=${AIME25_TEST_FILE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/AIME2025/test.parquet"}

test_files="['$AIME24_TEST_FILE', '$AIME25_TEST_FILE']"
MODEL_PATH=${MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B"}
Teacher_MODEL_PATH=${Teacher_MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B-Non-Thinking-RL-Math-Step500"}
ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
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
VLLM_STEER_DEBUG_ONCE=${VLLM_STEER_DEBUG_ONCE:-true}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-true}
TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
ACTOR_LR=${ACTOR_LR:-5e-5}
MODEL_USE_SHM=${MODEL_USE_SHM:-false}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-false}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
ACTOR_ONLY_REVERSE_KL_ADVANTAGES=${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}

SAVE_VECTOR=${SAVE_VECTOR:-true}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors"}

LAYER_TAG=${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}

if [ "$LORA_RANK" -gt 0 ]; then
    TRAIN_TAG="lora_r${LORA_RANK}_lr${ACTOR_LR}"
elif [ "$ENABLE_TRAINABLE_TOKEN_VECTOR" = "true" ]; then
    if [ "$TRAINABLE_TOKEN_VECTOR_MODE" = "multi" ] && [ "$TRAINABLE_TOKEN_VECTOR_NUM" -gt 1 ]; then
        TRAIN_TAG="multiv${TRAINABLE_TOKEN_VECTOR_NUM}_layers${LAYER_TAG}_lr${ACTOR_LR}"
    else
        TRAIN_TAG="singlev_layers${LAYER_TAG}_lr${ACTOR_LR}"
    fi
else
    TRAIN_TAG="fullparam_lr${ACTOR_LR}"
fi

VECTOR_ONLY_TRAINING=false
if [ "$LORA_RANK" -le 0 ] && [ "$ENABLE_TRAINABLE_TOKEN_VECTOR" = "true" ]; then
    VECTOR_ONLY_TRAINING=true
fi

DEFAULT_SAVE_FREQ=10
if [ "$VECTOR_ONLY_TRAINING" = "true" ]; then
    DEFAULT_SAVE_FREQ=-1
fi
SAVE_FREQ=${SAVE_FREQ:-$DEFAULT_SAVE_FREQ}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"repre_qwen3-4b_${TRAIN_TAG}"}
LOCAL_DIR_BASE=${LOCAL_DIR_BASE:-"/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/g_opd/repre_qwen3-4b-opd"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${LOCAL_DIR_BASE}/${TRAIN_TAG}"}
SAVE_VECTOR_DIR=${SAVE_VECTOR_DIR:-"${SAVE_VECTOR_BASE_DIR}/${TRAIN_TAG}"}

LAYER_ARGS=()
case "$TRAINABLE_TOKEN_VECTOR_LAYERS" in
    all)
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=true
            actor_rollout_ref.model.trainable_token_vector_layer_start=null
            actor_rollout_ref.model.trainable_token_vector_layer_end=null
        )
        ;;
    *:*)
        IFS=':' read -r VECTOR_LAYER_START VECTOR_LAYER_END <<< "$TRAINABLE_TOKEN_VECTOR_LAYERS"
        if [ -z "$VECTOR_LAYER_START" ] || [ -z "$VECTOR_LAYER_END" ]; then
            echo "Invalid TRAINABLE_TOKEN_VECTOR_LAYERS=$TRAINABLE_TOKEN_VECTOR_LAYERS, expected start:end or single idx or all" >&2
            exit 1
        fi
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_start=$VECTOR_LAYER_START
            actor_rollout_ref.model.trainable_token_vector_layer_end=$VECTOR_LAYER_END
        )
        ;;
    *)
        VECTOR_LAYER_IDX="$TRAINABLE_TOKEN_VECTOR_LAYERS"
        LAYER_ARGS+=(
            actor_rollout_ref.model.trainable_token_vector_all_layers=false
            actor_rollout_ref.model.trainable_token_vector_layer_idx=$VECTOR_LAYER_IDX
            actor_rollout_ref.model.trainable_token_vector_layer_start=null
            actor_rollout_ref.model.trainable_token_vector_layer_end=null
        )
        ;;
esac

MODEL_ARGS=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=$LORA_RANK
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA
    actor_rollout_ref.model.target_modules=$LORA_TARGET_MODULES
    actor_rollout_ref.model.enable_trainable_token_vector=$ENABLE_TRAINABLE_TOKEN_VECTOR
    actor_rollout_ref.model.trainable_token_vector_mode=$TRAINABLE_TOKEN_VECTOR_MODE
    actor_rollout_ref.model.trainable_token_vector_num=$TRAINABLE_TOKEN_VECTOR_NUM
    actor_rollout_ref.model.trainable_token_vector_sampling_method=$TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD
    actor_rollout_ref.model.trainable_token_vector_scale=$TRAINABLE_TOKEN_VECTOR_SCALE
    actor_rollout_ref.model.trainable_token_vector_learnable_alpha=$TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA
    actor_rollout_ref.model.trainable_token_vector_alpha_init=$TRAINABLE_TOKEN_VECTOR_ALPHA_INIT
    actor_rollout_ref.model.trainable_token_vector_curriculum=$TRAINABLE_TOKEN_VECTOR_CURRICULUM
    actor_rollout_ref.model.trainable_token_vector_warmup_steps=$TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS
    actor_rollout_ref.model.trainable_token_vector_warmup_end_step=$TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP
    actor_rollout_ref.model.trainable_token_vector_secondary_freeze_steps=$TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS
    actor_rollout_ref.model.trainable_token_vector_secondary_end_step=$TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP
    actor_rollout_ref.model.trainable_token_vector_primary_scale=$TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE
    actor_rollout_ref.model.trainable_token_vector_secondary_scale=$TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE
    actor_rollout_ref.model.trainable_token_vector_freeze_primary_after_warmup=$TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP
    actor_rollout_ref.model.trainable_token_vector_force_all_tokens=$TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS
    actor_rollout_ref.model.use_shm=$MODEL_USE_SHM
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.base_model_path="$MODEL_PATH"
    +actor_rollout_ref.ref.model.path="$Teacher_MODEL_PATH"
    "${LAYER_ARGS[@]}"
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=$ACTOR_ONLY_REVERSE_KL_ADVANTAGES
    actor_rollout_ref.actor.ppo_mini_batch_size=1024
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.dtype=$TRAIN_DTYPE
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.calculate_log_probs=true
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.dtype=$TRAIN_DTYPE
    actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT
    actor_rollout_ref.rollout.layered_summon=$ROLLOUT_LAYERED_SUMMON
    actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.n=4
)

REF_ARGS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True
    actor_rollout_ref.ref.fsdp_config.dtype=$TRAIN_DTYPE
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
)

DATA_ARGS=(
    data.train_files=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
    data.val_files="$test_files"
    data.train_batch_size=1024
    data.max_prompt_length=3072
    data.max_response_length=16384
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.seed=42
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.iterative_test_files=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/eff_opd/hard_100.parquet
    data.iterative_test_batch_size=null
    data.iterative_test_max_samples=-1
)

ALGO_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=5.0
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.bypass_mode=false
    algorithm.use_kl_in_reward=False
)

TRAINER_ARGS=(
    trainer.enable_drop_wrong_generations=False
    trainer.save_vector=$SAVE_VECTOR
    trainer.save_vector_dir="$SAVE_VECTOR_DIR"
    trainer.enable_iterative_test=False
    trainer.max_test_iterations=5
    trainer.critic_warmup=0
    trainer.val_before_train=True
    trainer.logger='["console","wandb"]'
    trainer.log_val_generations=10
    trainer.project_name='on-policy-rep-distillation'
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.save_freq=$SAVE_FREQ
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
    trainer.test_freq=10
    trainer.total_epochs=10
)

# Standard OPD 
export PYTHONUNBUFFERED=1 
export VERL_VLLM_STEER_DEBUG_ONCE=$VLLM_STEER_DEBUG_ONCE

python3 -m verl.trainer.main_ppo \
        "${ALGO_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${ACTOR_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${REF_ARGS[@]}" \
        reward_model.reward_manager=naive \
        "${TRAINER_ARGS[@]}" "$@"
