#!/usr/bin/env bash

#!/bin/bash
# set -x
# export PYTHONUNBUFFERED=1

# # export TMPDIR="/apdcephfs_zwfy3/share_302867165/ewencai/c"
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DEBUGGER_ENABLED=True
export USED_MODEL="no_api"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# export VLLM_ATTENTION_BACKEND=XFORMERS
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

RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external

aime24_test_path=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/dapo/AIME2024/test.parquet
aime25_test_path=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/dapo/AIME2025/test.parquet

test_files="['$aime24_test_path', '$aime25_test_path']"

MODEL_PATH=${MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-14B-Base"}
Teacher_MODEL_PATH=${Teacher_MODEL_PATH:-"/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-14B-RL"}
# Standard OPD 

# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/g_opd/run_qwen3-14b.sh >opd14b.log 2>&1 &
python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/dapo/DeepMath-103K/train_filtered_level6.parquet \
        data.val_files="$test_files" \
        data.train_batch_size=256 \
        data.max_prompt_length=3072 \
        data.max_response_length=12000 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        +actor_rollout_ref.ref.model.path="$Teacher_MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='qwen3_14b_base_teacher_qwen3-14b_rl' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=20 \
        trainer.default_local_dir=/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/g_opd/qwen3_14b_base_teacher_qwen3-14b_rl \
        trainer.test_freq=4 \
        trainer.enable_drop_wrong_generations=True \
        trainer.total_epochs=4 $@