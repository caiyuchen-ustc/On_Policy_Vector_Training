#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Multi-vector IFEvalG RL wrapper around ifevalg_rl.sh.
# 与 ifevalg_rl_single.sh 同一模型/数据/RL 配置（DeepSeek-R1-Distill-Qwen-7B），
# 只把"单向量"换成"每层一组 N 个可训练基向量"，训练它们张成的操控子空间。
#
# 与 4b_opd_multi.sh 对齐的 multi 设置：
#   MODE=multi, learnable_alpha=true + alpha_init=0.1(非零防死锁), curriculum=none,
#   MODEL_DTYPE=fp32, gradient_checkpointing=关, force_all_tokens=true。
#   （alpha 的 detach bug 已在 fsdp_workers.py 全局修复，两脚本共享。）
#   若 7B 遇 CPU-RAM/memcg OOM，可临时 export MODEL_DTYPE=bfloat16 缓解。
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8  TRAINABLE_TOKEN_VECTOR_LAYERS=10:20  bash ifevalg_rl_multi.sh
# ============================================================================

export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-multi}

# ----- 核心：每层基向量个数 N + 插入层（默认与 single 的 0:20 对齐）-----
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-4}
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-0:20}

# ----- 基向量采样 / 初始化 -----
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}

# ----- 可学习 alpha（强度自适应）；alpha_init 必须非零，否则 basis 梯度(∝alpha)开头为 0 -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.1}

# ----- Curriculum：关闭，纯 N 维基向量一起训 -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# ----- Vector-only RL 默认（与 ifevalg_rl_single.sh 对齐）-----
export LORA_RANK=${LORA_RANK:-0}
export ACTOR_LR=${ACTOR_LR:-1e-1}
export SAVE_VECTOR=${SAVE_VECTOR:-true}
export SAVE_FREQ=${SAVE_FREQ:--1}
# Trainable token vectors 与 FSDP param/optimizer offload 不兼容（step() 时 optimizer state/
# grad 设备不匹配），保持在 GPU。single 脚本同样设 false。
export OFFLOAD=${OFFLOAD:-false}

# 7B master/optimizer 精度：与 4b 对齐用 fp32（若遇 CPU-RAM/memcg OOM 可临时 export MODEL_DTYPE=bfloat16）。
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
# 长序列每卡 token 预算，收紧以缩小激活/all-gather 缓冲。
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-23552}

# EXPERIMENT_NAME / DEFAULT_LOCAL_DIR / SAVE_VECTOR_DIR 交给主脚本 TRAIN_TAG 逻辑自动生成
# （拼成 vector_multi{N}_layers{tag}_lr{lr}）。只把 multi 向量产物单独存，避免和 single 混。
export SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/ifevalg_rl_multi/trainable_vectors}

# gradient checkpointing：关闭，与 4b_opd_multi 对齐。
# multi 的 hook 在 residual 做随机组合+归一化，与 GC 的 recomputation 冲突（CheckpointError）；
# 模型冻结、只训向量，激活显存需求小，不需要 GC。
GC_OVERRIDE="${GC_OVERRIDE:-actor_rollout_ref.model.enable_gradient_checkpointing=False}"

exec bash "${SCRIPT_DIR}/ifevalg_rl.sh" "$@" ${GC_OVERRIDE}
