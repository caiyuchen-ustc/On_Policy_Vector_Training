#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Multi-vector IFEvalG instruction-following OPD wrapper around ifevalg_opd.sh.
# 与 ifevalg_opd_single.sh 同一模型/数据/OPD 配置（teacher 蒸馏），
# 只把"单向量"换成"每层一组 N 个可训练基向量"，训练它们张成的操控子空间。
#
# 与 4b_opd_multi.sh 对齐的 multi 设置：
#   MODE=multi, learnable_alpha=true + alpha_init=0.1(非零防死锁), curriculum=none,
#   gradient_checkpointing=关, force_all_tokens=true。
#   （model_dtype 在 ifevalg_opd.sh 里硬编码 fp32，天然与 4b 一致；alpha detach bug 已全局修复。）
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8  TRAINABLE_TOKEN_VECTOR_LAYERS=10:20  bash ifevalg_opd_multi.sh
# LAYERS 支持 start:end / all / idx（解析在 ifevalg_opd.sh 里）。
# ============================================================================


export ENABLE_TRAINABLE_TOKEN_VECTOR=${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}
export TRAINABLE_TOKEN_VECTOR_MODE=${TRAINABLE_TOKEN_VECTOR_MODE:-multi}

# ----- 核心：每层基向量个数 N + 插入层（默认与 single 的 20:27 对齐）-----
export TRAINABLE_TOKEN_VECTOR_NUM=${TRAINABLE_TOKEN_VECTOR_NUM:-32}
export TRAINABLE_TOKEN_VECTOR_LAYERS=${TRAINABLE_TOKEN_VECTOR_LAYERS:-10:11}

# ----- 基向量采样 / 初始化 -----
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD=${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}
export TRAINABLE_TOKEN_VECTOR_SCALE=${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}

# ----- 可学习 alpha（强度自适应）；alpha_init 必须非零，否则 basis 梯度(∝alpha)开头为 0 -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA=${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT=${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1.0}

# ----- Curriculum：alpha_then_basis 三阶段 -----
# 阶段1 [0,T1)      : 只训 basis[0]+alpha，系数固定[1,0,..]，学主方向+强度（解决 alpha 暴涨/basis 饿死）
# 阶段2 [T1,T1+T2)  : 冻结 alpha，放开全部 N 基，系数 g*[1,0,..]+(1-g)*随机正系数，g 从 1→0 平滑展开
# 阶段3 [T1+T2,∞)   : 冻结 alpha，纯随机正系数采样，训成"子空间内任意正组合都好"的鲁棒子空间
# 复用参数：T1=WARMUP_STEPS，T2=SECONDARY_FREEZE_STEPS
export TRAINABLE_TOKEN_VECTOR_CURRICULUM=${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-alpha_then_basis}
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-30}
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP=${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-50}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP=${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE=${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE=${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP=${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS=${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-true}

# ----- Vector-only OPD 默认（与 ifevalg_opd_single.sh 对齐）-----
export LORA_RANK=${LORA_RANK:-0}
export ACTOR_LR=${ACTOR_LR:-1e-1}
export SAVE_VECTOR=${SAVE_VECTOR:-true}
export SAVE_FREQ=${SAVE_FREQ:--1}
export OFFLOAD=${OFFLOAD:-false}

# EXPERIMENT_NAME / DEFAULT_LOCAL_DIR / SAVE_VECTOR_DIR 交给主脚本 TRAIN_TAG 逻辑自动生成
# （拼成 multiv{N}_layers{tag}）。只把 multi 向量产物单独存，避免和 single 混。
export SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/ifevalg_opd_multi/trainable_vectors}

# gradient checkpointing：关闭，与 4b_opd_multi 对齐。
# multi 的 hook 在 residual 做随机组合+归一化，与 GC 的 recomputation 冲突（CheckpointError）；
# 模型冻结、只训向量，激活显存需求小，不需要 GC。
GC_OVERRIDE="${GC_OVERRIDE:-actor_rollout_ref.model.enable_gradient_checkpointing=False}"

exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@" ${GC_OVERRIDE}
