#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Multi-vector wrapper around 4bopd.sh.
#
# 每层插入一组 N 个可训练基向量 (basis vectors)，冻结整个模型，只训练这组基。
# 每次 forward 从这 N 个基张成的子空间里采样一个方向加到 hidden states 上，
# 用 OPD reverse-KL (teacher 蒸馏) 目标只更新这些基向量。
#
# 自定义方式（任意一个都能覆盖默认值）：
#   1) 直接改下面的默认值
#   2) 运行前 export，例如：
#        export TRAINABLE_TOKEN_VECTOR_NUM=16          # 每层 16 个基向量
#        export TRAINABLE_TOKEN_VECTOR_LAYERS=8:32      # 第 8~32 层每层各一组
#        bash 4b_opd_multi.sh
#   3) 命令行内联：
#        TRAINABLE_TOKEN_VECTOR_NUM=4 bash 4b_opd_multi.sh
#
# TRAINABLE_TOKEN_VECTOR_LAYERS 支持三种写法：
#   start:end   -> 第 start~end 层每层各插一组 N 个基向量（含端点）
#   all         -> 所有 decoder 层都插
#   idx         -> 只在第 idx 层插一组
# ============================================================================

# ----- 核心：每层基向量个数 N（自己设定）-----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-16}"

# ----- 插入哪些层 -----
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-8:9}"

# ----- 开启可训练向量、走 multi 模式 -----
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-multi}"

# ----- 基向量采样 / 初始化 -----
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"

# ----- 可学习 alpha（缩放标量，强度自适应）-----
# 注意 alpha_init 必须非零：v = normalize(Σ coeffs·basis)·alpha，
#   若 alpha_init=0 则 basis 梯度(∝alpha)开头为 0，basis 起步不动（只有 alpha 先动），拖慢收敛。
#   给一个小的非零初值让 basis 和 alpha 从第 0 步一起学。
# （前提：已修复 get_effective_alpha 的 detach bug，使 alpha 梯度能回传。）
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1.0}"

# ----- Curriculum：关闭。纯 N 维基向量训练（不做"先1维再展开"的课程），
#        用于干净测量 RL 能力维度（rank 饱和扫描）。想开课程改回 warmup_expand。-----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-none}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"

# ----- token 覆盖范围：false 只加到 response token；true 加到所有 token -----
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}"

# ----- 学到的基向量保存位置 -----
SAVE_VECTOR=${SAVE_VECTOR:-true}
SAVE_VECTOR_BASE_DIR=${SAVE_VECTOR_BASE_DIR:-"${OPV_ROOT}/repre/opd/repre_qwen3-4b-opd_multi/trainable_vectors"}
export SAVE_VECTOR SAVE_VECTOR_BASE_DIR
# ----- 关闭 gradient checkpointing -----
# multi 模式的 hook 在 residual 上做"随机组合基向量+归一化"，会与 gradient checkpointing 的
# recomputation 冲突（forward 与 backward 重算的计算图张量数不一致 → CheckpointError:
# "A different number of tensors was saved during the original forward and recomputation"）。
# 本实验模型冻结、只训向量，激活显存需求小，不需要 checkpointing，直接关掉即可。
GC_OVERRIDE="actor_rollout_ref.model.enable_gradient_checkpointing=False"

# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
#nohup sh ${OPV_ROOT}/examples/representation/4b_opd_multi.sh >4bopd_multi_1e-1_8_9_16.log 2>&1 &
exec bash "${SCRIPT_DIR}/4bopd.sh" "$@" "${GC_OVERRIDE}"
