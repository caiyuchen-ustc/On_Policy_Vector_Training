#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# alpha_then_basis 三阶段课程版的 4b OPD multi-vector 训练。
# 是 4b_opd_multi.sh 的 wrapper，只把 curriculum 从 none 换成 alpha_then_basis，
# 用于"先学好主方向+强度、再固定强度展开成鲁棒正交子空间"（解决 alpha 暴涨 / basis 饿死）。
#
# 三阶段（复用 WARMUP_STEPS=T1, SECONDARY_FREEZE_STEPS=T2）：
#   阶段1 [0,T1)      : 只训 basis[0]+alpha，系数固定[1,0,..]，学主方向+强度
#   阶段2 [T1,T1+T2)  : 冻结 alpha，放开全部 N 基，系数 g*[1,0,..]+(1-g)*随机正系数，g 从 1→0 平滑展开
#   阶段3 [T1+T2,∞)   : 冻结 alpha，纯随机正系数采样，训成"子空间内任意正组合都好"的鲁棒子空间
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8  TRAINABLE_TOKEN_VECTOR_LAYERS=10:30  bash 4b_opd_multi_atb.sh
#   TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=40 TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS=60 ...
# ============================================================================

# ----- 每层基向量个数 N + 插入层 -----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-8:9}"

# ----- 可学习 alpha（阶段1 学强度；阶段2/3 冻结）；alpha_init 非零 -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1.0}"

# ----- Curriculum：alpha_then_basis 三阶段。T1=WARMUP_STEPS, T2=SECONDARY_FREEZE_STEPS -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-alpha_then_basis}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-30}"      # T1: 阶段1时长
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-15}"  # T2: 阶段2 ramp 时长

# 该 curriculum 不用到下面这些（保持默认即可）
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"

# ----- 产物存到 atb 专属目录，避免和 curriculum=none 的 multi 混 -----
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/opd/repre_qwen3-4b-opd_multi_atb/trainable_vectors}"
RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# 其余（关 GC、fp32、force_all_tokens=true、采样等）沿用 4b_opd_multi.sh 的默认。
exec bash "${SCRIPT_DIR}/4b_opd_multi.sh" "$@"
