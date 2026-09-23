#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# progressive_double 几何倍增课程版的 4b OPD multi-vector 训练。
# 是 4b_opd_multi.sh 的 wrapper，把 curriculum 换成 progressive_double。
#
# 子空间"一层层长出来"：active 数 1→2→4→8→...→N，每阶段 S 步。
#   阶段0 [0,S)      : 训 basis[0] + alpha（唯一学 alpha 的阶段），系数=[1,0,..]
#   阶段1 [S,2S)     : 冻 alpha & basis[0]，只训 basis[1]，从前 2 个基采随机正组合
#   阶段2 [2S,3S)    : 冻前 2 个，只训 basis[2:4]，从前 4 个采
#   阶段k            : 冻前 2^(k-1) 个，只训新增块 [2^(k-1), 2^k)，从前 2^k 个采
#   ...直到 active=N，之后停在最后阶段持续训 [N/2, N)
# 每阶段"固定已训向量、只训当前新增向量"，且采样是前 active 个基的随机正组合
# → 每个新向量学会"在已固定子空间上做好的补充"，直接服务"任意正组合都好"。
#
# S = warmup_steps（每阶段步数）。N=32 需要 6 个阶段(1,2,4,8,16,32)，建议 S 给够。
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=32  TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=20  bash 4b_opd_multi_prog.sh
# ============================================================================

# ----- 每层基向量个数 N（最终维度）+ 插入层 -----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-8:9}"

# ----- 可学习 alpha（只阶段0 学，之后冻结）；alpha_init 非零 -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1.0}"

# ----- Curriculum：progressive_double。每阶段 S=WARMUP_STEPS 步，阶段内前 S_ramp 步做 ramp -----
# 每个倍增阶段(active: M->2M) 内部分两半：
#   前 SECONDARY_FREEZE_STEPS 步：ramp，系数 g*[前M基组合]+(1-g)*[前2M基组合]，g 从 1→0（旧子空间主导权褪去）
#   剩余步：纯随机采样前 2M 个基
# 全程只训新增块 [M,2M)；alpha 只阶段0 学。
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-progressive_double}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-30}"          # 每阶段总步数 S
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-20}"  # 阶段内 ramp 步数 S_ramp

# 该 curriculum 不用到下面这些（保持默认即可）
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"

# ----- 产物存到 prog 专属目录 -----
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/opd/repre_qwen3-4b-opd_multi_prog/trainable_vectors}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/4b_opd_multi_prog.sh >4bopd_multi_prog.log 2>&1 &
# 其余（关 GC、fp32、force_all_tokens=true、采样等）沿用 4b_opd_multi.sh 的默认。
exec bash "${SCRIPT_DIR}/4b_opd_multi.sh" "$@"
