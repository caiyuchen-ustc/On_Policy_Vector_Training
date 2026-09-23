#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# raw_then_double 课程版的 IFEvalG OPD multi-vector 训练。
# 是 ifevalg_opd_multi.sh 的 wrapper，与 4b_opd_multi_raw.sh 完全对称
# （同一套 raw_then_double 课程，只是模型/数据走 ifevalg_opd.sh，DeepSeek-R1-Distill-Qwen-7B）。
#
# 流程：
#   阶段0 [0,T1)：basis[0] 从 0 起步，像 raw single 自由学（无 normalize/alpha，可自由长模长）。
#   切换(step==T1)：alpha := ‖basis[0]‖，basis[0] 归一化，其余基正交化，alpha 之后固定。
#   阶段1+ ：progressive_double 倍增(1→2→4→...→N)+每阶段内 ramp 衰减→随机采样，
#            注入 norm 恒 = alpha（=阶段0学到的主向量模长）。
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8 TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=15 bash ifevalg_opd_multi_raw.sh
# ============================================================================

# ----- 每层基向量个数 N（最终维度）+ 插入层（单层：第 8 层，28 层模型）-----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-8:8}"

# ----- basis[0] 初始模长：raw 阶段0 起步注入强度（vector_scale=basis[0]初始norm）。设1.0避免从0磨。-----
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-1}"

# ----- learnable_alpha 必须 true（alpha 承接主向量模长）；alpha_init 会被切换时覆盖 -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1}"

# ----- Curriculum：raw_then_double。T1=WARMUP_STEPS，S_ramp=SECONDARY_FREEZE_STEPS -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-raw_then_double}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-30}"           # 每阶段步数(含阶段0 raw)
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-28}"  # 倍增阶段内 ramp 步数

# 该 curriculum 不用到下面这些（保持默认即可）
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"

# ----- 产物存到 raw 专属目录 -----
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/ifevalg_opd_multi_raw/trainable_vectors}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/ifevalg_opd_multi_raw.sh >ifevalg_opd_multi_raw__1e-1.log 2>&1 &
# 其余（关 GC、fp32、force_all_tokens=true）沿用 ifevalg_opd_multi.sh 的默认。
exec bash "${SCRIPT_DIR}/ifevalg_opd_multi.sh" "$@"
