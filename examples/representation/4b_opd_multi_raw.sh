#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# raw_then_double 课程版的 4b OPD multi-vector 训练。
# 是 4b_opd_multi.sh 的 wrapper。
#
# 与 progressive_double 的唯一区别在阶段0：
#   阶段0 [0,T1)：basis[0] 从 0 起步，像 raw single 一样自由学（无 normalize、无 alpha、
#                 可自由增长模长），复现你验证过好使的 single 训练（lr 需求也一致）。
#   切换(step==T1)：alpha := ‖basis[0]‖（主向量学到的模长），basis[0] 归一化为单位向量，
#                   其余基对它正交化。alpha 之后固定不变。
#   阶段1+ ：progressive_double 倍增(active 1→2→4→...→N)+每阶段内 ramp 衰减→随机采样，
#            注入 v=normalize(Σc·basis)·alpha，norm 恒 = alpha（=阶段0学到的主向量模长）。
#
# 好处：alpha 不靠梯度慢慢涨，而是从主向量模长一次性读出，起步不慢；
#       且后续所有采样向量的强度都锁定在"主向量该有的强度"上。
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8 TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS=15 bash 4b_opd_multi_raw.sh
# ============================================================================

# ----- 每层基向量个数 N（最终维度）+ 插入层 -----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-7:8}"

# ----- basis[0] 初始模长：raw 阶段0 用它做起步注入强度（不再从0爬，避免前期磨半天不涨）。
#        vector_scale 在 raw_then_double 里 = basis[0] 初始 norm。设 1.0 让阶段0 一开始就有强度。-----
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-1}"

# ----- learnable_alpha 必须 true（alpha 要作为 Parameter 承接主向量模长）；alpha_init 无所谓(会被覆盖) -----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-1}"

# ----- Curriculum：raw_then_double。T1=WARMUP_STEPS（阶段0/每倍增阶段步数），S_ramp=SECONDARY_FREEZE_STEPS -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-raw_then_double}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-30}"          # 每阶段步数(含阶段0 raw)
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-15}"  # 倍增阶段内 ramp 步数

# 该 curriculum 不用到下面这些（保持默认即可）
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"

# ----- 产物存到 raw 专属目录 -----
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${OPV_ROOT}/repre/opd/repre_qwen3-4b-opd_multi_raw/trainable_vectors}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/4b_opd_multi_raw.sh >4bopd_multi_raw1e-2.log 2>&1 &
# 其余（关 GC、fp32、force_all_tokens=true）沿用 4b_opd_multi.sh 的默认。
exec bash "${SCRIPT_DIR}/4b_opd_multi.sh" "$@"
