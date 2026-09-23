#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# DeepSeek-R1-Distill-Qwen-1.5B 单层多向量 OPD 训练（raw_then_double 课程）。
# 与 deepseek1p5b_opd_single.sh 同一模型 / teacher / 数据，只把"单向量"换成
# "单层一组 N 个可训练基向量"，训练它们张成的操控子空间。
#
# 流程（raw_then_double）：
#   阶段0 [0,T1)：所有基从 0 起步，只训 basis[0]，像 raw single 一样自由学
#                 （无 normalize / 无 alpha，模长自由长）。其余基 *0 不参与。
#   切换(step==T1)：alpha := ‖basis[0]‖，basis[0] 归一化，其余基正交化，alpha 之后固定。
#   阶段1+ ：几何倍增 active 1->2->4->...->N，每阶段只训新增块，
#            每阶段内 ramp 衰减 -> 随机采样；注入 norm 恒 = alpha。
#   训 / 推（vLLM rollout、old_logp、new_logp）同一 global_step 下用同一组
#   确定性系数（seed 重放），逐 iteration 更新，每层独立。
#
# 自定义：
#   TRAINABLE_TOKEN_VECTOR_NUM=8  TRAINABLE_TOKEN_VECTOR_LAYERS=14:14 \
#   bash deepseek1p5b_opd_multi_raw.sh
# ============================================================================

export MODEL_PATH="${MODEL_PATH:-${OPV_LEGACY_MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export Teacher_MODEL_PATH="${Teacher_MODEL_PATH:-${OPV_ROOT}/deepseek1p5b_brorl_rl/checkpoints/fullparam_lr1e-6/global_step_820/hf_model}"

RUN_ROOT="${RUN_ROOT:-${OPV_ROOT}/deepseek1p5b_rl_opd}"

export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-true}"
export TRAINABLE_TOKEN_VECTOR_MODE="${TRAINABLE_TOKEN_VECTOR_MODE:-multi}"

# ----- 核心：每层基向量个数 N + 插入的单层（28 层模型取中间第 14 层）-----
export TRAINABLE_TOKEN_VECTOR_NUM="${TRAINABLE_TOKEN_VECTOR_NUM:-64}"
export TRAINABLE_TOKEN_VECTOR_LAYERS="${TRAINABLE_TOKEN_VECTOR_LAYERS:-7:8}"

# ----- 基向量采样 / 初始化（所有基从 0 起步，raw 阶段 basis[0] 自由学模长）-----
export TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD="${TRAINABLE_TOKEN_VECTOR_SAMPLING_METHOD:-hypersphere}"
export TRAINABLE_TOKEN_VECTOR_SCALE="${TRAINABLE_TOKEN_VECTOR_SCALE:-0.1}"

# ----- learnable_alpha 必须 true（切换时 alpha := ‖basis[0]‖ 承接主向量模长）-----
export TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA="${TRAINABLE_TOKEN_VECTOR_LEARNABLE_ALPHA:-true}"
export TRAINABLE_TOKEN_VECTOR_ALPHA_INIT="${TRAINABLE_TOKEN_VECTOR_ALPHA_INIT:-0.1}"

# ----- Curriculum：raw_then_double。T1=WARMUP_STEPS（每阶段步数），S_ramp=SECONDARY_FREEZE_STEPS -----
export TRAINABLE_TOKEN_VECTOR_CURRICULUM="${TRAINABLE_TOKEN_VECTOR_CURRICULUM:-raw_then_double}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS="${TRAINABLE_TOKEN_VECTOR_WARMUP_STEPS:-60}"
export TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP="${TRAINABLE_TOKEN_VECTOR_WARMUP_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS="${TRAINABLE_TOKEN_VECTOR_SECONDARY_FREEZE_STEPS:-40}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP="${TRAINABLE_TOKEN_VECTOR_SECONDARY_END_STEP:-null}"
export TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE="${TRAINABLE_TOKEN_VECTOR_PRIMARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE="${TRAINABLE_TOKEN_VECTOR_SECONDARY_SCALE:-1.0}"
export TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP="${TRAINABLE_TOKEN_VECTOR_FREEZE_PRIMARY_AFTER_WARMUP:-false}"
export TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS="${TRAINABLE_TOKEN_VECTOR_FORCE_ALL_TOKENS:-false}"

# ----- Vector-only OPD 默认（与 single 对齐）-----
export ACTOR_LR="${ACTOR_LR:-1e-1}"
export SAVE_VECTOR="${SAVE_VECTOR:-true}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek1p5b_rl_opd_multiv${TRAINABLE_TOKEN_VECTOR_NUM}_layers${TRAINABLE_TOKEN_VECTOR_LAYERS//:/-}_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${RUN_ROOT}/checkpoints}"
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${RUN_ROOT}/trainable_vectors_multi_raw}"
RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh ${OPV_ROOT}/examples/representation/deepseek1p5b_opd_multi_raw.sh >1p5b_opd_multi_raw.log 2>&1 &
exec bash "${SCRIPT_DIR}/1p5bopd.sh" "$@"
