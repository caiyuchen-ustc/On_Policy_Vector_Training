#!/usr/bin/env bash

OPV_ROOT="${OPV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPV_LEGACY_DATA_ROOT="${OPV_LEGACY_DATA_ROOT:-${OPV_ROOT}/../data}"
OPV_LEGACY_MODEL_ROOT="${OPV_LEGACY_MODEL_ROOT:-${OPV_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Layer-range full-parameter wrapper around 4bopd.sh.
#
# 只全参训练 decoder 第 [START, END] 层（闭区间，默认 26~32），冻结其余所有层。
# 训练目标仍是 OPD reverse-KL（teacher 蒸馏），但更新的是这几层的全部权重，
# 不是可训练向量、也不是 LoRA。
#
# 依赖 fsdp_workers.py 中的 train_layer_range_only 冻结模式。
#
# 自定义层范围：
#   export TRAIN_LAYER_RANGE_START=26
#   export TRAIN_LAYER_RANGE_END=32
#   bash 4bopd_layer_range.sh
# 或命令行内联：
#   TRAIN_LAYER_RANGE_START=20 TRAIN_LAYER_RANGE_END=35 bash 4bopd_layer_range.sh
# ============================================================================


# ----- 关闭向量训练和 LoRA，走全参（但只解冻指定层范围）-----
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# ----- 只训练哪几层（闭区间，含端点）-----
TRAIN_LAYER_RANGE_START="${TRAIN_LAYER_RANGE_START:-26}"
TRAIN_LAYER_RANGE_END="${TRAIN_LAYER_RANGE_END:-32}"

# 全参层训练用全参级别的小 lr，而不是向量训练的大 lr。
export ACTOR_LR="${ACTOR_LR:-1e-5}"

# 全参训练不需要保存向量。
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# 命名 / 输出目录。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-repre_qwen3-4b_layers${TRAIN_LAYER_RANGE_START}-${TRAIN_LAYER_RANGE_END}_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${OPV_ROOT}/examples/g_opd/repre_qwen3-4b-opd-layers${TRAIN_LAYER_RANGE_START}-${TRAIN_LAYER_RANGE_END}_lr${ACTOR_LR}}"

# 需要保存整模型 checkpoint（不是向量），给个正的 save_freq。
export SAVE_FREQ="${SAVE_FREQ:-20}"

# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external

#nohup sh ${OPV_ROOT}/examples/representation/4bopd_layer_range.sh >4bopd_layers26-321e-5.log 2>&1 &
exec bash "${SCRIPT_DIR}/4bopd.sh" \
    actor_rollout_ref.model.train_layer_range_only=true \
    actor_rollout_ref.model.train_layer_range_start="${TRAIN_LAYER_RANGE_START}" \
    actor_rollout_ref.model.train_layer_range_end="${TRAIN_LAYER_RANGE_END}" \
    "$@"

