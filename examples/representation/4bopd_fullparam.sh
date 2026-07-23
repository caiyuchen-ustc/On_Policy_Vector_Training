#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter wrapper around 4bopd.sh.
# Any env var set by the caller still overrides these defaults.
source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Full-parameter training is typically less stable with the vector-training LR.
export ACTOR_LR="${ACTOR_LR:-1e-7}"

# Vector artifacts are not needed for full-parameter training.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Optional convenience defaults for naming/outputs.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-repre_qwen3-4b_fullparam_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/g_opd/rlrlr_repre_qwen3-4b-opd-fullparam_lr${ACTOR_LR}}"
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/4bopd_fullparam.sh >4bopd_fullparam1e-7.log 2>&1 &
exec bash "${SCRIPT_DIR}/4bopd.sh" "$@"

