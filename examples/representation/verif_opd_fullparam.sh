#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter VerIF instruction-following OPD wrapper around verif_opd.sh.
# Any env var set by the caller still overrides these defaults.
source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd
export WANDB_API_KEY=ce312b01106d2ee09f038f0768661e2dc5f0872f
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Full-parameter distillation is typically less stable with the vector-training LR.
export ACTOR_LR="${ACTOR_LR:-1e-6}"

# Vector artifacts are not needed for full-parameter training.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# VerIF reward LLM judge is OFF by default (rule-only). To enable:
#   export VERIF_LLM_ENABLE=1 IF_LLM_VERIFIER_URL=http://127.0.0.1:8000/v1 IF_LLM_VERIFIER_MODEL=QwQ-32B

# Optional convenience defaults for naming/outputs.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-VerIF-Qwen3-4B-IF-OPD_fullparam_lr${ACTOR_LR}}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/verif_opd/verif_qwen3-4b-opd-fullparam_lr${ACTOR_LR}}"

# nohup sh /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/examples/representation/verif_opd_fullparam.sh >verif_opd_fullparam1e-6.log 2>&1 &
exec bash "${SCRIPT_DIR}/verif_opd.sh" "$@"
