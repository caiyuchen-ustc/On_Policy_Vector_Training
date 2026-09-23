#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/merge.sh ACTOR_CHECKPOINT OUTPUT_HF_MODEL" >&2
    exit 2
fi
exec python -m verl.model_merger merge --backend fsdp --local_dir "$1" --target_dir "$2"
