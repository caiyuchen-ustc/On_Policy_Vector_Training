#!/usr/bin/env bash
# Upload a new merged checkpoint; the four experiment teachers are already public.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: HF_TOKEN=... bash upload_teacher_models.sh LOCAL_HF_MODEL caiyuchen/REPO" >&2
    exit 2
fi
: "${HF_TOKEN:?Set HF_TOKEN in the environment}"
exec hf upload "$2" "$1" . --repo-type model \
    --include '*.safetensors' '*.json' '*.model' '*.jinja' 'vocab.json' 'merges.txt' 'README.md' \
    --exclude '*/*' --commit-message 'Upload merged teacher checkpoint'
