#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -eq 0 || "$1" == -* ]]; then
    set -- "${TASK:-science}" "$@"
fi
exec "${PYTHON:-python}" "$ROOT_DIR/scripts/train.py" alpha "$@"
