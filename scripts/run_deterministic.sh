#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
exec "${PYTHON_COMMAND:-python3}" src/main.py "$@"
