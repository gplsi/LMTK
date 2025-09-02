#!/bin/bash

# ===================================================================
# LMTK - Docker Container Execution Script
# ===================================================================
# This script runs inside the Docker container to execute the LMTK
# training or tokenization task. It provides detailed logging and
# proper error handling.
# ===================================================================

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures
set -x  # Print commands as they are executed

echo "===== Container Script Started ====="
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Current user ID: $(id -u)"
echo "Current group ID: $(id -g)"
echo "Python version: $(python3 --version)"
echo "====================================="

echo "===== Environment Variables ====="
echo "CONFIG_FILE: ${CONFIG_FILE:-not set}"
echo "PYTHON_COMMAND: ${PYTHON_COMMAND:-not set}"
echo "MAIN_SCRIPT: ${MAIN_SCRIPT:-not set}"
echo "CONTAINER_PROJECT_ROOT: ${CONTAINER_PROJECT_ROOT:-not set}"
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "=================================="

echo "===== Checking File System ====="
echo "Contents of workspace root:"
ls -la /workspace/
echo ""
echo "Contents of src directory:"
ls -la /workspace/src/ || echo "src directory not found"
echo ""
echo "Configuration file exists:"
ls -la "/workspace/${CONFIG_FILE}" || echo "Config file not found: /workspace/${CONFIG_FILE}"
echo ""
echo "Main script exists:"
ls -la "/workspace/${MAIN_SCRIPT}" || echo "Main script not found: /workspace/${MAIN_SCRIPT}"
echo "================================"

echo "===== Python Environment ====="
echo "Python executable: $(which python3)"
echo "Python path contents:"
python3 -c "import sys; print('\n'.join(sys.path))"
echo ""
echo "Python environment ready"
echo "=============================="

echo "===== WandB Configuration ====="
# Establish robust paths regardless of upstream env
ROOT_DIR="${CONTAINER_PROJECT_ROOT:-/workspace}"
# Force HOME to project root to keep .netrc and wandb files under the mounted workspace
export HOME="$ROOT_DIR"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/.cache/wandb}"
export WANDB_CONFIG_DIR="$WANDB_DIR"
export WANDB_CACHE_DIR="$WANDB_DIR"
mkdir -p "$WANDB_DIR" || true
chmod 0777 "$WANDB_DIR" || true
# Also ensure default fallback path used by wandb CLI exists when writable
if [ -w "$ROOT_DIR" ]; then
    mkdir -p "$ROOT_DIR/wandb" || true
    chmod 0777 "$ROOT_DIR/wandb" || true
fi
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_API_KEY}" != "" ]; then
    echo "WandB API key provided, logging in..."
    echo "WANDB_PROJECT: ${WANDB_PROJECT:-unknown}"
    echo "WANDB_ENTITY: ${WANDB_ENTITY:-unknown}"
    echo "WANDB_MODE: ${WANDB_MODE:-online}"
    
    # Login to WandB
    wandb login "$WANDB_API_KEY"
    if [ $? -eq 0 ]; then
        echo "✅ WandB login successful"
    else
        echo "❌ WandB login failed"
        exit 1
    fi
else
    echo "⚠️  No WandB API key provided - experiments will not be tracked"
    echo "   Set WANDB_MODE to offline if this is intentional"
fi
echo "==============================="

echo "===== Starting Training/Tokenization ====="
echo "Executing: ${PYTHON_COMMAND} ${MAIN_SCRIPT} --config ${CONFIG_FILE}"
echo "Full command: cd /workspace && ${PYTHON_COMMAND} ${MAIN_SCRIPT} --config ${CONFIG_FILE}"
echo "============================================"

# Change to the workspace directory
cd /workspace

# Execute the main command with full output
${PYTHON_COMMAND} ${MAIN_SCRIPT} --config ${CONFIG_FILE}

echo "===== Training/Tokenization Completed ====="
echo "Script finished successfully at: $(date)"
echo "============================================="

# Restore ownership of any created files to the submitting user if provided
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
  echo "Restoring ownership of workspace to UID:GID ${HOST_UID}:${HOST_GID}"
  chown -R ${HOST_UID}:${HOST_GID} /workspace || true
  if [ -n "${OUTPUT_DIR_NAME:-}" ]; then
    chown -R ${HOST_UID}:${HOST_GID} "/workspace/output/${OUTPUT_DIR_NAME}" 2>/dev/null || true
  fi
fi
