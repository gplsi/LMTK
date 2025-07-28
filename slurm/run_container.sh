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
echo "Current user: $(whoami 2>/dev/null || echo 'unknown')"
echo "Current user ID: $(id -u)"
echo "Current group ID: $(id -g)"

# Create home directory if it doesn't exist
if [ ! -d "$HOME" ]; then
    mkdir -p "$HOME" 2>/dev/null || true
    chmod 755 "$HOME" 2>/dev/null || true
fi

# Handle user database issue by setting environment variables
# This prevents transformers from trying to look up user info
if [ -z "$(whoami 2>/dev/null)" ]; then
    echo "⚠️  User database issue detected. Setting environment workarounds..."
    # These environment variables help libraries that need user info
    export PYTORCH_TRANSFORMERS_CACHE="$HOME/.cache/transformers"
    export HF_HOME="$HOME/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HOME/.cache/transformers"
    
    # Disable user-specific features that might cause issues
    export WANDB_CACHE_DIR="/tmp/wandb_cache"
    export WANDB_CONFIG_DIR="/tmp/wandb_config"
    
    # Create WandB directories if they don't exist
    mkdir -p "$WANDB_CACHE_DIR" 2>/dev/null || true
    mkdir -p "$WANDB_CONFIG_DIR" 2>/dev/null || true
    
    # Set additional environment variables that transformers might use
    export PWD=$(pwd)
    export SHELL="/bin/bash"
    
    echo "✅ Environment workarounds applied"
fi

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
# Ensure WandB directories exist (regardless of user database issues)
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
mkdir -p "$WANDB_CACHE_DIR" 2>/dev/null || true
mkdir -p "$WANDB_CONFIG_DIR" 2>/dev/null || true
chmod 755 "$WANDB_CACHE_DIR" 2>/dev/null || true
chmod 755 "$WANDB_CONFIG_DIR" 2>/dev/null || true

if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_API_KEY}" != "" ]; then
    echo "WandB API key provided, logging in..."
    echo "WANDB_PROJECT: ${WANDB_PROJECT:-unknown}"
    echo "WANDB_ENTITY: ${WANDB_ENTITY:-unknown}"
    echo "WANDB_MODE: ${WANDB_MODE:-online}"
    echo "WANDB_CACHE_DIR: $WANDB_CACHE_DIR"
    echo "WANDB_CONFIG_DIR: $WANDB_CONFIG_DIR"
    
    # Login to WandB
    echo "Attempting WandB login..."
    echo "Directory permissions:"
    ls -la "$WANDB_CACHE_DIR" || echo "WANDB_CACHE_DIR not accessible"
    ls -la "$WANDB_CONFIG_DIR" || echo "WANDB_CONFIG_DIR not accessible"
    
    wandb login "$WANDB_API_KEY"
    if [ $? -eq 0 ]; then
        echo "✅ WandB login successful"
    else
        echo "❌ WandB login failed"
        echo "Note: This may be expected for non-training tasks (e.g., dataset_merge)"
        echo "Continuing with execution..."
        # Don't exit on WandB login failure - some tasks don't need WandB
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
