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

# Ensure the container has an /etc/passwd entry for the current UID so libraries
# relying on getpwuid (e.g., transformers) work even when running with a host UID
# that is unknown inside the image.
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
HOME_DIR="${HOME:-/workspace}"
PASSWD_ENTRY_EXISTS=false

if command -v getent >/dev/null 2>&1; then
    if getent passwd "${USER_ID}" >/dev/null 2>&1; then
        PASSWD_ENTRY_EXISTS=true
    fi
else
    if grep -qE "^([^:]*:){2}${USER_ID}:" /etc/passwd; then
        PASSWD_ENTRY_EXISTS=true
    fi
fi

if [ "${PASSWD_ENTRY_EXISTS}" = "false" ]; then
    echo "UID ${USER_ID} not found in /etc/passwd, attempting to create a synthetic entry"
    USER_NAME="user_${USER_ID}"
    if [ -w /etc/passwd ]; then
        echo "${USER_NAME}:x:${USER_ID}:${GROUP_ID}:Generated User:${HOME_DIR}:/bin/bash" >> /etc/passwd
        echo "Added ${USER_NAME} to /etc/passwd"
    else
        echo "Warning: /etc/passwd is not writable; continuing without adding user entry"
    fi
    export USER="${USER_NAME}"
fi

# Guarantee HOME is set so Python falls back gracefully even if we could not
# create a passwd entry.
export HOME="${HOME_DIR}"

# Normalize Hugging Face authentication environment variables so any one of the
# supported names enables gated repository access for downstream libraries.
if [ -n "${HUGGINGFACE_API_KEY:-}" ]; then
    export HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACE_API_KEY}"
    export HF_TOKEN="${HUGGINGFACE_API_KEY}"
elif [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGINGFACEHUB_API_TOKEN}"
elif [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACEHUB_API_TOKEN="${HF_TOKEN}"
fi


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
if [ -n "${HUGGINGFACE_API_KEY:-}" ]; then
    echo "HUGGINGFACE_API_KEY: ${HUGGINGFACE_API_KEY:0:6}..."
else
    echo "HUGGINGFACE_API_KEY: not provided"
fi
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
