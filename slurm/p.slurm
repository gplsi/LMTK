#!/bin/bash

# ===================================================================
# LMTK - Generic SLURM Job Script for Production
# ===================================================================
# This script is designed to be generic and production-ready:
# - Only requires CONFIG_FILE parameter
# - WandB key is optional with proper warnings
# - Handles any experiment configuration dynamically
# - Follows SLURM best practices
# - No hardcoded values except sensible defaults
# ===================================================================

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# ===================================================================
# CONFIGURABLE VARIABLES
# ===================================================================

# --- Required Configuration ---
CONFIG_FILE="${CONFIG_FILE:-}"

# --- Job Configuration (passed from submit_job.sh or defaults) ---
JOB_NAME="${JOB_NAME:-lmtk}"
PARTITION="${PARTITION:-dgx}"
GPU_COUNT="${GPU_COUNT:-2}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
NODELIST="${NODELIST:-}"

# --- Project Paths (auto-detected) ---
PROJECT_NAME="${PROJECT_NAME:-LMTK}"
HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/home/gplsi/$(whoami)/$PROJECT_NAME}"

# --- Output Configuration (derived from config or defaults) ---
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_SUBDIR="${LOG_SUBDIR:-logs}"

# --- Cache Configuration ---
CACHE_DIR_NAME="${CACHE_DIR_NAME:-.cache}"

# --- WandB Configuration (optional) ---
WANDB_PROJECT="${WANDB_PROJECT:-lmtk-experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-}"

# --- Python Environment ---
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
MAIN_SCRIPT="${MAIN_SCRIPT:-src/main.py}"

# ===================================================================
# SLURM JOB EXECUTION
# ===================================================================

echo "===== LMTK Job Started ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "User: $(whoami)"
echo "============================="

# --- Validate Required Parameters ---
echo "===== Parameter Validation ====="
if [[ -z "$CONFIG_FILE" ]]; then
    echo "❌ ERROR: CONFIG_FILE is required but not provided"
    echo "This should be passed via submit_job.sh with -c flag"
    echo "Example: ./submit_job.sh -c config/experiments/test_continual.yaml"
    exit 1
fi
echo "✅ Configuration file: $CONFIG_FILE"
echo "==============================="

# --- SLURM Environment Information ---
echo "===== SLURM Environment ====="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unknown}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-unknown}"
echo "SLURM_NNODES: ${SLURM_NNODES:-unknown}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unknown}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-unknown}"
echo "SLURM_GPUS: ${SLURM_GPUS:-unknown}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "=============================="

# --- GPU Information ---
echo "===== GPU Information ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "============================"

# --- User and Group Setup ---
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

echo "===== User Information ====="
echo "Username: $USERNAME"
echo "User ID: $USER_ID"
echo "Group ID: $GROUP_ID"
echo "============================="

# --- Build Derived Paths ---
LOG_DIR="$HOST_PROJECT_ROOT/output/$OUTPUT_DIR_NAME/$LOG_SUBDIR"
CACHE_DIR="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME"
CONFIG_PATH="$HOST_PROJECT_ROOT/$CONFIG_FILE"

# --- Create Required Directories ---
echo "===== Setting up directories ====="
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR/datasets"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/transformers"
echo "Directories created successfully"
echo "==============================="

# --- Configuration Summary ---
echo "===== Job Configuration ====="
echo "Project Root: $HOST_PROJECT_ROOT"
echo "Config File: $CONFIG_FILE"
echo "Config Path: $CONFIG_PATH"
echo "Output Directory: $LOG_DIR"
if [[ -n "$NODELIST" ]]; then
echo "Nodelist: $NODELIST"
fi
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Entity: $WANDB_ENTITY"
echo "WandB API Key: ${WANDB_API_KEY:+[PROVIDED]}${WANDB_API_KEY:-[NOT PROVIDED]}"
echo "Hugging Face API Key: ${HUGGINGFACE_API_KEY:+[PROVIDED]}${HUGGINGFACE_API_KEY:-[NOT PROVIDED]}"
echo "=============================="

# --- Data Validation ---
echo "===== Validating paths ====="

if [ ! -d "$HOST_PROJECT_ROOT" ]; then
    echo "❌ ERROR: Project root directory not found: $HOST_PROJECT_ROOT"
    exit 1
fi
echo "✅ Project root: $HOST_PROJECT_ROOT"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi
echo "✅ Configuration file: $CONFIG_PATH"

echo "✅ All required paths validated successfully"
echo "================================="

# --- WandB Authentication Validation ---
echo "===== WandB Configuration ====="
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not provided"
    echo "   - Experiment tracking will be disabled"
    echo "   - To enable WandB logging, pass the API key via:"
    echo "     ./submit_job.sh -c $CONFIG_FILE -k your_key_here"
    echo "   - Or set environment variable: export WANDB_API_KEY=your_key"
    WANDB_MODE="disabled"
else
    echo "✅ WandB API key provided - experiment tracking enabled"
    WANDB_MODE="online"
fi
echo "==============================="

# --- Hugging Face Authentication Validation ---
echo "===== Hugging Face Configuration ====="
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "⚠️  WARNING: Hugging Face API key not provided"
    echo "   - Access to gated Hugging Face repositories will fail"
    echo "   - Provide a key via: ./submit_job.sh -c $CONFIG_FILE --hf-key your_key_here"
    echo "   - Or set environment variable: export HUGGINGFACEHUB_API_TOKEN=your_key"
else
    echo "✅ Hugging Face API key provided"
fi
echo "==============================="

# --- Load Required Modules (if needed) ---
# Uncomment and customize for your cluster
# echo "===== Loading modules ====="
# module load cuda/12.1
# module load singularity
# echo "Modules loaded successfully"
# echo "============================"

ENV_SCRIPT="$HOST_PROJECT_ROOT/scripts/set_environment.sh"

echo "===== Environment Setup ====="
if [ ! -f "$ENV_SCRIPT" ]; then
    echo "❌ ERROR: Environment script not found: $ENV_SCRIPT"
    exit 1
fi

echo "Sourcing environment script: $ENV_SCRIPT"
source "$ENV_SCRIPT"

echo "Python executable after environment setup: $(which python || true)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "================================"

# Ensure cache directories mirror container defaults
export HF_DATASETS_CACHE="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME/datasets"
export HF_HOME="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME/huggingface"
export TRANSFORMERS_CACHE="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME/transformers"
export PYTORCH_TRANSFORMERS_CACHE="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME/transformers"
export WANDB_CACHE_DIR="$HOST_PROJECT_ROOT/tmp/wandb_cache"
export WANDB_CONFIG_DIR="$HOST_PROJECT_ROOT/tmp/wandb_config"

mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Normalise HF tokens for downstream libraries
if [ -n "$HUGGINGFACE_API_KEY" ]; then
    export HUGGINGFACEHUB_API_TOKEN="$HUGGINGFACE_API_KEY"
    export HF_TOKEN="$HUGGINGFACE_API_KEY"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="disabled"
fi

TRAIN_CMD=("$PYTHON_COMMAND" "$MAIN_SCRIPT" "--config" "$CONFIG_PATH")

echo "===== Launching Training ====="
echo "Command: ${TRAIN_CMD[*]}"
echo "Running with srun to leverage SLURM allocation"
echo "================================"

cd ..
SRUN_TOTAL_TASKS="${TOTAL_TASKS:-${SLURM_NTASKS:-}}"
SRUN_ARGS=()
if [[ -n "$SRUN_TOTAL_TASKS" ]]; then
    SRUN_ARGS+=(--ntasks="$SRUN_TOTAL_TASKS")
fi
if [[ -n "${NTASKS_PER_NODE:-}" ]]; then
    SRUN_ARGS+=(--ntasks-per-node="$NTASKS_PER_NODE")
fi
srun "${SRUN_ARGS[@]}" "${TRAIN_CMD[@]}"
# Capture exit code
EXIT_CODE=$?

echo "===== Job Completion ====="
echo "Container finished with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Job failed with exit code: $EXIT_CODE"
    echo "===== Debugging Information ====="
    echo "SLURM job details:"
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null || echo "SLURM job details not available"
    echo "Active modules:"
    module list 2>/dev/null || echo "Module system not in use"
    echo "================================"
else
    echo "✅ Job completed successfully"
fi

echo "=========================="
exit $EXIT_CODE
