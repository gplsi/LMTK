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

# --- Project Paths (auto-detected) ---
PROJECT_NAME="${PROJECT_NAME:-LMTK}"
HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/home/gplsi/$(whoami)/$PROJECT_NAME}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace}"

# --- Docker Configuration ---
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-lmtk:latest}"
DOCKERFILE_RELATIVE_PATH="${DOCKERFILE_RELATIVE_PATH:-docker/Dockerfile}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

# --- Output Configuration (derived from config or defaults) ---
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_SUBDIR="${LOG_SUBDIR:-logs}"

# --- Cache Configuration ---
CACHE_DIR_NAME="${CACHE_DIR_NAME:-.cache}"

# --- WandB Configuration (optional) ---
WANDB_PROJECT="${WANDB_PROJECT:-lmtk-experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

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
DOCKERFILE_PATH="$HOST_PROJECT_ROOT/$DOCKERFILE_RELATIVE_PATH"
LOG_DIR="$HOST_PROJECT_ROOT/output/$OUTPUT_DIR_NAME/$LOG_SUBDIR"
CACHE_DIR="$HOST_PROJECT_ROOT/$CACHE_DIR_NAME"
CONFIG_PATH="$HOST_PROJECT_ROOT/$CONFIG_FILE"

echo $CACHE_DIR
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
echo "Container Root: $CONTAINER_PROJECT_ROOT"
echo "Docker Image: $DOCKER_IMAGE_NAME"
echo "Config File: $CONFIG_FILE"
echo "Config Path: $CONFIG_PATH"
echo "Output Directory: $LOG_DIR"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Entity: $WANDB_ENTITY"
echo "WandB API Key: ${WANDB_API_KEY:+[PROVIDED]}${WANDB_API_KEY:-[NOT PROVIDED]}"
echo "=============================="

# --- Data Validation ---
echo "===== Validating paths ====="

if [ ! -d "$HOST_PROJECT_ROOT" ]; then
    echo "❌ ERROR: Project root directory not found: $HOST_PROJECT_ROOT"
    exit 1
fi
echo "✅ Project root: $HOST_PROJECT_ROOT"

if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "❌ ERROR: Dockerfile not found: $DOCKERFILE_PATH"
    exit 1
fi
echo "✅ Dockerfile: $DOCKERFILE_PATH"

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

# --- Load Required Modules (if needed) ---
# Uncomment and customize for your cluster
# echo "===== Loading modules ====="
# module load cuda/12.1
# module load singularity
# echo "Modules loaded successfully"
# echo "============================"

# --- Docker Image Management ---
echo "===== Docker Image Management ====="
echo "Checking for Docker image: $DOCKER_IMAGE_NAME"

# Check if force rebuild is requested
if [[ "$FORCE_REBUILD" == "true" ]]; then
    echo "🔨 Force rebuild requested. Rebuilding Docker image: $DOCKER_IMAGE_NAME"
    echo "Build arguments: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USERNAME=$USERNAME"
    
    # Remove existing image first
    docker rmi "$DOCKER_IMAGE_NAME" 2>/dev/null || true
    
    docker build \
        --build-arg USER_ID="$USER_ID" \
        --build-arg GROUP_ID="$GROUP_ID" \
        --build-arg USERNAME="$USERNAME" \
        --network=host \
        -t "$DOCKER_IMAGE_NAME" \
        -f "$DOCKERFILE_PATH" \
        "$HOST_PROJECT_ROOT"
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully: $DOCKER_IMAGE_NAME"
    else
        echo "❌ Docker build failed"
        exit 1
    fi
# More robust image existence check - actually try to run a simple command
elif docker run --rm --entrypoint /bin/echo "$DOCKER_IMAGE_NAME" "Image test successful" &> /dev/null; then
    echo "✅ Using existing Docker image: $DOCKER_IMAGE_NAME"
else
    echo "🔨 Docker image not found or corrupted. Building: $DOCKER_IMAGE_NAME"
    echo "Build arguments: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USERNAME=$USERNAME"
    
    # Remove any potentially corrupted image first
    docker rmi "$DOCKER_IMAGE_NAME" 2>/dev/null || true
    
    docker build \
        --build-arg USER_ID="$USER_ID" \
        --build-arg GROUP_ID="$GROUP_ID" \
        --build-arg USERNAME="$USERNAME" \
        --network=host \
        -t "$DOCKER_IMAGE_NAME" \
        -f "$DOCKERFILE_PATH" \
        "$HOST_PROJECT_ROOT"
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully: $DOCKER_IMAGE_NAME"
    else
        echo "❌ Docker build failed"
        exit 1
    fi
fi
echo "================================="

# --- Container Execution ---
echo "===== Starting Training Container ====="
CONTAINER_NAME="${SLURM_JOB_ID}_lmtk"
echo "Container name: $CONTAINER_NAME"
echo "Running as user: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"
echo "======================================="

# Trap to clean up container on script exit
cleanup() {
    echo "===== Cleanup ====="
    echo "Stopping and removing container if it exists..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "Cleanup completed"
    echo "=================="
}
trap cleanup EXIT

# Run the training container
echo "===== Starting Docker Container ====="
echo "Container name: $CONTAINER_NAME"
echo "Docker image: $DOCKER_IMAGE_NAME"
echo "Host project root: $HOST_PROJECT_ROOT"
echo "Container project root: $CONTAINER_PROJECT_ROOT"
echo "Docker version: $(docker --version)"
echo "Available Docker images:"
docker images | grep -E "(REPOSITORY|lmtk)" || echo "No lmtk images found"
echo "======================================="

docker run \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --rm \
    --network host \
    --user "$USER_ID:$GROUP_ID" \
    --volume "$HOST_PROJECT_ROOT:$CONTAINER_PROJECT_ROOT" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    --env "PYTHONPATH=$CONTAINER_PROJECT_ROOT/src:$CONTAINER_PROJECT_ROOT:${PYTHONPATH:-}" \
    --env "HF_DATASETS_CACHE=$CONTAINER_PROJECT_ROOT/$CACHE_DIR_NAME/datasets" \
    --env "HF_HOME=$CONTAINER_PROJECT_ROOT/$CACHE_DIR_NAME/huggingface" \
    --env "TRANSFORMERS_CACHE=$CONTAINER_PROJECT_ROOT/$CACHE_DIR_NAME/transformers" \
    --env "WANDB_PROJECT=$WANDB_PROJECT" \
    --env "WANDB_ENTITY=$WANDB_ENTITY" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "WANDB_MODE=$WANDB_MODE" \
    --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
    --env "SLURM_NODELIST=$SLURM_NODELIST" \
    --env "USER=$USERNAME" \
    --env "HOME=/home/$USERNAME" \
    --env "LOGNAME=$USERNAME" \
    --env "PYTHON_COMMAND=$PYTHON_COMMAND" \
    --env "MAIN_SCRIPT=$MAIN_SCRIPT" \
    --env "CONFIG_FILE=$CONFIG_FILE" \
    --workdir "$CONTAINER_PROJECT_ROOT" \
    "$DOCKER_IMAGE_NAME" \
    /workspace/slurm/run_container.sh

# Capture exit code
EXIT_CODE=$?

echo "===== Job Completion ====="
echo "Container finished with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Job failed with exit code: $EXIT_CODE"
    echo "===== Debugging Information ====="
    echo "Docker images available:"
    docker images | head -10
    
    echo "Recent Docker events:"
    docker events --since="1h" --until="$(date --iso-8601=seconds)" | tail -10 || echo "No Docker events available"
    
    echo "Docker system info:"
    docker system df || echo "Docker system info not available"
    
    echo "Testing Docker image again:"
    docker run --rm --entrypoint /bin/echo "$DOCKER_IMAGE_NAME" "Test after failure" || echo "Image test failed"
    
    echo "SLURM job details:"
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null || echo "SLURM job details not available"
    echo "================================"
else
    echo "✅ Job completed successfully"
fi

echo "=========================="
exit $EXIT_CODE
