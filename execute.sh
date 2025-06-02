#!/bin/bash
# --- Configuration ---
HOST_PROJECT_ROOT="/home/gplsi/estevanell/LMTK" # Actual path on the Slurm node
CONTAINER_PROJECT_ROOT="/workspace"   # Desired path inside the container
DOCKER_IMAGE_NAME="lmtk:latest"
# Dockerfile path relative to the HOST_PROJECT_ROOT
DOCKERFILE_PATH="$HOST_PROJECT_ROOT/docker/Dockerfile"

# Get current user and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

echo "Building Docker image with user mapping: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USERNAME=$USERNAME"

# # Build Docker image with user arguments
# docker build \
#   --build-arg USER_ID=$USER_ID \
#   --build-arg GROUP_ID=$GROUP_ID \
#   --build-arg USERNAME=$USERNAME \
#   -f "$DOCKERFILE_PATH" \
#   -t "$DOCKER_IMAGE_NAME" \
#   "$HOST_PROJECT_ROOT"

# if [ $? -ne 0 ]; then
#     echo "Docker build failed!"
#     exit 1
# fi

# --- Output Directories ---
# Logs will be stored relative to the HOST_PROJECT_ROOT on the Slurm node
LOG_DIR="$HOST_PROJECT_ROOT/output/salamandra_epochs_3-4/logs"
mkdir -p "$LOG_DIR" # Ensure log directory exists

# --- Cache Directories ---
# Create cache directories for HuggingFace and other dependencies
CACHE_DIR="$HOST_PROJECT_ROOT/.cache"
mkdir -p "$CACHE_DIR/datasets"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/transformers"

# --- Data Validation ---
echo "Validating required data directories..."
DATASET_PATH="$HOST_PROJECT_ROOT/data/tokenized/anonymized-va-salamandra"
CHECKPOINT_PATH="$HOST_PROJECT_ROOT/output/salamandra-2b-2epochs/iter-172684-ckpt.pth"
echo "CUDA_VISIBLE_DEVICES (original Slurm): $CUDA_VISIBLE_DEVICES" 

echo "Running Docker container with user: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"

docker run \
  --name "14760_lmtk_salamandra_continual" \
  --gpus all \
  --rm \
  --network host \
  -it \
  --user "$USER_ID:$GROUP_ID" \
  -v "$HOST_PROJECT_ROOT:$CONTAINER_PROJECT_ROOT" \
  -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  -e "PYTHONPATH=/workspace/src:/workspace:$PYTHONPATH" \
  -e "HF_DATASETS_CACHE=/workspace/.cache/datasets" \
  -e "HF_HOME=/workspace/.cache/huggingface" \
  -e "TRANSFORMERS_CACHE=/workspace/.cache/transformers" \
  -e "WANDB_PROJECT=salamandra-2b_fsdp__allen" \
  -e "WANDB_ENTITY=gplsi_continual" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  -w "$CONTAINER_PROJECT_ROOT" \
  "$DOCKER_IMAGE_NAME" \
  bash