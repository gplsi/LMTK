#!/bin/bash
#SBATCH --job-name=lmtk-salamandra-continual
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=%j_salamandra_continual.out        # Name of the output file (using job ID)
#SBATCH --error=%j_salamandra_continual.err         # Name of the error file (using job ID)
#SBATCH --time=48:00:00

# LMTK Salamandra Continual Training Experiment
# 
# Usage with WandB API Key:
# sbatch --export=WANDB_API_KEY=your_wandb_api_key_here p1-dgx.slurm
#
# To skip Docker build (if image already exists):
# sbatch --export=WANDB_API_KEY=your_key,SKIP_DOCKER_BUILD=true p1-dgx.slurm
#
# Or set the WANDB_API_KEY variable directly in this script (less secure)

# Print job information
echo "Iniciando trabajo en `hostname` a las `date` $SLURM_IDX"
nvidia-smi


# Get current user and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

# --- Configuration ---
HOST_PROJECT_ROOT="/home/gplsi/$USERNAME/LMTK" # Actual path on the Slurm node
CONTAINER_PROJECT_ROOT="/workspace"   # Desired path inside the container
DOCKER_IMAGE_NAME="lmtk:latest"
# Dockerfile path relative to the HOST_PROJECT_ROOT
DOCKERFILE_PATH="$HOST_PROJECT_ROOT/docker/Dockerfile"

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

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Data validation passed - dataset and checkpoint found."

# --- Print Slurm and environment info ---
echo "===== Slurm Job Information ====="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES (original Slurm): $CUDA_VISIBLE_DEVICES" 
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "==============================="

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.3 # Example
# module load docker # Example, if docker is a module

# Activate your environment if needed (e.g., for Docker client if not in PATH)
# source /path/to/some_env/bin/activate

# --- Docker Image Management ---
echo "Checking for Docker image: $DOCKER_IMAGE_NAME..."


echo "Building Docker image with user mapping: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USERNAME=$USERNAME"

if docker image inspect "$DOCKER_IMAGE_NAME" &> /dev/null; then
  echo "Using existing Docker image '$DOCKER_IMAGE_NAME'."
else
  echo "Docker image '$DOCKER_IMAGE_NAME' not found. Building..."
  docker build \
      --build-arg USER_ID=$USER_ID \
      --build-arg GROUP_ID=$GROUP_ID \
      --build-arg USERNAME=$USERNAME \
      --network=host \
      -t "$DOCKER_IMAGE_NAME" \
      -f "$DOCKERFILE_PATH" "$HOST_PROJECT_ROOT" # Context is HOST_PROJECT_ROOT
    echo "Docker image '$DOCKER_IMAGE_NAME' built successfully."
fi

# --- WandB Authentication ---
echo "Setting up WandB authentication..."
# Set your WandB API key here (you can also pass it as an environment variable when submitting the job)
# Option 1: Set directly in script (less secure)
WANDB_API_KEY="dc0af4bc96e41099c70e942ed29acfd9220392f5"

# Option 2: Use environment variable (more secure)
# Export WANDB_API_KEY when submitting: sbatch --export=WANDB_API_KEY=your_key p1-dgx.slurm
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging may fail."
    echo "To set it, use: sbatch --export=WANDB_API_KEY=your_key p1-dgx.slurm"
else
    echo "WandB API key found and will be passed to container."
fi

# --- Run Experiment in Docker Container ---
echo "Running LMTK Salamandra continual training experiment in Docker container: $DOCKER_IMAGE_NAME..."
echo "Container will be named: ${SLURM_JOB_ID}_lmtk_salamandra_continual"
echo "Running with user: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"
echo "Container logs will be visible in real-time below..."
echo "=========================================="

# The CUDA_VISIBLE_DEVICES for inside the container will be determined by --gpus all
# and what Slurm has allocated. Docker handles the mapping.
docker run \
  --name "${SLURM_JOB_ID}_lmtk_salamandra_continual" \
  --gpus all \
  --rm \
  --network host \
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
  bash -c "
    set -e  # Exit on any error
    set -x  # Print commands as they are executed
    
    echo '=========================================='
    echo 'Container started successfully'
    echo 'Current working directory: \$(pwd)'
    echo 'Python version: \$(python3 --version)'
    echo 'Available files in workspace:'
    ls -la /workspace/
    echo '=========================================='
    
    if [ -n '$WANDB_API_KEY' ]; then
      echo 'Logging into WandB...'
      wandb login $WANDB_API_KEY || echo 'WandB login failed, continuing without WandB logging'
    else
      echo 'No WandB API key provided, skipping WandB login'
    fi
    
    echo 'Checking config file exists:'
    ls -la config/experiments/salamandra-2b-from-epoch-1/continual.yaml
    
    echo 'Starting training...'
    echo 'Command: python src/main.py --config config/experiments/salamandra-2b-from-epoch-1/continual.yaml'
    python src/main.py --config config/experiments/salamandra-2b-from-epoch-1/continual.yaml
  "

EXIT_CODE=$?
echo "=========================================="
echo "Docker container finished with exit code: $EXIT_CODE"

# If the container failed, try to get more information
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Container failed with exit code $EXIT_CODE"
    echo "Checking if container still exists for debugging..."
    docker ps -a | grep "${SLURM_JOB_ID}_lmtk_salamandra_continual" || echo "Container not found in docker ps -a"
    
    echo "Recent Docker events (last 20):"
    docker events --since="1h" --until="$(date --iso-8601=seconds)" | tail -20 || echo "No recent Docker events"
    
    echo "Docker system info:"
    docker system df || echo "Could not get Docker system info"
fi

echo "Job finished at: $(date)"
echo "=========================================="
exit $EXIT_CODE
