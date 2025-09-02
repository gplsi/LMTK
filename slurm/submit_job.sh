#!/bin/bash

# ===================================================================
# LMTK - SLURM Job Submission Helper Script
# ===================================================================
# This script provides an easy way to submit SLURM jobs for the LMTK
# framework. Only configuration path is required, all other parameters
# are optional with sensible defaults.
# ===================================================================

set -e

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/p.slurm"

# Initialize variables (will be set by config file, CLI args, or defaults in that order)
CONFIG_FILE="config/experiments/turismo/training_ddp.yaml"
WANDB_API_KEY="0002c6b12094cc0806f5d415088555cc552ca961"
GPU_COUNT="4"
MEMORY="32G"
TIME_LIMIT="24:00:00"
PARTITION="postiguet1"
JOB_NAME="turismo_training_ddp"
CPUS_PER_TASK="1"
NODES="1"
NTASKS_PER_NODE="1"
OUTPUT_DIR="turismo_training_ddp"
OUTPUT_FILE_PATTERN="%j_turismo_training_ddp.out"
ERROR_FILE_PATTERN="%j_turismo_training_ddp.err"
DRY_RUN=false
FORCE_REBUILD=false

# Load configuration file defaults if it exists
SLURM_CONFIG_FILE="$SCRIPT_DIR/slurm_config.env"
CONFIG_LOADED=false
if [[ -f "$SLURM_CONFIG_FILE" ]]; then
    # Source the configuration file to load defaults
    source "$SLURM_CONFIG_FILE"
    CONFIG_LOADED=true
fi

# Set final defaults for any unset variables
GPU_COUNT="${GPU_COUNT:-2}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
PARTITION="${PARTITION:-dgx}"
JOB_NAME="${JOB_NAME:-lmtk}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
OUTPUT_FILE_PATTERN="${OUTPUT_FILE_PATTERN:-%j_lmtk.out}"
ERROR_FILE_PATTERN="${ERROR_FILE_PATTERN:-%j_lmtk.err}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -c CONFIG_FILE [OPTIONS]

Submit SLURM jobs for the LMTK framework. Only configuration file is required.

CONFIGURATION PRECEDENCE (highest to lowest):
    1. Command line arguments (highest priority)
    2. Environment variables
    3. slurm_config.env file defaults
    4. Built-in defaults (lowest priority)

REQUIRED:
    -c, --config CONFIG_FILE     Path to experiment config file (required)

OPTIONAL:
    -k, --wandb-key API_KEY      WandB API key (optional, warning if not provided)
    -g, --gpus GPU_COUNT         Number of GPUs to request (default: 2)
    -m, --memory MEMORY          Memory to request (default: 64G)
    -t, --time TIME_LIMIT        Time limit (default: 48:00:00)
    -p, --partition PARTITION    SLURM partition (default: dgx)
    -j, --job-name JOB_NAME      Job name (default: lmtk)
    --cpus CPUS                  CPUs per task (default: 16)
    --nodes NODES                Number of nodes (default: 1)
    -o, --output OUTPUT_DIR      Output directory name (optional)
    --rebuild                    Force rebuild Docker image even if it exists
    -d, --dry-run                Show the command that would be executed without running it
    -h, --help                   Show this help message

EXAMPLES:
    # Basic usage - only config required
    $0 -c config/experiments/test_continual.yaml

    # With WandB logging
    $0 -c config/experiments/test_continual.yaml -k your_wandb_api_key

    # With custom resources
    $0 -c config/experiments/my_experiment.yaml -g 4 -m 200G -t 72:00:00

    # Quick test run with minimal resources
    $0 -c config/experiments/test_continual.yaml -g 1 -t 2:00:00 -o test_run

    # Different partition
    $0 -c config/experiments/my_experiment.yaml -p gpu -g 8 -m 400G

    # Force rebuild Docker image (useful after code changes)
    $0 -c config/experiments/test_continual.yaml --rebuild

    # Dry run to see what would be executed
    $0 -c config/experiments/my_experiment.yaml -d

NOTES:
    - Configuration file path must be relative to project root
    - WandB API key is optional but recommended for experiment tracking
    - If no WandB key is provided, a warning will be shown but job will continue
    - Use environment variable WANDB_API_KEY as alternative to -k flag
    - Job logs will be saved as {job_id}_lmtk.out and {job_id}_lmtk.err
    - Use --rebuild if you encounter Docker image issues or after code changes

SECURITY:
    - Never commit WandB API keys to version control
    - Use environment variables or pass keys via command line
    - API keys are only stored temporarily during job execution

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -k|--wandb-key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: Configuration file is required."
    echo ""
    echo "Usage: $0 -c CONFIG_FILE [other_options]"
    echo "Run '$0 --help' for more information."
    exit 1
fi

# Handle relative/absolute paths for config file
if [[ "$CONFIG_FILE" = /* ]]; then
    # Absolute path
    FULL_CONFIG_PATH="$CONFIG_FILE"
else
    # Relative path - always resolve from project root
    FULL_CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"
fi

# Ensure OUTPUT_DIR is always under the project root unless absolute
if [[ -n "$OUTPUT_DIR" ]]; then
    if [[ "$OUTPUT_DIR" = /* ]]; then
        FINAL_OUTPUT_DIR="$OUTPUT_DIR"
    else
        FINAL_OUTPUT_DIR="$PROJECT_ROOT/output/$OUTPUT_DIR"
    fi
else
    # Default: auto-generate output directory under project root
    FINAL_OUTPUT_DIR="$PROJECT_ROOT/output/experiment_$(date +%Y%m%d_%H%M%S)"
fi

# Make sure output directory exists
mkdir -p "$FINAL_OUTPUT_DIR"

# Validate config file exists
if [[ ! -f "$FULL_CONFIG_PATH" ]]; then
    echo "ERROR: Configuration file not found: $FULL_CONFIG_PATH"
    exit 1
fi

# Check if WandB key is set via environment variable if not provided via command line
if [[ -z "$WANDB_API_KEY" && -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY}"
fi

# Warn about WandB key but don't fail
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "⚠️  WARNING: No WandB API key provided."
    echo "   - Experiment tracking will be disabled"
    echo "   - To enable WandB, use: $0 -c $CONFIG_FILE -k your_wandb_key"
    echo "   - Or set environment variable: export WANDB_API_KEY=your_key"
    echo ""
fi

# Always pass CONFIG_FILE relative to PROJECT_ROOT for the container
if [[ "$CONFIG_FILE" = /* ]]; then
    # Remove leading PROJECT_ROOT from absolute path to make it relative
    REL_CONFIG_FILE="${CONFIG_FILE#$PROJECT_ROOT/}"
else
    REL_CONFIG_FILE="$CONFIG_FILE"
fi
EXPORT_VARS="CONFIG_FILE=$REL_CONFIG_FILE,HOST_PROJECT_ROOT=$PROJECT_ROOT"

if [[ -n "$WANDB_API_KEY" ]]; then
    EXPORT_VARS="${EXPORT_VARS},WANDB_API_KEY=$WANDB_API_KEY"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    EXPORT_VARS="${EXPORT_VARS},OUTPUT_DIR_NAME=$OUTPUT_DIR"
fi

if [[ "$FORCE_REBUILD" == "true" ]]; then
    EXPORT_VARS="${EXPORT_VARS},FORCE_REBUILD=true"
fi

# Always include job configuration variables for reference
EXPORT_VARS="${EXPORT_VARS},GPU_COUNT=$GPU_COUNT"
EXPORT_VARS="${EXPORT_VARS},MEMORY=$MEMORY"
EXPORT_VARS="${EXPORT_VARS},TIME_LIMIT=$TIME_LIMIT"
EXPORT_VARS="${EXPORT_VARS},PARTITION=$PARTITION"
EXPORT_VARS="${EXPORT_VARS},JOB_NAME=$JOB_NAME"
EXPORT_VARS="${EXPORT_VARS},CPUS_PER_TASK=$CPUS_PER_TASK"
EXPORT_VARS="${EXPORT_VARS},NODES=$NODES"
EXPORT_VARS="${EXPORT_VARS},NTASKS_PER_NODE=$NTASKS_PER_NODE"

# Build the sbatch command with proper SLURM directives
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --job-name=$JOB_NAME"
SBATCH_CMD="$SBATCH_CMD --partition=$PARTITION"
SBATCH_CMD="$SBATCH_CMD --gres=gpu:$GPU_COUNT"
SBATCH_CMD="$SBATCH_CMD --mem=$MEMORY"
SBATCH_CMD="$SBATCH_CMD --time=$TIME_LIMIT"
SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS_PER_TASK"
SBATCH_CMD="$SBATCH_CMD --nodes=$NODES"
SBATCH_CMD="$SBATCH_CMD --ntasks-per-node=$NTASKS_PER_NODE"
SBATCH_CMD="$SBATCH_CMD --output=$OUTPUT_FILE_PATTERN"
SBATCH_CMD="$SBATCH_CMD --error=$ERROR_FILE_PATTERN"

# Add environment variables
if [[ -n "$EXPORT_VARS" ]]; then
    SBATCH_CMD="$SBATCH_CMD --export=$EXPORT_VARS"
fi

# Add the script
SBATCH_CMD="$SBATCH_CMD $SLURM_SCRIPT"

# Show configuration summary
echo "===== LMTK Job Submission Summary ====="
echo "Project Root: $PROJECT_ROOT"
echo "SLURM Script: $SLURM_SCRIPT"
if [[ -f "$SLURM_CONFIG_FILE" ]]; then
    echo "Configuration loaded from: $SLURM_CONFIG_FILE"
fi
echo "Config File: $CONFIG_FILE"
echo "Full Config Path: $FULL_CONFIG_PATH"
echo "Job Name: $JOB_NAME"
echo "Partition: $PARTITION"
echo "GPU Count: $GPU_COUNT"
echo "Memory: $MEMORY"
echo "Time Limit: $TIME_LIMIT"
echo "CPUs per Task: $CPUS_PER_TASK"
echo "Nodes: $NODES"
if [[ -n "$WANDB_API_KEY" ]]; then
    echo "WandB API Key: ${WANDB_API_KEY:0:10}..."
else
    echo "WandB API Key: [NOT PROVIDED]"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output Directory: $OUTPUT_DIR"
fi
echo "======================================="
# Execute or show the command
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN - Command that would be executed:"
    echo "$SBATCH_CMD"
    echo ""
    echo "To actually submit the job, remove the -d/--dry-run flag"
else
    echo "Submitting job..."
    echo "Running: $SBATCH_CMD"
    echo ""
    eval "$SBATCH_CMD"
    
    if [ $? -eq 0 ]; then
        echo "✅ Job submitted successfully!"
        echo "Monitor with: squeue -u \$(whoami)"
        echo "Cancel with: scancel <job_id>"
    else
        echo "❌ Job submission failed!"
        exit 1
    fi
fi
