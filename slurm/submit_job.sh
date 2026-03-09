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
CONFIG_FILE=""
WANDB_API_KEY="${WANDB_API_KEY:-}"
HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-}"
GPU_COUNT=""
MEMORY=""
TIME_LIMIT=""
PARTITION=""
QOS=""
JOB_NAME=""
CPUS_PER_TASK=""
NODES=""
NTASKS_PER_NODE=""
TOTAL_TASKS=""
NODELIST=""
OUTPUT_DIR=""
OUTPUT_FILE_PATTERN="${OUTPUT_FILE_PATTERN:-}"
ERROR_FILE_PATTERN="${ERROR_FILE_PATTERN:-}"
DRY_RUN=false
CONFIG_LOGGING_MODE=""
CONFIG_WANDB_MODE=""

ENV_OUTPUT_FILE_PATTERN="$OUTPUT_FILE_PATTERN"
ENV_ERROR_FILE_PATTERN="$ERROR_FILE_PATTERN"

# Load configuration file defaults if it exists
SLURM_CONFIG_FILE="$SCRIPT_DIR/slurm_config.env"
CONFIG_LOADED=false
if [[ -f "$SLURM_CONFIG_FILE" ]]; then
    # Source the configuration file to load defaults
    source "$SLURM_CONFIG_FILE"
    CONFIG_LOADED=true
fi

if [[ -n "$ENV_OUTPUT_FILE_PATTERN" ]]; then
    OUTPUT_FILE_PATTERN="$ENV_OUTPUT_FILE_PATTERN"
fi
if [[ -n "$ENV_ERROR_FILE_PATTERN" ]]; then
    ERROR_FILE_PATTERN="$ENV_ERROR_FILE_PATTERN"
fi

# Set final defaults for any unset variables
GPU_COUNT="${GPU_COUNT:-2}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
PARTITION="${PARTITION:-dgx}"
QOS="${QOS:-}"
JOB_NAME="${JOB_NAME:-lmtk}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-}"
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
        --hf-key API_KEY         Hugging Face API key (optional, enables gated repo access)
    -g, --gpus GPU_COUNT         Number of GPUs to request (default: 2)
    -m, --memory MEMORY          Memory to request (default: 64G)
    -t, --time TIME_LIMIT        Time limit (default: 48:00:00)
    -p, --partition PARTITION    SLURM partition (default: dgx)
    -q, --qos QOS                SLURM QoS (optional; omitted when unset)
    -j, --job-name JOB_NAME      Job name (default: lmtk)
    --cpus CPUS                  CPUs per task (default: 16)
    --nodes NODES                Number of nodes (default: 1)
    --ntasks-per-node COUNT      Tasks per node (default: 1)
    --nodelist NODELIST          Specific nodes to use (optional, e.g., lovelace.iuii.ua.es)
    -o, --output OUTPUT_DIR      Output directory name (optional)
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

    # Multi-node run (tasks per node must match GPUs per node)
    $0 -c config/experiments/my_experiment.yaml --nodes 2 --ntasks-per-node 4 -g 4

    # Specific node
    $0 -c config/experiments/my_experiment.yaml --nodelist=lovelace.iuii.ua.es

    # Dry run to see what would be executed
    $0 -c config/experiments/my_experiment.yaml -d

NOTES:
    - Configuration file path must be relative to project root
    - WandB API key is optional but recommended for experiment tracking
    - If no WandB key is provided, a warning will be shown but job will continue
    - Use environment variable WANDB_API_KEY as alternative to -k flag
    - Job logs will be saved as {job_id}_lmtk.out and {job_id}_lmtk.err
    - Total tasks is computed as nodes * ntasks-per-node and passed to sbatch
SECURITY:
    - Never commit WandB API keys to version control
    - Use environment variables or pass keys via command line
    - API keys are only stored temporarily during job execution

EOF
}

validate_conda_preflight() {
    local env_script="$PROJECT_ROOT/scripts/set_environment.sh"
    local preflight_output=""

    if [[ ! -f "$env_script" ]]; then
        echo "ERROR: Environment script not found: $env_script"
        echo "This job was not submitted because SLURM would fail during environment setup."
        exit 1
    fi

    if ! preflight_output=$(bash -c "source \"$env_script\"" 2>&1); then
        echo "ERROR: Conda environment initialization failed during SLURM submission preflight."
        echo "This job was not submitted because the compute node would fail during environment setup."
        echo ""
        echo "scripts/set_environment.sh output:"
        echo "$preflight_output"
        exit 1
    fi

    echo "Conda environment preflight passed."
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
        --hf-key)
            HUGGINGFACE_API_KEY="$2"
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
        -q|--qos)
            QOS="$2"
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
        --ntasks-per-node)
            NTASKS_PER_NODE="$2"
            shift 2
            ;;
        --nodelist)
            NODELIST="$2"
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

if ! [[ "$NODES" =~ ^[0-9]+$ ]] || [[ "$NODES" -lt 1 ]]; then
    echo "ERROR: --nodes must be a positive integer; got '$NODES'"
    exit 1
fi

if ! [[ "$NTASKS_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NTASKS_PER_NODE" -lt 1 ]]; then
    echo "ERROR: --ntasks-per-node must be a positive integer; got '$NTASKS_PER_NODE'"
    exit 1
fi

TOTAL_TASKS=$((NODES * NTASKS_PER_NODE))

# Check if WandB key is set via environment variable if not provided via command line
if [[ -z "$WANDB_API_KEY" && -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY}"
fi

# Check for Hugging Face key from common environment variables if not provided via CLI
if [[ -z "$HUGGINGFACE_API_KEY" ]]; then
    if [[ -n "${HUGGINGFACE_API_KEY:-}" ]]; then
        HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY}"
    elif [[ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]]; then
        HUGGINGFACE_API_KEY="${HUGGINGFACEHUB_API_TOKEN}"
    elif [[ -n "${HF_TOKEN:-}" ]]; then
        HUGGINGFACE_API_KEY="${HF_TOKEN}"
    fi
fi

# Warn about WandB key but don't fail (unless offline mode requested)
if [[ -z "$WANDB_API_KEY" ]]; then
    if [[ "$CONFIG_LOGGING_MODE" == "wandb" && "$CONFIG_WANDB_MODE" != "offline" ]]; then
        echo "⚠️  WARNING: No WandB API key provided."
        echo "   - Experiment tracking will be disabled"
        echo "   - To enable WandB, use: $0 -c $CONFIG_FILE -k your_wandb_key"
        echo "   - Or set environment variable: export WANDB_API_KEY=your_key"
        echo ""
    fi
fi

# Default ntasks-per-node to the GPU count if not explicitly provided
if [[ -z "$NTASKS_PER_NODE" ]]; then
    NTASKS_PER_NODE="$GPU_COUNT"
fi

# Default total ntasks to the GPU count if not explicitly provided
if [[ -z "$NTASKS" ]]; then
    NTASKS="$GPU_COUNT"
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

if [[ -n "$HUGGINGFACE_API_KEY" ]]; then
    EXPORT_VARS="${EXPORT_VARS},HUGGINGFACE_API_KEY=$HUGGINGFACE_API_KEY"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    EXPORT_VARS="${EXPORT_VARS},OUTPUT_DIR_NAME=$OUTPUT_DIR"
fi

# Always include job configuration variables for reference
EXPORT_VARS="${EXPORT_VARS},GPU_COUNT=$GPU_COUNT"
EXPORT_VARS="${EXPORT_VARS},MEMORY=$MEMORY"
EXPORT_VARS="${EXPORT_VARS},TIME_LIMIT=$TIME_LIMIT"
EXPORT_VARS="${EXPORT_VARS},PARTITION=$PARTITION"
if [[ -n "$QOS" ]]; then
    EXPORT_VARS="${EXPORT_VARS},QOS=$QOS"
fi
EXPORT_VARS="${EXPORT_VARS},JOB_NAME=$JOB_NAME"
EXPORT_VARS="${EXPORT_VARS},CPUS_PER_TASK=$CPUS_PER_TASK"
EXPORT_VARS="${EXPORT_VARS},NODES=$NODES"
EXPORT_VARS="${EXPORT_VARS},NTASKS_PER_NODE=$NTASKS_PER_NODE"
EXPORT_VARS="${EXPORT_VARS},TOTAL_TASKS=$TOTAL_TASKS"
if [[ -n "${PYTHON_COMMAND:-}" ]]; then
    EXPORT_VARS="${EXPORT_VARS},PYTHON_COMMAND=$PYTHON_COMMAND"
fi
if [[ -n "${MAIN_SCRIPT:-}" ]]; then
    EXPORT_VARS="${EXPORT_VARS},MAIN_SCRIPT=$MAIN_SCRIPT"
fi

if [[ -n "$NODELIST" ]]; then
    EXPORT_VARS="${EXPORT_VARS},NODELIST=$NODELIST"
fi

if [[ "$CONFIG_LOGGING_MODE" == "wandb" && "$CONFIG_WANDB_MODE" == "offline" ]]; then
    EXPORT_VARS="${EXPORT_VARS},WANDB_MODE=offline"
fi

# Build the sbatch command with proper SLURM directives
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --job-name=$JOB_NAME"
SBATCH_CMD="$SBATCH_CMD --partition=$PARTITION"
if [[ -n "$QOS" ]]; then
    SBATCH_CMD="$SBATCH_CMD --qos=$QOS"
fi
SBATCH_CMD="$SBATCH_CMD --gres=gpu:$GPU_COUNT"
SBATCH_CMD="$SBATCH_CMD --mem=$MEMORY"
SBATCH_CMD="$SBATCH_CMD --time=$TIME_LIMIT"
SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS_PER_TASK"
SBATCH_CMD="$SBATCH_CMD --nodes=$NODES"
SBATCH_CMD="$SBATCH_CMD --ntasks=$TOTAL_TASKS"
SBATCH_CMD="$SBATCH_CMD --ntasks-per-node=$NTASKS_PER_NODE"
SBATCH_CMD="$SBATCH_CMD --output=$OUTPUT_FILE_PATTERN"
SBATCH_CMD="$SBATCH_CMD --error=$ERROR_FILE_PATTERN"

# Add nodelist if specified
if [[ -n "$NODELIST" ]]; then
    SBATCH_CMD="$SBATCH_CMD --nodelist=$NODELIST"
fi

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
if [[ -n "$QOS" ]]; then
    echo "QoS: $QOS"
else
    echo "QoS: [NOT SET]"
fi
echo "GPU Count: $GPU_COUNT"
echo "Memory: $MEMORY"
echo "Time Limit: $TIME_LIMIT"
echo "CPUs per Task: $CPUS_PER_TASK"
echo "Tasks per Node: $NTASKS_PER_NODE"
echo "Nodes: $NODES"
echo "Total Tasks: $TOTAL_TASKS"
if [[ -n "$NODELIST" ]]; then
    echo "Nodelist: $NODELIST"
fi
if [[ -n "$WANDB_API_KEY" ]]; then
    echo "WandB API Key: ${WANDB_API_KEY:0:10}..."
else
    echo "WandB API Key: [NOT PROVIDED]"
fi
if [[ -n "$HUGGINGFACE_API_KEY" ]]; then
    echo "Hugging Face API Key: ${HUGGINGFACE_API_KEY:0:10}..."
else
    echo "Hugging Face API Key: [NOT PROVIDED]"
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
    validate_conda_preflight
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
