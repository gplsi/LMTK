#!/bin/bash

# ===================================================================
# SLURM Script Validation Tool
# ===================================================================
# This script validates the SLURM configuration without submitting a job
# It checks paths, variables, and configuration consistency
# ===================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/p.slurm"
SLURM_CONFIG_ENV="$SCRIPT_DIR/slurm_config.env"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_job.sh"

echo "===== SLURM Script Validation ====="
echo "Script Directory: $SCRIPT_DIR"
echo "SLURM Script: $SLURM_SCRIPT"
echo "Config File: $SLURM_CONFIG_ENV"
echo "Submit Script: $SUBMIT_SCRIPT"
echo "==============================="

# Function to extract default values from SLURM script
extract_defaults() {
    echo "=== Default Configuration Values ==="
    grep -E "^[A-Z_]+=\"\$\{[A-Z_]+:-" "$SLURM_SCRIPT" | while read -r line; do
        var_name=$(echo "$line" | cut -d'=' -f1)
        default_value=$(echo "$line" | sed 's/.*:-\([^}]*\)}.*/\1/')
        echo "$var_name: $default_value"
    done
    echo
}

# Function to validate file existence
validate_files() {
    echo "=== File Validation ==="
    
    # Load configuration similar to submit_job.sh
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    ENV_SCRIPT="$PROJECT_ROOT/scripts/set_environment.sh"
    
    # Source slurm_config.env if it exists to get defaults
    if [[ -f "$SLURM_CONFIG_ENV" ]]; then
        echo "Loading configuration from: $SLURM_CONFIG_ENV"
        source "$SLURM_CONFIG_ENV"
    else
        echo "⚠️  No configuration file found at: $SLURM_CONFIG_ENV"
    fi
    
    echo "Project Root: $PROJECT_ROOT"
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "❌ Project root directory does not exist: $PROJECT_ROOT"
    else
        echo "✅ Project root exists: $PROJECT_ROOT"
    fi
    
    echo "SLURM Script: $SLURM_SCRIPT"
    if [ ! -f "$SLURM_SCRIPT" ]; then
        echo "❌ SLURM script not found: $SLURM_SCRIPT"
    else
        echo "✅ SLURM script exists"
    fi
    
    echo "Submit Script: $SUBMIT_SCRIPT"
    if [ ! -f "$SUBMIT_SCRIPT" ]; then
        echo "❌ Submit script not found: $SUBMIT_SCRIPT"
    else
        echo "✅ Submit script exists"
    fi

    echo "Environment script: $ENV_SCRIPT"
    if [ ! -f "$ENV_SCRIPT" ]; then
        echo "❌ Environment script not found: $ENV_SCRIPT"
        echo "   This is required for Conda-based SLURM runs."
    else
        echo "✅ Environment script exists"
    fi
    
    echo "Main script: $PROJECT_ROOT/$MAIN_SCRIPT"
    if [ ! -f "$PROJECT_ROOT/$MAIN_SCRIPT" ]; then
        echo "❌ Main script not found: $PROJECT_ROOT/$MAIN_SCRIPT"
    else
        echo "✅ Main script exists"
    fi
    echo
}

# Function to check Conda availability (informational; compute nodes may differ)
check_conda() {
    echo "=== Conda Validation ==="
    echo "ℹ  Conda validation note:"
    echo "   This script runs on the login node; compute node environments may differ."
    echo ""

    if command -v conda &> /dev/null; then
        echo "✅ conda command found: $(conda --version 2>/dev/null || echo 'version unknown')"
    else
        echo "⚠️  conda command not found in PATH on this node"
        echo "   p.slurm sources scripts/set_environment.sh to activate Conda on the compute node."
    fi
    echo
}

# Function to validate SLURM syntax
validate_slurm_syntax() {
    echo "=== SLURM Syntax Validation ==="
    
    # Check for SLURM directives
    sbatch_lines=$(grep -c "^#SBATCH" "$SLURM_SCRIPT" || echo "0")
    echo "SLURM directives found: $sbatch_lines"
    
    if [ "$sbatch_lines" -eq 0 ]; then
        echo "❌ No SLURM directives found in script"
        return 1
    fi
    
    echo "SLURM directives:"
    grep "^#SBATCH" "$SLURM_SCRIPT"
    
    # Check if sbatch command is available
    if command -v sbatch &> /dev/null; then
        echo "✅ SLURM sbatch command available"
        
        # Check SLURM version
        echo "SLURM version: $(sbatch --version 2>/dev/null || echo 'Unknown')"
        
        # Test syntax without submitting (only if we have SLURM access)
        echo "Testing SLURM script syntax..."
        if sbatch --test-only "$SLURM_SCRIPT" 2>/dev/null; then
            echo "✅ SLURM script syntax is valid"
        else
            echo "⚠️  SLURM script syntax test failed or requires cluster access"
            echo "   This may be normal if running from a login node without job submission access"
            echo "   Try manually: sbatch --test-only $SLURM_SCRIPT"
        fi
    else
        echo "⚠️  SLURM sbatch command not available"
        echo "   Make sure you're on a SLURM cluster and SLURM tools are loaded"
    fi
    
    # Basic shell syntax check
    echo "Checking shell syntax..."
    if bash -n "$SLURM_SCRIPT"; then
        echo "✅ Shell syntax is valid"
    else
        echo "❌ Shell syntax errors found"
    fi
    echo
}

# Function to show environment variables
show_environment() {
    echo "=== Environment Variables ==="
    echo "Current user: $(whoami)"
    echo "User ID: $(id -u)"
    echo "Group ID: $(id -g)"
    echo "HOME: $HOME"
    echo "PWD: $PWD"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    echo "WANDB_API_KEY: ${WANDB_API_KEY:0:10}${WANDB_API_KEY:+...}"
    echo
}

# Function to simulate the script execution
simulate_execution() {
    echo "=== Execution Simulation ==="
    echo "This is what would happen when the script runs:"
    echo

    # Note: do not source p.slurm (it is an executable job script and may exit).
    PROJECT_NAME="${PROJECT_NAME:-LMTK}"
    HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/home/gplsi/$(whoami)/$PROJECT_NAME}"
    OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-experiment_$(date +%Y%m%d_%H%M%S)}"
    LOG_SUBDIR="${LOG_SUBDIR:-logs}"
    CACHE_DIR_NAME="${CACHE_DIR_NAME:-.cache}"
    WANDB_PROJECT="${WANDB_PROJECT:-lmtk-experiments}"
    WANDB_ENTITY="${WANDB_ENTITY:-}"
    PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
    MAIN_SCRIPT="${MAIN_SCRIPT:-src/main.py}"

    # submit_job.sh passes the experiment config via CONFIG_FILE for p.slurm.
    EXPERIMENT_CONFIG="${CONFIG_FILE:-<provided by submit_job.sh via --export CONFIG_FILE=...>}"
    
    echo "1. Job Configuration:"
    echo "   - Job Name: $JOB_NAME"
    echo "   - Partition: $PARTITION"
    echo "   - GPUs: $GPU_COUNT"
    echo "   - Memory: $MEMORY"
    echo "   - Time Limit: $TIME_LIMIT"
    echo
    
    echo "2. Directory Setup:"
    echo "   - Create log directory: $HOST_PROJECT_ROOT/output/$OUTPUT_DIR_NAME/$LOG_SUBDIR"
    echo "   - Create cache directories under: $HOST_PROJECT_ROOT/$CACHE_DIR_NAME"
    echo

    echo "3. Environment Setup:"
    echo "   - Source: $HOST_PROJECT_ROOT/scripts/set_environment.sh"
    echo "   - Activate Conda env: lmtk"
    echo

    echo "4. Training Execution:"
    echo "   - Command: $PYTHON_COMMAND $MAIN_SCRIPT --config $EXPERIMENT_CONFIG"
    echo "   - WandB Project: $WANDB_PROJECT"
    echo "   - WandB Entity: $WANDB_ENTITY"
    echo
}

# Main execution
main() {
    extract_defaults
    validate_files
    check_conda
    validate_slurm_syntax
    show_environment
    simulate_execution
    
    echo "===== Validation Complete ====="
    echo "Review the output above for any issues."
    echo "If everything looks good, you can submit with:"
    echo "  sbatch $SLURM_SCRIPT"
    echo "Or use the helper:"
    echo "  ./submit_job.sh"
}

# Run validation
main "$@"
