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
CONFIG_FILE="$SCRIPT_DIR/slurm_config.env"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_job.sh"

echo "===== SLURM Script Validation ====="
echo "Script Directory: $SCRIPT_DIR"
echo "SLURM Script: $SLURM_SCRIPT"
echo "Config File: $CONFIG_FILE"
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
    
    # Source slurm_config.env if it exists to get defaults
    if [[ -f "$CONFIG_FILE" ]]; then
        echo "Loading configuration from: $CONFIG_FILE"
        source "$CONFIG_FILE"
    else
        echo "⚠️  No configuration file found at: $CONFIG_FILE"
    fi
    
    echo "Project Root: $PROJECT_ROOT"
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "❌ Project root directory does not exist: $PROJECT_ROOT"
    else
        echo "✅ Project root exists"
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
    
    echo "Dockerfile: $PROJECT_ROOT/$DOCKERFILE_RELATIVE_PATH"
    if [ ! -f "$PROJECT_ROOT/$DOCKERFILE_RELATIVE_PATH" ]; then
        echo "⚠️  Dockerfile not found: $PROJECT_ROOT/$DOCKERFILE_RELATIVE_PATH"
    else
        echo "✅ Dockerfile exists"
    fi
    
    echo "Main script: $PROJECT_ROOT/$MAIN_SCRIPT"
    if [ ! -f "$PROJECT_ROOT/$MAIN_SCRIPT" ]; then
        echo "❌ Main script not found: $PROJECT_ROOT/$MAIN_SCRIPT"
    else
        echo "✅ Main script exists"
    fi
    echo
}

# Function to check Docker availability
check_docker() {
    echo "=== Docker Validation ==="
    echo "ℹ  Docker validation note:"
    echo "   Docker is typically only available on SLURM compute nodes, not login nodes"
    echo "   This validation script runs on the login node, so Docker may not be accessible here"
    echo ""
    
    if command -v docker &> /dev/null; then
        echo "✅ Docker command is available on this node"
        if docker info &> /dev/null; then
            echo "✅ Docker daemon is accessible"
            echo "Docker version: $(docker --version)"
            echo "ℹ  Note: Docker being available here doesn't guarantee it's available on compute nodes"
        else
            echo "⚠️  Docker daemon not accessible on this node"
            echo "   This is expected on login nodes - Docker should be available on compute nodes"
        fi
    else
        echo "ℹ  Docker command not found on this node (expected on login nodes)"
        echo "   The SLURM script will use Docker on the allocated compute nodes"
    fi
    
    echo "✅ Docker image name configured: ${DOCKER_IMAGE_NAME:-lmtk:latest}"
    echo "✅ Dockerfile path configured: ${DOCKERFILE_RELATIVE_PATH:-docker/Dockerfile}"
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
    
    # Source variables
    source "$SLURM_SCRIPT" 2>/dev/null || {
        echo "ERROR: Could not source variables from SLURM script"
        return 1
    }
    
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
    
    echo "3. Docker Operations:"
    echo "   - Check for image: $DOCKER_IMAGE_NAME"
    echo "   - If not found, build from: $HOST_PROJECT_ROOT/$DOCKERFILE_RELATIVE_PATH"
    echo "   - Container name: \${SLURM_JOB_ID}_lmtk_continual"
    echo
    
    echo "4. Training Execution:"
    echo "   - Mount: $HOST_PROJECT_ROOT -> $CONTAINER_PROJECT_ROOT"
    echo "   - Command: $PYTHON_COMMAND $MAIN_SCRIPT --config $CONFIG_FILE"
    echo "   - WandB Project: $WANDB_PROJECT"
    echo "   - WandB Entity: $WANDB_ENTITY"
    echo
}

# Main execution
main() {
    extract_defaults
    validate_files
    check_docker
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
