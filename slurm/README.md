# LMTK SLURM Production Guide

This directory contains production-ready SLURM scripts for running LMTK framework experiments on SLURM clusters with Docker integration, comprehensive logging, and WandB experiment tracking.

## 🚀 Quick Reference

**Most Common Commands:**
```bash
# Basic job submission
./submit_job.sh -c config/experiments/test_continual.yaml

# With experiment tracking
./submit_job.sh -c config/experiments/test_continual.yaml -k your_wandb_key

# Custom resources
./submit_job.sh -c config.yaml -g 4 -m 128G -t 72:00:00 -k your_key

# Check job status
squeue -u $(whoami)

# Monitor logs
tail -f JOBID_lmtk.out
```

**Quick Parameter Guide:**
- `-c`: Config file (required)
- `-k`: WandB API key (enables tracking)
- `-g`: GPU count (default: 1)
- `-m`: Memory (default: 32G)
- `-t`: Time limit (default: 48:00:00)
- `-j`: Job name (optional)
- `-p`: Partition (default: postiguet1)

---

## 📖 Table of Contents

1. [🚀 Getting Started](#-getting-started)
2. [📖 Understanding the Basics](#-understanding-the-basics)
3. [⚙️ Step-by-Step Usage](#️-step-by-step-usage)
4. [🎛️ Configuration Deep Dive](#️-configuration-deep-dive)
5. [📊 Advanced Features](#-advanced-features)
6. [🔍 Monitoring & Debugging](#-monitoring--debugging)
7. [🔧 Troubleshooting](#-troubleshooting)

---

## 🚀 Getting Started

### Your First Job - The Simplest Case

Start with this single command. It's all you need to run any LMTK experiment:

```bash
./submit_job.sh -c config/experiments/test_continual.yaml
```

**What happens:**
- ✅ Uses default resources (1 GPU, 32GB RAM, 48 hours)
- ✅ Runs on your cluster's default GPU partition
- ✅ Creates a Docker container with proper environment
- ⚠️ **No experiment tracking** (you'll see a warning)

**Expected output:**
```
⚠️  WARNING: No WandB API key provided.
   - Experiment tracking will be disabled
   - To enable WandB, use: ./submit_job.sh -c config/experiments/test_continual.yaml -k your_wandb_key

===== LMTK Job Submission Summary =====
Job Name: lmtk-experiment
Partition: postiguet1
GPU Count: 1
Memory: 32G
Time Limit: 48:00:00
=======================================
Submitting job...
✅ Job submitted successfully!
Monitor with: squeue -u $(whoami)
```

### Adding Experiment Tracking (Recommended)

```bash
./submit_job.sh -c config/experiments/test_continual.yaml -k your_wandb_api_key
```

**What changes:**
- ✅ **Automatic WandB login** inside the container
- ✅ **Full experiment tracking** with logs, metrics, and artifacts
- ✅ **Experiment visibility** in your WandB dashboard

---

## � Understanding the Basics

### What Happens When You Submit a Job?

When you run `./submit_job.sh -c your_config.yaml`, here's the complete workflow:

1. **Configuration Loading** 📋
   - Loads defaults from `slurm_config.env`
   - Applies any environment variables you've set
   - Overrides with command-line arguments

2. **Job Submission** 🚀
   - Creates SLURM job with specified resources
   - Sets up environment variables for the container
   - Submits to the queue

3. **Container Execution** 🐳
   - Starts Docker container with GPU access
   - Maps your user ID for file permissions
   - Sets up Python environment and paths

4. **Application Launch** 🎯
   - Validates environment and files
   - Logs into WandB (if key provided)
   - Runs your LMTK experiment with proper arguments

### File Architecture Overview

```
slurm/
├── submit_job.sh      # 🎯 Main script - your entry point
├── p.slurm           # 🔧 SLURM job template (handles Docker)
├── run_container.sh  # 🐳 Container script (environment setup)
├── slurm_config.env  # ⚙️ Default settings for your cluster
└── README.md         # 📚 This documentation
```

**You only interact with `submit_job.sh`** - the other scripts work automatically.

---

## ⚙️ Step-by-Step Usage

### Level 1: Basic Usage (Single Job Customization)

These examples show how to customize individual job runs using command-line arguments:

#### 1.1 Adjusting Resources for Larger Experiments

```bash
# More GPUs for faster training
./submit_job.sh -c config/experiments/continual_llama_3.1_7b.yaml -g 4

# More memory for large datasets
./submit_job.sh -c config/experiments/large_dataset.yaml -m 128G

# Longer time for extended training
./submit_job.sh -c config/experiments/long_training.yaml -t 72:00:00

# Combine multiple resource adjustments
./submit_job.sh -c config/experiments/big_experiment.yaml -g 8 -m 256G -t 96:00:00
```

**When to use what:**
- **More GPUs (`-g`)**: Multi-GPU training, larger models, faster processing
- **More Memory (`-m`)**: Large datasets, big models, tokenization tasks
- **More Time (`-t`)**: Extended training, large datasets, complex experiments

#### 1.2 Different Cluster Partitions

```bash
# Use a specific GPU partition
./submit_job.sh -c config/experiments/test.yaml -p gpu_small

# High-priority partition (if available)
./submit_job.sh -c config/experiments/urgent.yaml -p gpu_priority

# CPU-only partition for tokenization
./submit_job.sh -c config/experiments/tokenize.yaml -p cpu -g 0
```

#### 1.3 Custom Job Names and Organization

```bash
# Descriptive job name for easy identification
./submit_job.sh -c config/experiments/test.yaml -j "llama-3b-continual-test"

# Custom output directory
./submit_job.sh -c config/experiments/test.yaml -o "experiment_$(date +%Y%m%d)_llama3b"

# Combine both for organized experiments
./submit_job.sh -c config/experiments/continual_llama_3.2_3b.yaml \
    -j "llama-3.2-3b-continual-v1" \
    -o "llama_3.2_3b_continual_$(date +%Y%m%d_%H%M)" \
    -k your_wandb_key
```

### Level 2: Systematic Configuration (Multi-Job Setup)

For running multiple similar experiments, modify `slurm_config.env` instead of repeating command-line arguments:

#### 2.1 Setting Up Your Environment File

Edit `slurm/slurm_config.env` to establish defaults for your research project:

```bash
# Example: Large-scale experiment setup
export GPU_COUNT="4"           # All jobs use 4 GPUs by default
export MEMORY="128G"           # All jobs get 128GB RAM
export TIME_LIMIT="72:00:00"   # All jobs get 72 hours
export PARTITION="gpu_large"   # Use high-memory partition

# Project-specific WandB settings
export WANDB_PROJECT="llama-continual-learning"
export WANDB_ENTITY="your_research_team"
```

#### 2.2 Benefits of Environment Configuration

After setting up `slurm_config.env`, all these commands use your defaults:

```bash
# All use 4 GPUs, 128GB RAM, 72h time limit automatically
./submit_job.sh -c config/experiments/experiment1.yaml -k your_key
./submit_job.sh -c config/experiments/experiment2.yaml -k your_key
./submit_job.sh -c config/experiments/experiment3.yaml -k your_key

# Override only when needed
./submit_job.sh -c config/experiments/small_test.yaml -g 1 -m 32G -t 2:00:00
```

#### 2.3 Environment vs Command Line

**Use `.env` configuration when:**
- Running multiple related experiments
- Working on a specific research project
- Want consistent resource allocation
- Team members sharing the same setup

**Use command-line arguments when:**
- Testing with different resources
- One-off experiments
- Overriding project defaults
- Quick prototyping

### Level 3: Advanced Workflows

#### 3.1 Batch Experiment Submission

```bash
# Run multiple configurations with same resources
for config in config/experiments/llama-3b/*.yaml; do
    echo "Submitting $(basename $config)"
    ./submit_job.sh -c "$config" -k your_wandb_key
    sleep 5  # Small delay between submissions
done
```

#### 3.2 Parameter Sweeps

```bash
# Test different GPU configurations
gpu_counts=(1 2 4 8)
for gpus in "${gpu_counts[@]}"; do
    ./submit_job.sh -c config/experiments/scaling_test.yaml \
        -g $gpus \
        -j "scaling-test-${gpus}gpu" \
        -o "scaling_experiment_${gpus}gpu" \
        -k your_wandb_key
done
```

#### 3.3 Different Resource Profiles

```bash
# Quick tests (small resources)
./submit_job.sh -c config/experiments/debug.yaml -g 1 -m 16G -t 30:00

# Development runs (medium resources) 
./submit_job.sh -c config/experiments/dev.yaml -g 2 -m 64G -t 4:00:00

# Production runs (full resources)
./submit_job.sh -c config/experiments/production.yaml -g 8 -m 512G -t 168:00:00
```

---

## 🎛️ Configuration Deep Dive

### Understanding Configuration Hierarchy

The system uses a **4-level hierarchy** where each level can override the previous ones:

```
Command Line Args  >  Environment Variables  >  slurm_config.env  >  Built-in Defaults
    (highest)                                                           (lowest)
```

#### Example of Hierarchy in Action:

```bash
# Built-in default: GPU_COUNT=1

# 1. slurm_config.env sets: GPU_COUNT=2
# 2. Environment variable: export GPU_COUNT=4  
# 3. Command line: ./submit_job.sh -c config.yaml -g 8

# Result: Job uses 8 GPUs (command line wins)
```

### Complete Parameter Reference

#### 🔧 Resource Configuration Parameters

| Parameter | CLI Flag | Default | Effect | When to Adjust |
|-----------|----------|---------|--------|----------------|
| **GPU Count** | `-g, --gpus` | 1 | Number of GPUs allocated | Multi-GPU training, larger models, faster processing |
| **Memory** | `-m, --memory` | 32G | RAM allocation | Large datasets, big models, OOM errors |
| **Time Limit** | `-t, --time` | 48:00:00 | Max job runtime | Long training, large datasets |
| **CPU Cores** | `--cpus` | 16 | CPU cores per task | Data preprocessing, tokenization |
| **Nodes** | `-n, --nodes` | 1 | Number of compute nodes | Distributed training |
| **Tasks per Node** | `--ntasks-per-node` | 1 | Parallel tasks | Multi-processing workflows |

#### 🏷️ Job Organization Parameters

| Parameter | CLI Flag | Default | Effect | When to Use |
|-----------|----------|---------|--------|-------------|
| **Job Name** | `-j, --job-name` | lmtk-experiment | SLURM job identifier | Organize multiple experiments |
| **Partition** | `-p, --partition` | postiguet1 | SLURM queue/partition | Different hardware, priorities |
| **Output Directory** | `-o, --output-dir` | auto-generated | Results storage location | Organize experiment outputs |

#### 📊 Experiment Tracking Parameters

| Parameter | CLI Flag | Default | Effect | When to Use |
|-----------|----------|---------|--------|-------------|
| **WandB API Key** | `-k, --wandb-key` | none | Enables experiment tracking | Production experiments |
| **WandB Project** | `--wandb-project` | lmtk-experiments | Project organization | Different research projects |
| **WandB Entity** | `--wandb-entity` | gplsi_continual | Team/organization | Shared team experiments |

#### 🛠️ Utility Parameters

| Parameter | CLI Flag | Effect | When to Use |
|-----------|----------|--------|-------------|
| **Dry Run** | `-d, --dry-run` | Show command without executing | Testing, debugging |
| **Help** | `-h, --help` | Display usage information | Learning, reference |

### Detailed Parameter Effects

#### GPU Count (`-g, --gpus`)
```bash
# Single GPU (debugging, small models)
./submit_job.sh -c config.yaml -g 1

# Multi-GPU (faster training, larger models)  
./submit_job.sh -c config.yaml -g 4

# Maximum utilization (large-scale experiments)
./submit_job.sh -c config.yaml -g 8
```

**Impact:**
- **Training Speed**: More GPUs = faster training (if model supports it)
- **Model Size**: Larger models require more GPU memory
- **Cost**: More GPUs = higher resource usage
- **Queue Time**: More GPUs may mean longer wait times

#### Memory (`-m, --memory`)
```bash
# Small experiments, testing
./submit_job.sh -c config.yaml -m 16G

# Standard training
./submit_job.sh -c config.yaml -m 64G

# Large datasets, big models
./submit_job.sh -c config.yaml -m 256G

# Tokenization, data processing
./submit_job.sh -c config.yaml -m 512G
```

**Signs you need more memory:**
- `OutOfMemoryError` in logs
- Job killed by SLURM
- Large datasets not fitting in RAM
- Complex tokenization tasks

#### Time Limit (`-t, --time`)
```bash
# Quick tests
./submit_job.sh -c config.yaml -t 1:00:00

# Standard experiments  
./submit_job.sh -c config.yaml -t 24:00:00

# Long training runs
./submit_job.sh -c config.yaml -t 168:00:00  # 7 days

# Data preprocessing
./submit_job.sh -c config.yaml -t 12:00:00
```

**Format options:**
- `HH:MM:SS` (e.g., `2:30:00` = 2.5 hours)
- `DD-HH:MM:SS` (e.g., `3-12:00:00` = 3 days, 12 hours)

### Sample `slurm_config.env` Configurations

#### Configuration A: Development & Testing
```bash
# Quick iteration setup
export JOB_NAME="dev-test"
export PARTITION="gpu_small"
export GPU_COUNT="1"
export MEMORY="32G"
export TIME_LIMIT="4:00:00"
export CPUS_PER_TASK="8"

# Development WandB project
export WANDB_PROJECT="lmtk-development"
export WANDB_ENTITY="your_username"
```

#### Configuration B: Production Training
```bash
# Large-scale experiment setup
export JOB_NAME="production-training"
export PARTITION="gpu_large"
export GPU_COUNT="4"
export MEMORY="128G"
export TIME_LIMIT="72:00:00"
export CPUS_PER_TASK="32"

# Production WandB project
export WANDB_PROJECT="lmtk-production"
export WANDB_ENTITY="your_research_team"
```

#### Configuration C: Data Processing
```bash
# Tokenization and preprocessing setup
export JOB_NAME="data-processing"
export PARTITION="cpu"
export GPU_COUNT="0"          # No GPUs needed
export MEMORY="256G"          # High memory for large datasets
export TIME_LIMIT="24:00:00"
export CPUS_PER_TASK="64"     # Many CPUs for parallel processing
```

### Environment Variable Override Examples

#### Temporary Overrides (Single Session)
```bash
# Override for current terminal session
export GPU_COUNT=8
export MEMORY="256G"
export WANDB_PROJECT="special-experiment"

# All subsequent jobs use these settings
./submit_job.sh -c config/experiment1.yaml -k your_key
./submit_job.sh -c config/experiment2.yaml -k your_key

# Until you close the terminal or unset them
unset GPU_COUNT MEMORY WANDB_PROJECT
```

#### Per-Job Overrides (One-Line)
```bash
# Override just for this one command
GPU_COUNT=2 MEMORY="64G" ./submit_job.sh -c config.yaml -k your_key

# Multiple variables
GPU_COUNT=4 MEMORY="128G" WANDB_PROJECT="test-run" \
    ./submit_job.sh -c config.yaml -k your_key
```

#### Script-Based Overrides
```bash
#!/bin/bash
# Set environment for a series of related experiments

export GPU_COUNT=4
export MEMORY="128G"  
export TIME_LIMIT="48:00:00"
export WANDB_PROJECT="llama-ablation-study"

# Run ablation experiments
for lr in 1e-4 5e-5 1e-5; do
    ./submit_job.sh -c "config/experiments/llama_lr_${lr}.yaml" \
        -j "llama-lr-${lr}" \
        -o "llama_ablation_lr_${lr}_$(date +%Y%m%d_%H%M)" \
        -k your_wandb_key
done
```

## 📊 Advanced Features

### 🐳 Docker Container Integration

#### What Happens Inside the Container

When your job runs, it creates a Docker container that:

1. **Maps Your User Identity**
   ```bash
   # Container runs as your user (not root)
   --user "$USER_ID:$GROUP_ID"
   ```
   **Benefit**: Files created have correct ownership

2. **Mounts Project Directory**
   ```bash
   # Your entire project is available at /workspace
   --volume "$HOST_PROJECT_ROOT:$CONTAINER_PROJECT_ROOT"
   ```
   **Benefit**: Code changes and results are preserved

3. **Provides GPU Access**
   ```bash
   # All allocated GPUs are available
   --gpus all
   ```
   **Benefit**: Full CUDA support for training

4. **Sets Up Python Environment**
   ```bash
   # Automatic PYTHONPATH configuration
   PYTHONPATH=/workspace/src:/workspace:$PYTHONPATH
   ```
   **Benefit**: All imports work correctly

#### Container Environment Details

The container automatically configures:

```bash
# Python and Package Management
PYTHONPATH=/workspace/src:/workspace
PYTHON_COMMAND=python3
MAIN_SCRIPT=src/main.py

# HuggingFace Cache (saves bandwidth)
HF_DATASETS_CACHE=/workspace/.cache/datasets
HF_HOME=/workspace/.cache/huggingface  
TRANSFORMERS_CACHE=/workspace/.cache/transformers

# CUDA Environment
CUDA_VISIBLE_DEVICES=0,1,2,3  # Based on allocated GPUs

# SLURM Integration
SLURM_JOB_ID=12345
SLURM_NODELIST=gpu-node-01
```

### 🔄 WandB Integration Deep Dive

#### Automatic Authentication Flow

When you provide a WandB API key, the container:

1. **Validates Configuration**
   ```bash
   echo "WandB API key provided, logging in..."
   echo "WANDB_PROJECT: ${WANDB_PROJECT}"
   echo "WANDB_ENTITY: ${WANDB_ENTITY}"
   ```

2. **Performs Login**
   ```bash
   wandb login "$WANDB_API_KEY"
   ```

3. **Handles Results**
   ```bash
   if [ $? -eq 0 ]; then
       echo "✅ WandB login successful"
   else
       echo "❌ WandB login failed"
       exit 1  # Job fails to prevent invalid experiments
   fi
   ```

#### WandB Configuration Options

```bash
# Basic tracking
./submit_job.sh -c config.yaml -k your_api_key

# Custom project and entity
./submit_job.sh -c config.yaml -k your_api_key \
    --wandb-project "my-research-project" \
    --wandb-entity "my-team"

# Environment variable approach
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="continual-learning"
export WANDB_ENTITY="research-lab"
./submit_job.sh -c config.yaml
```

#### WandB Best Practices

**Project Organization:**
```bash
# By model type
WANDB_PROJECT="llama-experiments"
WANDB_PROJECT="gpt-experiments"

# By research area  
WANDB_PROJECT="continual-learning"
WANDB_PROJECT="multilingual-models"

# By development stage
WANDB_PROJECT="development"
WANDB_PROJECT="production"
```

**Entity Usage:**
```bash
# Personal experiments
WANDB_ENTITY="your_username"

# Team experiments
WANDB_ENTITY="your_research_team"

# Organization-wide
WANDB_ENTITY="your_institution"
```

### 🎯 Practical Use Cases & Examples

#### Use Case 1: Model Development Workflow

```bash
# 1. Quick prototype (minimal resources)
./submit_job.sh -c config/experiments/prototype.yaml \
    -g 1 -m 16G -t 2:00:00 \
    -j "prototype-test" \
    -k your_wandb_key

# 2. Development iteration (medium resources)
./submit_job.sh -c config/experiments/development.yaml \
    -g 2 -m 64G -t 8:00:00 \
    -j "dev-iteration-v1" \
    -k your_wandb_key

# 3. Full training (production resources)
./submit_job.sh -c config/experiments/production.yaml \
    -g 8 -m 256G -t 72:00:00 \
    -j "production-training-v1" \
    -k your_wandb_key
```

#### Use Case 2: Dataset Processing Pipeline

```bash
# 1. Tokenization (CPU-intensive, high memory)
./submit_job.sh -c config/experiments/tokenization.yaml \
    -g 0 -m 512G -t 24:00:00 --cpus 64 \
    -p cpu \
    -j "tokenize-dataset"

# 2. Dataset validation (minimal resources)
./submit_job.sh -c config/experiments/validate_dataset.yaml \
    -g 1 -m 32G -t 2:00:00 \
    -j "validate-tokenized-data"

# 3. Training on processed data
./submit_job.sh -c config/experiments/train_on_tokenized.yaml \
    -g 4 -m 128G -t 48:00:00 \
    -j "training-tokenized-data" \
    -k your_wandb_key
```

#### Use Case 3: Hyperparameter Exploration

```bash
# Set common configuration
export GPU_COUNT=2
export MEMORY="64G"
export TIME_LIMIT="12:00:00"
export WANDB_PROJECT="hyperparameter-search"

# Learning rate sweep
learning_rates=(1e-4 5e-5 1e-5 5e-6)
for lr in "${learning_rates[@]}"; do
    ./submit_job.sh -c "config/experiments/lr_${lr}.yaml" \
        -j "lr-search-${lr}" \
        -o "lr_search_${lr}_$(date +%Y%m%d)" \
        -k your_wandb_key
done

# Batch size sweep  
batch_sizes=(16 32 64 128)
for bs in "${batch_sizes[@]}"; do
    ./submit_job.sh -c "config/experiments/bs_${bs}.yaml" \
        -j "bs-search-${bs}" \
        -k your_wandb_key
done
```

#### Use Case 4: Multi-Stage Experiment

```bash
#!/bin/bash
# Complex experiment with multiple stages

# Stage 1: Data preparation
echo "Starting data preparation..."
job1=$(./submit_job.sh -c config/experiments/stage1_data_prep.yaml \
    -g 0 -m 256G -t 8:00:00 --cpus 32 \
    -j "stage1-data-prep" | grep -o '[0-9]\+')

# Stage 2: Model training (depends on stage 1)
echo "Starting model training (depends on job $job1)..."
job2=$(sbatch --dependency=afterok:$job1 \
    --job-name="stage2-training" \
    --export=CONFIG_FILE=config/experiments/stage2_training.yaml,WANDB_API_KEY=your_key \
    slurm/p.slurm | grep -o '[0-9]\+')

# Stage 3: Evaluation (depends on stage 2)
echo "Starting evaluation (depends on job $job2)..."
sbatch --dependency=afterok:$job2 \
    --job-name="stage3-evaluation" \
    --export=CONFIG_FILE=config/experiments/stage3_evaluation.yaml \
    slurm/p.slurm

echo "Multi-stage pipeline submitted successfully!"
```

### 🔧 Advanced Configuration Strategies

#### Strategy 1: Environment-Based Profiles

Create different environment files for different scenarios:

```bash
# slurm_config_dev.env (development)
export GPU_COUNT="1"
export MEMORY="32G"  
export TIME_LIMIT="4:00:00"
export WANDB_PROJECT="development"

# slurm_config_prod.env (production)
export GPU_COUNT="4"
export MEMORY="128G"
export TIME_LIMIT="72:00:00" 
export WANDB_PROJECT="production"
```

Use them selectively:
```bash
# Load development profile
source slurm/slurm_config_dev.env
./submit_job.sh -c config.yaml -k your_key

# Load production profile
source slurm/slurm_config_prod.env  
./submit_job.sh -c config.yaml -k your_key
```

#### Strategy 2: Resource Templates

```bash
# Resource calculation based on model size
model_size=$1  # Pass as argument: small, medium, large

case $model_size in
    "small")
        GPU_COUNT=1; MEMORY="32G"; TIME="8:00:00"
        ;;
    "medium") 
        GPU_COUNT=2; MEMORY="64G"; TIME="24:00:00"
        ;;
    "large")
        GPU_COUNT=4; MEMORY="128G"; TIME="72:00:00"
        ;;
    "xlarge")
        GPU_COUNT=8; MEMORY="256G"; TIME="168:00:00"
        ;;
esac

./submit_job.sh -c config/experiments/${model_size}_model.yaml \
    -g $GPU_COUNT -m $MEMORY -t $TIME \
    -k your_wandb_key
```

#### Strategy 3: Conditional Configuration

```bash
#!/bin/bash
# Intelligent resource allocation

config_file=$1
wandb_key=$2

# Extract experiment type from config filename
if [[ $config_file == *"tokeniz"* ]]; then
    # Tokenization: CPU-heavy, high memory
    resources="-g 0 -m 512G --cpus 64 -p cpu"
elif [[ $config_file == *"llama"* ]] && [[ $config_file == *"7b"* ]]; then
    # Large model: Multi-GPU, high memory
    resources="-g 4 -m 256G -t 72:00:00"
elif [[ $config_file == *"test"* ]] || [[ $config_file == *"debug"* ]]; then
    # Testing: Minimal resources
    resources="-g 1 -m 16G -t 1:00:00"
else
    # Default: Standard resources
    resources="-g 2 -m 64G -t 24:00:00"
fi

echo "Auto-detected resources: $resources"
./submit_job.sh -c "$config_file" $resources -k "$wandb_key"
```

## 📈 Monitoring Jobs

```bash
# Check job status
squeue -u $(whoami)

# Detailed job information  
scontrol show job JOBID

# Job accounting and efficiency
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Start,End,ElapsedTime

# Monitor job logs in real-time (replace JOB_ID)
tail -f JOB_ID_lmtk.out
tail -f JOB_ID_lmtk.err

# Follow both logs simultaneously
tail -f JOB_ID_lmtk.out JOB_ID_lmtk.err

# Cancel job
scancel JOB_ID
```

### 🆕 Enhanced Log Output
The new container system provides detailed logs including:

**Standard Output (`JOBID_lmtk.out`):**
- Container startup information
- Environment variable verification  
- File system validation
- Python environment details
- WandB configuration status
- Command execution details
- Training/tokenization progress

**Error Output (`JOBID_lmtk.err`):**
- Bash script execution trace (with `set -x`)
- Detailed command-by-command execution
- Error messages and stack traces
- Container debugging information

### Log Sections
```bash
===== Container Script Started =====
# Basic environment info (user, directory, Python version)

===== Environment Variables =====  
# All relevant environment variables and their values

===== Checking File System =====
# Verification that all required files exist

===== Python Environment =====
# Python executable, PYTHONPATH, installed packages

===== WandB Configuration =====
# WandB setup, login status, project settings

===== Starting Training/Tokenization =====
# Command execution and application output
```

## 🔍 Troubleshooting

### 🚨 Quick Problem Resolution

| Issue | Symptoms | Quick Fix |
|-------|----------|-----------|
| **Job fails immediately** | `Job violates accounting policy` | Reduce resources: `-g 1 -m 16G -t 1:00:00` |
| **No application output** | Container runs, no logs | Check `JOBID_lmtk.err` for detailed trace |
| **Config file not found** | `Config file not found: /workspace/...` | Use absolute path or check file exists |
| **WandB login fails** | `❌ WandB login failed` | Verify API key: `wandb login your_key` |
| **Python import errors** | `ModuleNotFoundError` | Check PYTHONPATH in container logs |
| **Out of memory** | Job killed, no output | Increase memory: `-m 128G` or `-m 256G` |
| **Time limit exceeded** | Job killed after time limit | Increase time: `-t 72:00:00` |
| **Docker image missing** | `Container finished with exit code: 0` | Rebuild: `docker build -t lmtk:latest .` |

### Common Issues

#### 1. **Argument Parsing Error**
```
main.py: error: unrecognized arguments: config/experiments/gpt-2/tokenizer.yaml
```
**Solution**: This has been fixed! The system now correctly uses `--config` flag as expected by main.py.

#### 2. **PYTHONPATH Unbound Variable Error**
```
/var/spool/slurm/job14900/slurm_script: line 241: PYTHONPATH: unbound variable
```
**Solution**: This has been fixed! The script now handles undefined PYTHONPATH gracefully.

#### 3. **Container Output Missing**
**Problem**: Container runs but no application output visible
**Solution**: 
- Check `JOBID_lmtk.err` for detailed execution trace
- Look for the "Starting Training/Tokenization" section
- Verify the configuration file path is correct

#### 4. **WandB Login Failures**
```
❌ WandB login failed
```
**Solutions**:
- Verify API key is correct: `wandb login your_api_key` 
- Check network connectivity from compute nodes
- Verify project and entity names exist in your WandB account
- Try running without WandB first to isolate the issue

#### 5. **Configuration File Not Found**
```
Config file not found: /workspace/config/experiments/gpt-2/tokenizer.yaml
```
**Solutions**:
- Ensure path is correct relative to project root
- Use absolute path if needed: `-c /full/path/to/config.yaml`
- Verify file exists: `ls -la config/experiments/gpt-2/tokenizer.yaml`

#### 6. **Docker Image Issues**
```
✅ Using existing Docker image: lmtk:latest
Container finished with exit code: 0
```
**Problem**: Container exits immediately without running the application
**Solutions**:
- Check if Docker image was built correctly: `docker images | grep lmtk`
- Verify Dockerfile exists: `ls -la docker/Dockerfile`
- Rebuild image if needed: `docker build -t lmtk:latest -f docker/Dockerfile .`

#### 7. **Resource Allocation Denied**
```
sbatch: error: Batch job submission failed: Job violates accounting policy
```
**Solutions**:
- Reduce resource requirements: `-g 1 -m 16G -t 1:00:00`
- Try different partition: `-p gpu_small` or `-p cpu`
- Check cluster availability: `sinfo -p your_partition`
- Verify resource limits: `sacctmgr show qos format=name,maxjobs,maxwall`

#### 8. **Python Import Errors**
```
ModuleNotFoundError: No module named 'src.config.config_loader'
```
**Solutions**:
- Verify PYTHONPATH is set correctly (check container logs)
- Ensure all dependencies are installed in Docker image
- Check that src/ directory structure is correct

### 🆕 Debug Mode & Diagnostics

#### Dry Run
Use dry run to see exactly what will be executed:
```bash
./submit_job.sh -c config.yaml --dry-run
```

#### Container Debugging
The new container script provides extensive debugging information:
1. **Environment inspection** - All variables and their values
2. **File system validation** - Confirms all required files exist  
3. **Python environment** - Shows executable, paths, installed packages
4. **Command verification** - Shows exact command being executed

#### Validation Script
Run comprehensive validation before submitting:
```bash
./validate.sh
```

#### Manual Container Testing
Test the container locally:
```bash
docker run --rm -it \
  --gpus all \
  --volume "$PWD:/workspace" \
  --env CONFIG_FILE=config/experiments/test_continual.yaml \
  --env PYTHON_COMMAND=python3 \
  --env MAIN_SCRIPT=src/main.py \
  lmtk:latest \
  /workspace/slurm/run_container.sh
```

## 🔒 Security Notes

- ❌ **Never commit API keys to version control**
- ✅ Use command line parameters or environment variables for sensitive data
- ✅ API keys are only stored temporarily during job execution
- ✅ All secrets are passed via SLURM's secure environment passing
- ✅ Container runs with your UID/GID for proper file ownership
- ✅ No hardcoded credentials in any script files

## 🚀 Advanced Usage

### Environment Variable Override
Override any setting with environment variables:
```bash
export GPU_COUNT=4
export MEMORY="128G"
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="my-custom-project"
./submit_job.sh -c config/experiments/your_experiment.yaml
```

### Custom Output Directories
```bash
# Specify custom output directory
./submit_job.sh -c config/experiments/your_experiment.yaml \
    -o my_custom_experiment_$(date +%Y%m%d)
```

### Batch Submissions
```bash
# Submit multiple experiments
for config in config/experiments/*.yaml; do
    echo "Submitting $config"
    ./submit_job.sh -c "$config" -k your_wandb_key
    sleep 5  # Small delay between submissions
done

# Submit with different resource configurations
configs=("config1.yaml" "config2.yaml" "config3.yaml")
gpus=(1 2 4)
for i in "${!configs[@]}"; do
    ./submit_job.sh -c "${configs[$i]}" -g "${gpus[$i]}" -k your_wandb_key
done
```

### 🆕 Container Customization
The run_container.sh script can be customized for specific needs:

```bash
# Add custom environment setup
echo "===== Custom Setup ====="
export CUSTOM_VAR="value"
# ... your custom setup code ...
echo "========================="
```

### Resource Optimization
```bash
# Memory-intensive tasks
./submit_job.sh -c config.yaml -g 2 -m 256G --cpus 32

# Quick testing with minimal resources  
./submit_job.sh -c config.yaml -g 1 -m 8G -t 30:00 --cpus 4

# Large-scale multi-node training
./submit_job.sh -c config.yaml -g 8 --nodes 2 -m 512G -t 168:00:00
```

## ✅ Validation

Before submitting jobs, validate your SLURM setup:

```bash
# Run comprehensive validation checks
./validate.sh
```

The validation script checks:
- ✅ **File existence** - Scripts, configs, Dockerfile, main.py
- ✅ **Configuration consistency** - Default values and paths  
- ✅ **Docker availability** - Image existence and buildability
- ✅ **Project structure** - Required directories and files
- ✅ **Environment setup** - Python paths and dependencies

### Sample Validation Output
```
===== SLURM Script Validation =====
Script Directory: /home/user/LMTK/slurm
SLURM Script: /home/user/LMTK/slurm/p.slurm
Config File: /home/user/LMTK/slurm/slurm_config.env
Submit Script: /home/user/LMTK/slurm/submit_job.sh
Container Script: /home/user/LMTK/slurm/run_container.sh
===============================

=== File Validation ===
✅ Project root exists
✅ SLURM script exists
✅ Submit script exists  
✅ Container script exists
✅ Dockerfile exists
✅ Main script exists
✅ Configuration file exists

=== 🆕 Container Integration ===
✅ Container script is executable
✅ Docker image available
✅ Environment variables properly configured
```

## 🔄 Recent Improvements & Changelog

### Version 2.0 - Enhanced Container Integration
- 🆕 **Dedicated container script** (`run_container.sh`) for better debugging
- 🆕 **Automatic WandB login** with proper error handling
- 🆕 **Enhanced logging** with environment inspection and validation
- 🆕 **Fixed argument parsing** - Now correctly uses `--config` flag
- 🆕 **Resolved PYTHONPATH issues** - Handles undefined variables gracefully
- 🆕 **Comprehensive error reporting** - Clear debugging information

### Migration from v1.0
If you're upgrading from the previous version:
1. **No configuration changes needed** - All existing configs work
2. **Enhanced debugging** - More detailed logs automatically available
3. **Better error messages** - Clearer troubleshooting guidance
4. **Improved reliability** - Fixed several edge cases and bugs

## 🏗️ Cluster Adaptation

To adapt for your specific cluster:

### 1. Update Default Configuration
Edit `slurm_config.env`:
```bash
# Modify for your cluster
export PARTITION="your_gpu_partition"     # Your cluster's GPU partition name
export WANDB_ENTITY="your_wandb_team"     # Your WandB organization
export DOCKER_IMAGE_NAME="your_image:tag" # Custom Docker image if needed
```

### 2. Adjust Resource Defaults
```bash
# For clusters with different GPU types
export GPU_COUNT="2"                      # Default GPU allocation
export MEMORY="64G"                       # Default memory allocation  
export CPUS_PER_TASK="32"                # Default CPU allocation
```

### 3. Module Loading (if required)
Uncomment and modify the module loading section in `p.slurm`:
```bash
# Example for clusters requiring module loading
module load cuda/11.8
module load singularity
```

### 4. Custom Paths
Update project-specific paths if needed:
```bash
export PROJECT_NAME="YOUR_PROJECT_NAME"
export HOST_PROJECT_ROOT="/custom/path/to/project"
```
