# Pretraining Task

This module implements continual pretraining capabilities for language models with support for various distributed training strategies.

## Overview

The pretraining task enables efficient continual training of language models using multiple distributed strategies:
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed with ZeRO optimization
- Distributed Data Parallel (DDP)
- Data Parallel (DP)

## Configuration

Example configuration:
```yaml
task: pretraining
output_dir: outputs/gpt2_pretrain
model_name: gpt2
precision: "bf16-mixed"  # Options: "32-true", "16-mixed", "bf16-mixed"
number_epochs: 3
batch_size: 32
gradient_accumulation_steps: 4

# Training parameters
lr: 1e-4
lr_scheduler: "cosine"  # Options: "constant", "linear", "cosine"
warmup_proportion: 0.1
weight_decay: 0.01
beta1: 0.9
beta2: 0.999

# Dataset configuration
dataset:
  source: "local"  # Options: "local", "huggingface"
  nameOrPath: "path/to/dataset"

# Distributed training configuration
parallelization_strategy: "fsdp"  # Options: "fsdp", "deepspeed", "ddp", "dp"
```

## Features

- **Multiple Distribution Strategies**: Support for FSDP, DeepSpeed, DDP, and DP
- **Gradient Accumulation**: Train with large effective batch sizes on limited hardware
- **Mixed Precision Training**: Support for FP16 and BF16 mixed precision
- **Checkpointing**: Save and resume training from checkpoints
- **Monitoring**: Built-in training speed and loss monitoring
- **Flexible Dataset Loading**: Support for local and HuggingFace datasets

## Usage

1. Create a configuration file following the schema in `config/schemas/pretraining.schema.yaml`
2. Run the pretraining task:
```bash
python src/main.py config/experiments/your_config.yaml
```

## Implementation Details

- `orchestrator.py`: Manages the training workflow and strategy selection
- `fabric/`: Contains Lightning Fabric implementations of training strategies
  - `base.py`: Base trainer with common functionality
  - `distributed.py`: Distributed training strategy implementations
  - `generation.py`: Model generation and forward pass handling
  - `speed_monitor.py`: Training speed monitoring utilities

## Advanced Features

### Gradient Accumulation
Configure larger effective batch sizes:
```yaml
batch_size: 8
gradient_accumulation_steps: 4  # Effective batch size = 32
```

### DeepSpeed ZeRO
Optimize memory usage with DeepSpeed:
```yaml
parallelization_strategy: "deepspeed"
zero_stage: 3  # ZeRO optimization stage
offload_optimizer: true  # CPU offloading for memory efficiency
```

### FSDP Configuration
Fine-tune FSDP behavior:
```yaml
parallelization_strategy: "fsdp"
auto_wrap_policy: "gpt2"  # Model-specific wrapping
sharding_strategy: "FULL_SHARD"
```