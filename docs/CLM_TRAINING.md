# Causal Language Model Training Task Guide

## Overview

This guide explains the continual pretraining workflow in LMTK, including configuration, execution, and best practices for large-scale language model adaptation.

## What is Continual Pretraining?
Continual pretraining refers to further training of a pretrained language model on new data, allowing adaptation to new domains or tasks while retaining prior knowledge.

## Key Features
- **Flexible YAML Configuration**: All aspects of training are controlled via type-safe YAML configs.
- **Resumable Training**: Checkpoints allow training to resume from any point.
- **Distributed Training**: Supports FSDP, DeepSpeed, and multi-GPU setups.
- **Curriculum Learning**: Phase-wise data mixing and curriculum config support.

## Example Workflow

1. **Prepare your config** (see `tutorials/clm_training_tutorial.yaml`):

```yaml
task: "clm_training"
experiment_name: "quijote_gpt2_clm_training"
verbose_level: 4

# Model configuration
dataset:
  source: "local"
  nameOrPath: "data/tokenized/quijote"
  format: "hf"

model_name: "openai-community/gpt2"
precision: "bf16-true"

# Training arguments
number_epochs: 1
batch_size: 8
gradient_accumulation: true
gradient_accumulation_steps: 1
grad_clip: 1.0
lr:  0.00002
lr_decay: true
weight_decay: 0.01
beta1: 0.9
beta2: 0.95
lr_scheduler: "warmup_linear"
warmup_proportion: 0.06

# Validation configuration
validate_after_epoch: false
validate_on_end: false
validate_after_k_steps: null

# Saving configuration
save_on_validate: false
save_on_end: true

# Output
output_dir: "output/"

# Parallelization
parallelization_strategy: "fsdp"

# FSDP specific parameters
auto_wrap_policy: "gpt2"
sharding_strategy: "FULL_SHARD"
state_dict_type: "sharded"
limit_all_gathers: true
cpu_offload: false
num_workers: 4
gradient_checkpointing: true

# Logging
logging_config: "wandb"

# Wandb logging specific parameters
wandb_project: "prueba_publish"
wandb_entity: "gplsi_continual"
log_model: false
log_iter_interval: 10


# Seed
seed: 42
```

2. **Launch training**:

```bash
python src/main.py --config tutorials/clm_training_tutorial.yaml
```

3. **Monitor progress**:
- Logs and checkpoints are saved in the output directory.
- Use Weights & Biases or TensorBoard for live monitoring.

## Tips & Best Practices
- Always validate your config with `make validate` before launching jobs.
- Use curriculum configs for multi-phase or domain-adaptive training.
- For SLURM clusters, use the provided scripts in `slurm/`.

## References
- See `tutorials/clm_training_tutorial.ipynb` for a full notebook walkthrough.
- See the main README for installation and environment setup.
