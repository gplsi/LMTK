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
experiment_name: "tutorial_clm_training"
model_name: "openai-community/gpt2"
dataset:
  nameOrPath: "tutorials/data/raw_text_data/quijote.txt"
  format: "files"
output_dir: "tutorials/output/clm_training"
parallelization_strategy: "fsdp"
# ... more options ...
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
