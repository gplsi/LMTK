# Running the Tutorials with YAML Configuration

This document explains how to run the CLM Training and Publish tutorials using YAML configuration files and the framework's main entry point.

## Running the CLM Training Tutorial

### Complete YAML Configuration

The `clm_training_config_example.yaml` file contains a complete configuration for CLM training:

```yaml
task: pretraining
experiment_name: tutorial_clm_training
verbose_level: 4

# Model configuration
model:
  name: gpt2
  pretrained: true

# Dataset configuration
dataset:
  source: local
  nameOrPath: /path/to/your/tokenized_dataset
  streaming: false
  shuffle: true
  
# Training configuration
training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_steps: 1000
  save_steps: 200
  eval_steps: 200
  logging_steps: 50
  
# Distributed training configuration
distributed:
  strategy: fsdp
  fsdp:
    sharding_strategy: FULL_SHARD
    cpu_offload: false
    mixed_precision: true
    
# Output configuration
output:
  dir: /path/to/output_directory
```

### Terminal Command to Run CLM Training

To run the CLM training with the YAML configuration:

```bash
# Navigate to the root directory of the framework
cd /path/to/continual-pretraining-framework

# Run the training with the configuration file
python src/main.py /path/to/clm_training_config_example.yaml
```

For distributed training with multiple GPUs:

```bash
# Using torchrun for distributed training
torchrun --nproc_per_node=NUM_GPUS src/main.py /path/to/clm_training_config_example.yaml
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Running the Publish Tutorial

### Complete YAML Configuration

The `publish_config_example.yaml` file contains a complete configuration for model publishing:

```yaml
task: publish
experiment_name: tutorial_publish
verbose_level: 4

# Publish configuration
publish:
  # Format conversion
  format: fsdp
  base_model: gpt2
  checkpoint_path: /path/to/your/checkpoint.pt
  
  # Upload configuration
  host: huggingface
  repo_id: your-username/your-model-name
  commit_message: Add gpt2 model trained with Continual Pretraining Framework
  
  # Advanced options
  max_shard_size: 5GB
  safe_serialization: true
  create_pr: false
```

### Terminal Command to Run Model Publishing

To run the model publishing with the YAML configuration:

```bash
# Navigate to the root directory of the framework
cd /path/to/continual-pretraining-framework

# Run the publishing with the configuration file
python src/main.py /path/to/publish_config_example.yaml
```

## Important Notes

1. **Path Customization**: Make sure to update all paths in the YAML files to match your actual file system paths.

2. **HuggingFace Authentication**: Before running the publish task, ensure you are authenticated with HuggingFace:
   ```bash
   huggingface-cli login
   # Or use the Python API
   # from huggingface_hub import login
   # login()
   ```

3. **GPU Requirements**: The framework works best with CUDA-enabled GPUs, but will fall back to CPU if no GPUs are available.

4. **Configuration Validation**: The framework validates all configuration files against JSON schemas, so make sure your YAML files follow the correct schema.

5. **Logging**: The `verbose_level` parameter controls the amount of logging output. Higher values (1-4) provide more detailed logs.

6. **Output Directory**: Always specify an output directory that exists and has write permissions.

7. **Checkpoint Path**: For publishing, ensure the checkpoint path points to a valid checkpoint file created by the CLM training process.
