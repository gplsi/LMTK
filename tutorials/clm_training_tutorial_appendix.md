# Appendix: Running the CLM Training Tutorial with YAML Configuration

## Complete YAML Configuration

Below is a complete YAML configuration file for the CLM training module that follows the schema defined in the framework. You can save this to a file (e.g., `clm_training_config.yaml`) and use it to run the training task:

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

Make sure to replace the following placeholders with your actual values:
- `/path/to/your/tokenized_dataset`: The path to your tokenized dataset
- `/path/to/output_directory`: The directory where checkpoints and logs will be saved

## Running the CLM Training Task

To run the CLM training task using the YAML configuration:

```bash
# Navigate to the root directory of the framework
cd /path/to/continual-pretraining-framework

# Run the training with the configuration file
python src/main.py /path/to/clm_training_config.yaml
```

For distributed training with multiple GPUs:

```bash
# Using torchrun for distributed training
torchrun --nproc_per_node=NUM_GPUS src/main.py /path/to/clm_training_config.yaml
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Important Notes

1. **Dataset Preparation**: Ensure your dataset is properly tokenized before training. You can use the tokenization module of the framework for this purpose.

2. **GPU Requirements**: For efficient training, CUDA-enabled GPUs are recommended. The framework will use all available GPUs by default when using distributed training.

3. **Memory Considerations**: Adjust the batch size, gradient accumulation steps, and model size based on your available GPU memory.

4. **Checkpointing**: The framework automatically saves checkpoints at the frequency specified by `save_steps`. These checkpoints can be used for model publishing.

5. **Evaluation**: The framework performs evaluation at the frequency specified by `eval_steps`. Evaluation metrics are logged to the output directory.

6. **Logging**: The `verbose_level` parameter controls the amount of logging output. Higher values (1-4) provide more detailed logs.

7. **Distributed Strategies**: The framework supports multiple distributed training strategies (FSDP, DDP, DeepSpeed, DP). Choose the one that best fits your hardware and training requirements.
