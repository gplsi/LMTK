# Appendix: Running the Publish Tutorial with YAML Configuration

## Complete YAML Configuration

Below is a complete YAML configuration file for the publish module that follows the schema defined in the framework. You can save this to a file (e.g., `publish_config.yaml`) and use it to run the publish task:

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

Make sure to replace the following placeholders with your actual values:
- `/path/to/your/checkpoint.pt`: The path to your trained model checkpoint
- `your-username/your-model-name`: Your HuggingFace username and desired model name

## Running the Publish Task

To run the publish task using the YAML configuration:

```bash
# Navigate to the root directory of the framework
cd /path/to/continual-pretraining-framework

# Run the publishing with the configuration file
python src/main.py /path/to/publish_config.yaml
```

## Integration with CLM Training

If you've completed the CLM training tutorial, you can use the checkpoint generated from that process for publishing:

```yaml
publish:
  format: fsdp
  base_model: gpt2
  checkpoint_path: /path/to/output_directory/checkpoint.pt  # Path to your CLM training output
  host: huggingface
  repo_id: your-username/your-model-name
```

## Important Notes

1. **HuggingFace Authentication**: Before running the publish task, ensure you are authenticated with HuggingFace:
   ```bash
   huggingface-cli login
   ```

2. **Configuration Validation**: The framework validates all configuration files against JSON schemas, so make sure your YAML file follows the correct schema.

3. **Checkpoint Format**: The `format` parameter must match the format of your checkpoint. For checkpoints trained with FSDP, use `format: fsdp`.

4. **Base Model**: The `base_model` parameter should match the model used during training.

5. **Repository ID**: The `repo_id` parameter must follow the format `username/model-name` and you must have write permissions to this repository.
