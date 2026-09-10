# Publish Task Guide

## Overview

This guide describes how to use LMTK's publish task to upload trained models, tokenizers, or datasets to the Hugging Face Hub.

## Key Features
- **Automated Uploads**: Publish models and assets directly from your training pipeline.
- **YAML-Driven**: All publish parameters are specified in a config file.
- **Authentication**: Supports both environment variable and CLI-based Hugging Face authentication.
- **Safe Serialization**: Optionally enable safe serialization for maximum compatibility.

## Example Workflow

1. **Prepare your publish config** (see `tutorials/configs/publish_tutorial.yaml`):

```yaml
task: "publish"
experiment_name: "quijote_gpt2_publish"
verbose_level: 4

publish:
  format: "fsdp"                                 
  host: "huggingface"                           
  base_model: "openai-community/gpt2"           
  checkpoint_path: "output/epoch-001-final-ckpt.pth"  
  repo_id: "gplsi/quijote-gpt2-clm"      
  commit_message: "Add Quijote GPT-2 CLM checkpoint"
  max_shard_size: "5GB"
  safe_serialization: true
  create_pr: false
```

2. **Authenticate with Hugging Face**:
- Via CLI: `huggingface-cli login`
- Or set `HUGGINGFACE_HUB_TOKEN` in your environment.

3. **Run the publish task**:

```bash
python src/main.py --config tutorials/configs/publish_tutorial.yaml
```

4. **Verify upload**:
- Check your model repo on https://huggingface.co

## Troubleshooting
- Ensure your token has write access to the target repo.
- Manually create the repo on Hugging Face if you encounter 404 errors.
- See the notebook `tutorials/publish_tutorial.ipynb` for a full example.
