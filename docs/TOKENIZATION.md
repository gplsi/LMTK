# Tokenization Task Guide

## Overview

This guide explains the tokenization task in LMTK, covering configuration, execution, and best practices for preparing datasets for language model pretraining or finetuning.

## What is the Tokenization Task?
The tokenization task converts raw text data into tokenized datasets compatible with Hugging Face Transformers models. It supports both fast (Rust) and slow (Python) tokenizers, automatic batching, parallelism, and memory optimization.

## Key Features
- **YAML-Driven**: All options are specified in a config file for reproducibility.
- **Automatic Parallelism**: Detects and optimizes for tokenizer type and available CPU cores.
- **Batch and Memory Optimization**: Smart defaults for batch size and writer chunking.
- **Flexible Output**: Save as Hugging Face `DatasetDict`, push to Hub, or export to disk.

## Example Workflow

1. **Prepare your config** (see `tutorials/configs/tokenization_config.yaml`):

```yaml
task: "tokenization"
experiment_name: "quijote_gpt2_tokenization"
verbose_level: 2

tokenizer:
  tokenizer_name: "openai-community/gpt2"
  context_length: 1024
  overlap: 256
  task: "clm_training"
  batch_size: 1024
  num_proc: 2
  show_progress: true

dataset:
  source: "local"
  nameOrPath: "/workspace/data/testing"
  format: "files"
  use_txt_as_samples: true
  file_config:
    format: "txt"
    encoding: "utf-8"

output:
  path: "data/tokenized/quijote"

test_size: 0
```

2. **Run the tokenization task**:

```bash
python src/main.py --config tutorials/configs/tokenization_config.yaml
```

3. **Inspect the output**:
- The tokenized dataset will be saved to the specified output directory.
- Use `datasets.load_from_disk()` to load and inspect the result.

## Tips & Best Practices
- Use fast tokenizers for best performance (see `TOKENIZATION_PERFORMANCE.md`).
- Let the framework auto-detect batch size and `num_proc` unless you have special requirements.
- Always validate your config before launching jobs.
- For large datasets, ensure sufficient disk space for output.

## References
- See `tutorials/tokenization_tutorial.ipynb` for a full notebook walkthrough.
- See the main README and `TOKENIZATION_INDEX.md` for additional details and troubleshooting.
