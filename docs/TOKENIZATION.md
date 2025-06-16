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
tokenizer:
  name: "openai-community/gpt2"
  add_special_tokens: true
  padding: "max_length"
  truncation: true
  max_length: 512
  save_directory: "tutorials/output/tokenizer"

dataset:
  nameOrPath: "tutorials/data/raw_text_data/quijote.txt"
  format: "files"
  test_size: 0.1
  shuffle: true
  text_column: "text"
  output_dir: "tutorials/output/tokenized_dataset"

output:
  save_format: "hf"
  push_to_hub: false
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
