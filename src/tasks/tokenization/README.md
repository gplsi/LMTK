# Tokenization Task

This module handles text preprocessing and tokenization for language model training, supporting various input formats and tokenizer configurations.

## Overview

The tokenization task prepares raw text data for language model training by:
- Converting text into token sequences
- Handling overlapping contexts
- Managing different input formats
- Saving tokenized datasets efficiently

## Configuration

Example configuration:
```yaml
task: tokenization
output_dir: data/tokenized/output
tokenizer:
  name: gpt2  # Any HuggingFace tokenizer
  context_length: 1024
  overlap: 64  # Context overlap length
  task: causal_pretraining

# Dataset configuration
dataset:
  source: local
  nameOrPath: data/raw/corpus
  format: files  # Options: files, jsonl, csv
  file_config:
    format: txt
    
# Optional test split
test_size: 0.1  # 10% test split
```

## Features

- **Flexible Input Formats**: Support for TXT, JSON, CSV, and other formats
- **Efficient Processing**: Streaming tokenization for memory efficiency
- **Context Management**: Configurable context length and overlap
- **Dataset Validation**: Statistical checks on tokenized outputs
- **HuggingFace Integration**: Compatible with all HF tokenizers
- **Data Splitting**: Optional train/test split generation

## Usage

1. Prepare your configuration file following `config/schemas/tokenization.schema.yaml`
2. Run the tokenization task:
```bash
python src/main.py config/experiments/your_tokenizer_config.yaml
```

## Output Format

The tokenized dataset is saved in the HuggingFace datasets format with:
- `input_ids`: Token IDs
- `attention_mask`: Attention masks for padded sequences
- Optional test split if configured

## Advanced Configuration

### Custom Input Processing
Handle specific file formats:
```yaml
dataset:
  format: jsonl
  file_config:
    text_key: content  # Key containing text in JSON
    filter_empty: true
```

### Multi-File Processing
Process multiple input sources:
```yaml
dataset:
  format: files
  file_config:
    paths:
      - data/raw/corpus1
      - data/raw/corpus2
    recursive: true  # Search subdirectories
```

### Tokenizer Options
Fine-tune tokenizer behavior:
```yaml
tokenizer:
  add_special_tokens: true
  padding: max_length
  truncation: true
  return_overflowing_tokens: false
```

## Best Practices

1. **Memory Management**:
   - Use streaming for large datasets
   - Set appropriate batch sizes for processing

2. **Data Quality**:
   - Set `test_size` for validation
   - Enable statistical checks

3. **Performance**:
   - Use appropriate context overlap
   - Enable multiprocessing when available