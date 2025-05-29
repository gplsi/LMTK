# Fast vs Slow Tokenizers: Parallelism Strategy

## Overview

The tokenization implementation now intelligently handles parallelism based on whether you're using a fast (Rust-based) or slow (Python-based) tokenizer.

## Fast Tokenizers (Recommended)

**What they are:**
- Written in Rust using the 🤗 Tokenizers library
- Available for most modern models (BERT, GPT-2, LLaMA, etc.)
- Identified by `tokenizer.is_fast == True`

**Parallelism strategy:**
- Uses internal Rust-based parallelism
- `num_proc=None` in `dataset.map()` 
- `TOKENIZERS_PARALLELISM=true` environment variable enables tokenizer-level parallelism
- **Much faster than Python multiprocessing**

**Performance benefits:**
- 5-10x faster than slow tokenizers
- Better memory efficiency
- Automatic parallelism without Python GIL limitations

## Slow Tokenizers

**What they are:**
- Pure Python implementations
- Legacy tokenizers or custom implementations
- Identified by `tokenizer.is_fast == False` or missing `is_fast` attribute

**Parallelism strategy:**
- Uses Python multiprocessing via `num_proc` parameter
- Default: `num_proc = cpu_count() // 2` (leaves cores for system)
- Can be configured via `config.num_proc`

## Configuration Examples

### Automatic (Recommended)
```yaml
tokenizer:
  tokenizer_name: "meta-llama/Llama-2-7b-hf"  # Fast tokenizer
  batch_size: 2000
  # num_proc: null  # Auto-detected based on tokenizer type
```

### Force Specific Configuration
```yaml
tokenizer:
  tokenizer_name: "some-slow-tokenizer"
  batch_size: 1000
  num_proc: 4  # Only used for slow tokenizers
```

### Single Process (Debugging)
```yaml
tokenizer:
  tokenizer_name: "meta-llama/Llama-2-7b-hf"
  batch_size: 100
  num_proc: 1  # Forces single process even for slow tokenizers
```

## Performance Logging

The system logs the chosen parallelism strategy:

```
INFO: Performance configuration: batch_size=2000, parallelism=fast_tokenizer_internal
```

or 

```
INFO: Performance configuration: batch_size=2000, num_proc=4, parallelism=python_multiprocessing
```

## Migration Guide

If you have existing configs with `num_proc` settings:

1. **Fast tokenizers**: `num_proc` will be ignored (logged for transparency)
2. **Slow tokenizers**: `num_proc` will be used as configured
3. **Auto-detection**: Set `num_proc: null` to use intelligent defaults

## Troubleshooting

### "Using slow tokenizer" warning
- Consider upgrading to a fast tokenizer for your model
- Check if a fast version exists: `AutoTokenizer.from_pretrained(model_name, use_fast=True)`

### Performance not improving
- Verify you're using a fast tokenizer with `.is_fast == True`
- Check that `TOKENIZERS_PARALLELISM=true` is set
- Monitor CPU usage during tokenization

### Memory issues
- Reduce `batch_size` for large models or limited memory
- For slow tokenizers, reduce `num_proc` if memory usage is high
