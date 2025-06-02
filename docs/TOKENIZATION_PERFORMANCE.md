# Tokenization Performance Guide

## Overview

This guide covers all performance optimizations implemented in the continual pretraining framework's tokenization system. Our tokenizer has been extensively optimized for speed, memory efficiency, and intelligent parallelism handling.

## Performance Optimizations Implemented

### 1. Intelligent Parallelism Strategy

The tokenizer automatically detects whether you're using a fast (Rust-based) or slow (Python-based) tokenizer and applies the optimal parallelism strategy:

#### Fast Tokenizers (Recommended)
- **Detection**: `tokenizer.is_fast == True`
- **Strategy**: Uses internal Rust parallelism (`num_proc=None`)
- **Environment**: `TOKENIZERS_PARALLELISM=true` + `RAYON_RS_NUM_THREADS=32`
- **Performance**: 5-10x faster than slow tokenizers

#### Slow Tokenizers
- **Detection**: `tokenizer.is_fast == False` or missing `is_fast` attribute
- **Strategy**: Python multiprocessing with intelligent defaults
- **Default**: `num_proc = cpu_count() // 2` (leaves cores for system)
- **Configurable**: Via `config.num_proc`

### 2. Optimized Batch Processing

```python
# Default batch sizes (auto-configured)
batch_size = 6000  # Increased from 1000 for better throughput
writer_batch_size = 40000  # Large chunks for efficient I/O (DatasetDict)
writer_batch_size = 10000  # For single datasets
```

### 3. Vectorized Label Generation

Replaced slow nested list comprehension with NumPy vectorized operations:

```python
# OLD (slow)
labels = [[input_id if mask else -100 for input_id, mask in zip(input_ids, attention_mask)] 
          for input_ids, attention_mask in zip(all_input_ids, all_attention_masks)]

# NEW (fast)
labels = np.where(attention_mask == 1, input_ids, -100)
```

**Performance gain**: ~3-5x faster label generation

### 4. Memory Optimizations

- **Fixed-length sequences**: Pre-defined feature schema for better memory allocation
- **Efficient data types**: `int64` for tokens, `int8` for attention masks
- **Keep in memory**: `keep_in_memory=True` for DatasetDict processing
- **Return format**: `return_tensors="np"` for NumPy arrays (faster than lists)

### 5. Environment Variables

Set automatically for optimal performance:

```bash
TOKENIZERS_PARALLELISM=true     # Enable tokenizer-level parallelism
RAYON_RS_NUM_THREADS=32        # Rust thread pool size
```
## Configuration Examples

### Automatic Configuration (Recommended)

```yaml
tokenizer:
  tokenizer_name: "meta-llama/Llama-2-7b-hf"  # Fast tokenizer
  context_length: 4096
  overlap: 256
  batch_size: 6000  # Optional, uses intelligent defaults
  # num_proc: null  # Auto-detected based on tokenizer type
  show_progress: true
```

### Manual Configuration

```yaml
tokenizer:
  tokenizer_name: "some-custom-tokenizer"
  context_length: 2048
  overlap: 128
  batch_size: 4000  # Custom batch size
  num_proc: 8       # Force specific process count (slow tokenizers only)
  show_progress: true
```

### Debug/Single Process Mode

```yaml
tokenizer:
  tokenizer_name: "meta-llama/Llama-2-7b-hf"
  context_length: 1024
  overlap: 64
  batch_size: 100   # Small batches for debugging
  num_proc: 1       # Force single process
  show_progress: true
```

## Performance Monitoring

### Automatic Logging

The tokenizer logs comprehensive performance information:

```bash
# Performance configuration
INFO: Performance configuration: batch_size=6000, parallelism=fast_tokenizer_internal
# or
INFO: Performance configuration: batch_size=4000, num_proc=8, parallelism=python_multiprocessing

# Tokenization progress
INFO: Tokenizing split 'train' with 50000 examples
INFO: Completed tokenizing split 'train' in 125.45 seconds
INFO: Processed 50000 examples at 398.6 examples/sec
```

### Tokenizer Type Detection

```bash
# Fast tokenizer
DEBUG: Fast tokenizer detected - using internal Rust parallelism (num_proc=None)

# Slow tokenizer
DEBUG: Slow tokenizer detected - using Python multiprocessing with 8 processes
WARNING: Using slow tokenizer - consider upgrading to a fast tokenizer for better performance
```

## Benchmarks and Expected Performance

### Fast Tokenizers (e.g., LLaMA, GPT-2, BERT)

| Dataset Size | Context Length | Expected Throughput | Memory Usage |
|-------------|----------------|-------------------|--------------|
| 10K examples | 2048 tokens   | 400-800 ex/sec   | ~2GB RAM     |
| 100K examples | 2048 tokens   | 300-600 ex/sec   | ~8GB RAM     |
| 1M examples | 2048 tokens    | 200-400 ex/sec   | ~15GB RAM    |

### Slow Tokenizers

| Dataset Size | Context Length | Expected Throughput | Memory Usage |
|-------------|----------------|-------------------|--------------|
| 10K examples | 2048 tokens   | 50-150 ex/sec    | ~3GB RAM     |
| 100K examples | 2048 tokens   | 40-120 ex/sec    | ~12GB RAM    |
| 1M examples | 2048 tokens    | 30-100 ex/sec    | ~25GB RAM    |

## Troubleshooting Performance Issues

### 1. Slow Performance

**Check tokenizer type:**
```python
print(f"Fast tokenizer: {tokenizer.is_fast}")
```

**Solutions:**
- Use fast tokenizers when available: `AutoTokenizer.from_pretrained(model_name, use_fast=True)`
- Increase `batch_size` (try 4000-8000)
- For slow tokenizers, increase `num_proc`

### 2. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow performance with high memory usage

**Solutions:**
- Reduce `batch_size` (try 1000-2000)
- Reduce `writer_batch_size`
- For slow tokenizers, reduce `num_proc`
- Remove `keep_in_memory=True` for very large datasets

### 3. CPU Underutilization

**For fast tokenizers:**
- Check `TOKENIZERS_PARALLELISM=true` is set
- Increase `RAYON_RS_NUM_THREADS` if needed
- Monitor with: `htop` or `nvidia-smi`

**For slow tokenizers:**
- Increase `num_proc` up to CPU core count
- Check Python multiprocessing is working

### 4. Configuration Debugging

Add debugging configuration:

```yaml
tokenizer:
  verbose_level: DEBUG  # Enable debug logging
  show_progress: true   # Show progress bars
```

## Migration from Previous Versions

### If you have existing configs:

1. **Remove manual `num_proc` settings** for auto-detection:
   ```yaml
   # OLD
   num_proc: 8
   
   # NEW (recommended)
   # num_proc: null  # Auto-detected
   ```

2. **Update batch sizes** for better performance:
   ```yaml
   # OLD
   batch_size: 1000
   
   # NEW
   batch_size: 6000  # Or remove for auto-defaults
   ```

3. **Check for fast tokenizer warnings** and upgrade if suggested

## Advanced Configuration

### Environment Variables

You can override the automatic environment settings:

```bash
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism
export RAYON_RS_NUM_THREADS=16       # Limit Rust threads
```

### Custom Performance Tuning

For specific hardware configurations:

```yaml
tokenizer:
  batch_size: 8000      # Large batches for high-memory systems
  num_proc: 16          # High core count systems (slow tokenizers)
  show_progress: false  # Disable for non-interactive environments
```

## Best Practices Summary

1. **Use fast tokenizers** whenever possible
2. **Let the system auto-detect** parallelism strategy (`num_proc: null`)
3. **Start with default batch sizes** and adjust based on memory
4. **Monitor performance logs** to verify optimal configuration
5. **Upgrade slow tokenizers** when fast versions become available
6. **Profile your specific dataset** to find optimal settings

## Future Improvements

- Automatic batch size tuning based on available memory
- GPU-accelerated tokenization for supported models
- Streaming tokenization for extremely large datasets
- Advanced memory management for low-resource environments

## Configuration Schema Reference

The updated schema validates:

```yaml
tokenizer:
  context_length: int                    # Required
  overlap: int                          # Optional
  tokenizer_name: str                   # Optional
  verbose_level: str                    # DEBUG|INFO|WARNING|ERROR
  batch_size: int                       # 1-10000, default: auto-detected
  num_proc: int                         # 1-32, default: auto-detected
  show_progress: bool                   # default: true
```

All performance parameters are optional and use intelligent defaults based on your system and tokenizer type.
