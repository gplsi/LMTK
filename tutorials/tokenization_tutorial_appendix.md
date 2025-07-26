# Appendix: Running the Tokenization Tutorial with YAML Configuration

## Complete YAML Configuration

Below is a complete YAML configuration file for the tokenization module that follows the schema defined in the framework. You can save this to a file (e.g., `tokenization_config.yaml`) and use it to run the tokenization task:

```yaml
task: tokenization
experiment_name: tutorial_tokenization
verbose_level: 4

# Tokenization configuration
tokenization:
  # Tokenizer configuration
  tokenizer:
    name: gpt2
    use_fast: true
    
  # Input data configuration
  input:
    source: local
    path: /path/to/your/raw_text_data
    format: text
    
  # Output configuration
  output:
    path: /path/to/output_tokenized_dataset
    save_format: arrow
    
  # Processing configuration
  processing:
    chunk_size: 1000
    max_length: 1024
    num_proc: 4
    batch_size: 1000
    
  # Text preprocessing configuration
  preprocessing:
    lowercase: false
    strip_accents: false
    add_special_tokens: true
```

Make sure to replace the following placeholders with your actual values:
- `/path/to/your/raw_text_data`: The path to your raw text data
- `/path/to/output_tokenized_dataset`: The directory where the tokenized dataset will be saved

## Running the Tokenization Task

To run the tokenization task using the YAML configuration:

```bash
# Navigate to the root directory of the framework
cd /path/to/continual-pretraining-framework

# Run the tokenization with the configuration file
python src/main.py /path/to/tokenization_config.yaml
```

## Integration with CLM Training

The tokenized dataset produced by the tokenization module can be directly used for CLM training:

```yaml
# In your CLM training configuration
dataset:
  source: local
  nameOrPath: /path/to/output_tokenized_dataset  # Path to your tokenized dataset output
  streaming: false
  shuffle: true
```

## Important Notes

1. **Input Formats**: The tokenization module supports various input formats including 'text', 'json', 'csv', and 'parquet'. Specify the appropriate format in the `input.format` parameter.

2. **Tokenizer Selection**: The `tokenizer.name` parameter can be any HuggingFace tokenizer name or a path to a local tokenizer.

3. **Fast Tokenizers**: Setting `tokenizer.use_fast` to `true` is recommended for better performance, but some tokenizers may not have a fast implementation.

4. **Parallelism**: Adjust the `processing.num_proc` parameter based on your CPU cores for optimal performance.

5. **Memory Considerations**: For large datasets, you may need to adjust the `processing.chunk_size` and `processing.batch_size` parameters to manage memory usage.

6. **Output Format**: The `output.save_format` parameter can be 'arrow', 'parquet', or 'json'. Arrow format is recommended for best performance with HuggingFace datasets.

7. **Preprocessing Options**: The preprocessing options allow you to customize how the text is processed before tokenization. Adjust these based on your specific requirements.
