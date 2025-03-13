# Configuration Guide

## Overview

The framework uses a hierarchical YAML-based configuration system with schema validation. Configurations are validated against predefined schemas to ensure type safety and completeness.

## Schema System

### Base Schema

All task configurations extend from `config/schemas/base.schema.yaml` which defines common fields:

```yaml
type: object
required:
  - task
  - output_dir
properties:
  task:
    type: string
    description: Task identifier
  output_dir:
    type: string
    description: Output directory for task artifacts
  verbose:
    type: string
    enum: [DEBUG, INFO, WARNING, ERROR]
    default: INFO
```

### Task-Specific Schemas

#### Pretraining Schema
Located in `config/schemas/pretraining.schema.yaml`:

```yaml
allOf:
  - $ref: base.schema.yaml
  - type: object
    required:
      - model
      - training
    properties:
      model:
        type: object
        required:
          - name
          - config
        properties:
          name:
            type: string
          config:
            type: object
      training:
        type: object
        required:
          - batch_size
          - learning_rate
        properties:
          batch_size:
            type: integer
          learning_rate:
            type: number
```

#### Tokenization Schema
Located in `config/schemas/tokenization.schema.yaml`:

```yaml
allOf:
  - $ref: base.schema.yaml
  - type: object
    required:
      - tokenizer
      - dataset
    properties:
      tokenizer:
        type: object
        required:
          - name
          - context_length
        properties:
          name:
            type: string
          context_length:
            type: integer
      dataset:
        type: object
        required:
          - source
          - format
```

## Configuration Files

### Directory Structure

```
config/
├── experiments/           # Task configurations
│   ├── pretraining.yaml
│   └── tokenization.yaml
├── schemas/              # JSON schemas
│   ├── base.schema.yaml
│   └── task.schema.yaml
└── test_grids/          # Test configurations
    └── minimal_test_grid.yaml
```

### Example Configurations

#### Pretraining Configuration
```yaml
# config/experiments/pretraining.yaml
task: pretraining
output_dir: outputs/gpt2_pretrain
model:
  name: gpt2
  config:
    vocab_size: 50257
    n_positions: 1024
training:
  batch_size: 32
  learning_rate: 1e-4
  distributed:
    strategy: fsdp
    params:
      sharding_strategy: 1
dataset:
  source: local
  format: jsonl
```

#### Tokenization Configuration
```yaml
# config/experiments/tokenization.yaml
task: tokenization
output_dir: outputs/tokenized
tokenizer:
  name: gpt2
  context_length: 1024
dataset:
  source: local
  format: txt
  paths:
    - data/raw/corpus.txt
```

## Validation System

### Configuration Loading

The framework validates configurations in two steps:

1. **Schema Validation**: Ensures configuration matches JSON schema
2. **Runtime Validation**: Task-specific validation in orchestrator

Example:
```python
from src.config import ConfigValidator

# Load and validate configuration
validator = ConfigValidator()
config = validator.validate("config/experiments/pretraining.yaml", "pretraining")

# Task-specific validation
orchestrator = PretrainingOrchestrator(config)
orchestrator.validate_config()
```

### Error Handling

Configuration errors are caught early with descriptive messages:

```
ValidationError: Configuration validation failed:
- training.batch_size: required field
- model.config.vocab_size: must be integer
```

## Advanced Features

### Environment Variables

Use environment variables in configurations:

```yaml
output_dir: ${OUTPUT_DIR:-outputs/default}
dataset:
  path: ${DATA_PATH}
```

### Include Files

Break down large configurations:

```yaml
# main.yaml
includes:
  - model.yaml
  - training.yaml
task: pretraining
```

### Grid Search

Define parameter grids for experiments:

```yaml
# grid.yaml
grid:
  learning_rate: [1e-4, 1e-5]
  batch_size: [16, 32]
base_config: base.yaml
```

## Best Practices

1. **Version Control**
   - Keep configurations in version control
   - Document changes in configuration files
   - Use explicit versions for dependencies

2. **Organization**
   - Group related parameters logically
   - Use consistent naming conventions
   - Keep configurations DRY

3. **Documentation**
   - Comment non-obvious parameter choices
   - Include example configurations
   - Document default values

4. **Validation**
   - Define strict schemas
   - Validate early in pipeline
   - Provide clear error messages