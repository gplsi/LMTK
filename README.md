# LM Continual Pretraining Framework

**A Modular Toolkit for Efficient Language Model Pretraining and Adaptation**  
*Streamlining continual pretraining of foundation language models through scalable pipelines and reproducible configurations*

## 🌟 Key Features

### 🔧 Core Infrastructure
- **Configuration System** - Type-safe YAML schemas with Pydantic validation
- **Task Orchestration** - Modular task execution via CLI/config mapping
- **Environment Management** - Dockerized training stack with Makefile control

### 🛠️ Training Capabilities
- **Resumable Workflows** - Atomic checkpoints with full state serialization
- **Distributed Strategies** - FSDP/DeepSpeed integration profiles
- **Curriculum Learning** - Phase-wise data mixing (`config/curricula`)

### 📊 Data Processing
- **Multi-format Corpus** - Unified processor for JSONL/Parquet/TXT
- **Streaming Tokenization** - Memory-efficient HF datasets integration
- **Data Health Checks** - Statistical validation pre-training

## 🚀 Quick Start

```bash
# Build environment
make build

# Validate configuration
make validate CONFIG=config/pretraining.yaml

# Run tokenization pipeline
make tokenize CONFIG=config/tokenization.yaml

# Launch distributed pretraining
make train CONFIG=config/pretraining.yaml
```

## 📚 Documentation

- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Guide for implementing new tasks
- [Testing Documentation](tests/README.md) - Testing framework and guidelines
- [Configuration Guide](docs/CONFIGURATION.md) - Schema system and validation

## 🏗️ Architecture

### Task System
```
src/tasks/
├── pretraining/         # Continual pretraining implementation
│   ├── orchestrator.py  # Training workflow coordination
│   └── fabric/         # PyTorch Lightning Fabric integration
│       ├── base.py     # Base trainer implementation
│       └── distributed/ # Distributed strategies
├── tokenization/        # Tokenization pipeline
│   ├── orchestrator.py  # Tokenization workflow
│   └── tokenizer/      # Tokenizer implementations
└── utils/              # Shared utilities
```

### Distributed Training Strategies

1. **FSDP (Fully Sharded Data Parallel)**
   - Memory-efficient large model training
   - Automatic sharding and optimization
   - Configurable via `config/pretraining.fsdp.yaml`

2. **DeepSpeed**
   - ZeRO optimization stages
   - Mixed precision training
   - Configurable via `config/pretraining.deepspeed.yaml`

3. **DDP (DistributedDataParallel)**
   - Classic PyTorch distribution
   - Multi-node support
   - Configurable via `config/pretraining.ddp.yaml`

### Configuration System

Hierarchical configuration with validation:
```yaml
task: pretraining
output_dir: outputs/example
model:
  name: gpt2
  config:
    vocab_size: 50257
training:
  batch_size: 32
  learning_rate: 1e-4
  distributed:
    strategy: fsdp
    params:
      sharding_strategy: 1
```

## 🤝 Contributing

1. Read the [Developer Guide](docs/DEVELOPER_GUIDE.md)
2. Follow the coding standards and patterns
3. Add tests following [Testing Documentation](tests/README.md)
4. Submit PR with description and test results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Lightning team for the Fabric library
- Hugging Face for transformers and datasets
- DeepSpeed team for optimization strategies
