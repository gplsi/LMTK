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

### 🔬 Model Monitoring
- **Training Telemetry** - Gradient/activation histograms in `utils/monitoring.py`
- **Early Warning System** - NaN/overflow detection
- **Optimizer Diagnostics** - Learning rate/parameter scale tracking

## 🚀 Quick Start

# Build environment
```
make build
```

# Validate configuration
```
make validate CONFIG=config/pretraining.yaml
```
# Run tokenization pipeline
```
make tokenize CONFIG=config/tokenization.yaml
```
# Launch distributed pretraining`
```
make train CONFIG=config/pretraining.yaml
```

## ⚙️ Configuration System

**Structured YAML Schemas**:
```
# src/config/base.py
class TaskConfig(BaseModel):
    task_name: str = Field(..., description="Name of task to execute")
    output_dir: Path = Field(..., description="Base output directory")
    
class TokenizationConfig(TaskConfig):
    dataset_path: str
    tokenizer_name: str
    chunk_size: int = 2048
    validation_ratio: float = 0.1
```

Validation flow:
```
CLI Command → Load YAML → Validate Schema → Build Config Object → Execute Task
```

## 📂 Project Structure

```
project/
├── config/                 # YAML configuration templates
│   ├── curricula/         # Data mixing schedules
│   ├── tokenization.yaml  # Tokenizer params
│   └── pretraining.yaml   # Model/training params
├── docker/                # Container definitions
│   └── Dockerfile         # CUDA+PyTorch base image
├── Makefile               # Project orchestration
├── requirements/          # Pinned dependencies
└── src/
    ├── config/            # Pydantic schema definitions
    ├── tasks/             # Task implementations
    └── utils/             # Monitoring/checkpointing
```

## 🐋 Docker & Makefile

**Dockerfile Highlights**:
```
FROM nvcr.io/nvidia/pytorch:23.10-py3
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["make"]
```

**Makefile Targets**:
```
validate:  # Config schema check
    python -m src.main --validate $(CONFIG)

tokenize:  # Process datasets
    python -m src.main --task tokenize $(CONFIG)

train:     # Launch training
    torchrun --nproc_per_node=$(GPUS) src/main.py --task train $(CONFIG)
```

## 📝 Reproducibility Features

1. **Configuration Hashing** - MD5 checksum of all config files
2. **Environment Snapshot** - `pip freeze` in training logs
3. **Deterministic Seeds** - Full random state preservation

## 🤝 Contributing

1. Implement new config schemas in `src/config/`
2. Add corresponding task modules in `src/tasks/`
3. Include validation tests:

```
def test_tokenization_config():
    with pytest.raises(ValidationError):
        TokenizationConfig(tokenizer_name="invalid/model")
```
