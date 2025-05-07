# LM Continual Pretraining Framework

**A Modular Toolkit for Efficient Language Model Pretraining and Adaptation**  
*Streamlining continual pretraining of foundation language models through scalable pipelines and reproducible configurations*

## 🚀 Installation

### Option 1: Using Poetry (Recommended)

```bash
# Install with Poetry
make install-poetry

# Or just use the default install target (defaults to Poetry)
make install
```

### Option 2: Using pip

```bash
# Install with pip in a virtual environment
make install-pip

# Or specify pip as the installation method
make INSTALL_METHOD=pip install
```

### Option 3: Direct pip install

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .  # Regular installation
pip install -e .[dev]  # With development dependencies
```

### Docker Installation

```bash
# Build development image (with Poetry)
make build-dev

# Run development container
make container
```

## 📋 Version Control System

The framework implements a comprehensive version control system using Poetry to manage semantic versioning. This ensures consistent versioning across all components and helps track changes systematically.

### Version Information

You can view the current version of the framework using:

```bash
# Show the current version
make version-show

# Display detailed version info including dependencies
python -m src.main --version
```

### Version Management Workflow

When making changes to the codebase, follow this workflow to manage versions properly:

1. **Make your changes** to the codebase
2. **Decide on the version increment** based on semantic versioning principles:
   - `major`: Breaking changes to the API
   - `minor`: New features, backward compatible
   - `patch`: Bug fixes, backward compatible

3. **Bump the version**:
```bash
# For a patch update (e.g., 0.1.0 → 0.1.1)
make version-bump

# For a minor update (e.g., 0.1.0 → 0.2.0)
make version-bump VERSION_BUMP=minor

# For a major update (e.g., 0.1.0 → 1.0.0)
make version-bump VERSION_BUMP=major
```

4. **Update the changelog** to document your changes:
```bash
make changelog
```
When editing the changelog, add your changes under the `[Unreleased]` section using appropriate categories (Added, Changed, Fixed, etc.).

5. **Finalize the release**:
```bash
make version-release
```
This command will:
- Update the changelog format
- Create a git tag for the version
- Commit the changes

6. **Push to the repository**:
```bash
git push && git push --tags
```

### Checking Version in Code

The framework provides utilities for checking version information in your code:

```python
from src.utils.version import get_version, display_version_info

# Get current version string
version = get_version()

# Display detailed version information
display_version_info()
```

### Versioning Strategy

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (0.x.x → 1.0.0): Incompatible API changes
- **MINOR** version (0.1.x → 0.2.0): New functionality in a backward-compatible manner
- **PATCH** version (0.1.0 → 0.1.1): Backward-compatible bug fixes

During development phase (before 1.0.0), minor version bumps may include breaking changes.

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

## 📚 Documentation

The project includes comprehensive documentation built with Sphinx featuring a beautiful custom design.

### Building Documentation

We provide a convenient script to build the documentation:

```bash
# Build documentation
./build_docs.sh

# Build and open in browser
./build_docs.sh --open

# More options
./build_docs.sh --help
```

Alternatively, you can use tox:

```bash
# Using tox
tox -e docs
```

For more details on documentation features, customization, and contributing guidelines, please refer to the [Documentation README](/docs/README.md).

## 🤝 Contributing

1. Implement new config schemas in `src/config/`
2. Add corresponding task modules in `src/tasks/`
3. Include validation tests:

```
def test_tokenization_config():
    with pytest.raises(ValidationError):
        TokenizationConfig(tokenizer_name="invalid/model")
```
````
