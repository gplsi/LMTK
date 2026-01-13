# LMTK
The Language Model ToolKit

**A Modular Toolkit for Efficient Language Model Pretraining and Adaptation**  
*Streamlining continual pretraining of foundation language models through scalable pipelines and reproducible configurations*

> **Note:** Docker support has been discontinued. The project now uses Conda for environment management and job launching.

## 🚀 Installation


### Setting up Conda Enviroment (Recommended)

```bash
# Create a new conda environment
conda create -n lmtk python=3.10 -y
conda activate lmtk

# Install the package
pip install -e .  # Regular installation
```

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
make build

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

## 🌟 Major Tasks

### 🧩 Tokenization
- Efficiently preprocess and tokenize large text corpora using YAML-driven configs.
- Supports fast and slow tokenizers, parallelism, and memory-optimized workflows.
- See [`TOKENIZATION_INDEX.md`](docs/TOKENIZATION_INDEX.md) and [`TOKENIZATION_PERFORMANCE.md`](docs/TOKENIZATION_PERFORMANCE.md).

### 🔁 Training
- Further train language models on new data using scalable, resumable pipelines.
- Distributed training, curriculum support, and robust checkpointing.
- See [`CLM_TRAINING.md`](docs/CLM_TRAINING.md).

### 🚀 Publish
- Upload trained models, tokenizers, or datasets to the Hugging Face Hub.
- YAML-driven, supports safe serialization and authentication best practices.
- See [`PUBLISH.md`](docs/PUBLISH.md).

## 📚 Documentation by Task
- [Tokenization Guide](docs/TOKENIZATION.md)
- [Training Guide](docs/CLM_TRAINING.md)
- [Publish Guide](docs/PUBLISH.md)

---

### 🔧 Core Infrastructure
- **Configuration System** - Type-safe YAML schemas with Pydantic validation
- **Task Orchestration** - Modular task execution via CLI/config mapping

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

All major tasks are YAML-driven. See the `/docs` folder for detailed per-task guides and example configs.

# Build environment
```
make build
```

# Validate configuration
```
make validate CONFIG=config/pretraining.yaml
```
# Run tokenization task (YAML-driven)
```
python src/main.py --config tutorials/configs/tokenization_tutorial.yaml
```
# Launch CLM training (continual pretraining)
```
python src/main.py --config tutorials/configs/clm_training_tutorial.yaml
```
# Publish a trained model to Hugging Face Hub
```
python src/main.py --config tutorials/configs/publish_tutorial.yaml
```

**Note:** For publish tasks, authenticate with Hugging Face via `huggingface-cli login` or set the `HUGGINGFACE_HUB_TOKEN` environment variable.

## ✅ Testing

Local tests can be run when dependencies are available:

```bash
python -m pytest -q
```

Some tests require the SLURM container runtime. In that case, prefer a SLURM-based test run and record the job ID and log paths in the issue card or ExecPlan. When available, use `slurm/tests/slurm_test.env` and `slurm/tests/run_tests.sh` to submit tests to the configured partition.

Test organization:
- Task-specific tests live under `src/tasks/<task>/`.
- Cross-cutting or integration tests live under `tests/`.

## 🖥️ SLURM Cluster Execution

For running on SLURM clusters, we provide a comprehensive set of configurable job scripts in the `slurm/` directory:

```bash
# Quick submission with defaults
cd slurm
sbatch p1-dgx.slurm

# Use the helper script for easy configuration
./submit_job.sh -c config/experiments/test_continual.yaml -k your_wandb_key -g 4

# Custom resource allocation
./submit_job.sh -g 8 -m 400G -t 72:00:00 -o large_experiment
```

**Key Features:**
- 🔧 **Fully Configurable** - All paths, resources, and settings via environment variables
- 📊 **WandB Integration** - Automatic experiment tracking and logging
- 🔍 **Debug Support** - Comprehensive error reporting and troubleshooting
- 📁 **Organized Structure** - Clean separation of job scripts and configurations

See [`slurm/README.md`](slurm/README.md) for detailed documentation and examples.

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
