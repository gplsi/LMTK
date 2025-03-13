<<<<<<< HEAD
<<<<<<< HEAD
# Testing Documentation

## Overview

This document describes the testing framework and guidelines for the LM Continual Pretraining Framework.

## Test Structure

```
tests/
├── conftest.py           # Global test fixtures and configuration
├── fixtures/            
│   ├── configs.py       # Configuration fixtures
│   └── data_fixtures.py # Dataset fixtures
├── integration/         # End-to-end workflow tests
│   ├── test_pretraining_pipeline.py
│   └── test_tokenization_pipeline.py
└── unit/               # Unit tests for individual components
=======
# Continual Pretraining Framework Testing Guide
=======
# Testing Documentation
>>>>>>> a51282d (Add comprehensive configuration and developer guides with schema validation details)

## Overview

This document describes the testing framework and guidelines for the LM Continual Pretraining Framework.

## Test Structure

```
tests/
├── conftest.py           # Global test fixtures and configuration
├── fixtures/            
│   ├── configs.py       # Configuration fixtures
│   └── data_fixtures.py # Dataset fixtures
├── integration/         # End-to-end workflow tests
│   ├── test_pretraining_pipeline.py
│   └── test_tokenization_pipeline.py
<<<<<<< HEAD
└── unit/                       # Unit tests for individual components
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)
=======
└── unit/               # Unit tests for individual components
>>>>>>> a51282d (Add comprehensive configuration and developer guides with schema validation details)
    ├── test_config.py
    ├── test_dataset.py
    ├── test_fabric.py
    ├── test_model_configurations.py
    ├── test_pretraining_orchestrator.py
    ├── test_tokenization.py
    └── test_utils.py
```

<<<<<<< HEAD
<<<<<<< HEAD
## Running Tests

### Quick Start
=======
## Test Categories

1. **Unit Tests**: Tests for individual components in isolation
   - Configuration validation
   - Dataset handling
   - Tokenization
   - Model configuration
   - Utility functions

2. **Integration Tests**: Tests for component interactions
   - Tokenization pipeline
   - Pretraining pipeline with different strategies

3. **GPU-Specific Tests**: Tests requiring GPU hardware
   - Distributed training (FSDP, DDP)
   - Mixed precision operations
   - Multi-GPU synchronization

## Running Tests

### Basic Test Commands
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)
=======
## Running Tests

### Quick Start
>>>>>>> a51282d (Add comprehensive configuration and developer guides with schema validation details)

```bash
# Run all tests
make test

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Run specific test suite
make test SUITE=unit
make test SUITE=integration

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
make test-coverage
```

### Test Configuration

Test configurations are managed through:
- `pytest.ini` - Global pytest settings
- `tests/conftest.py` - Shared fixtures
- `config/test_grids/` - Parameterized test configurations

## Writing Tests

### Test Categories

1. **Unit Tests**: Test individual components in isolation
   - Location: `tests/unit/`
   - Naming: `test_*.py`
   - Use mocking for external dependencies

2. **Integration Tests**: Test complete workflows
   - Location: `tests/integration/`
   - Test full pipelines (tokenization, pretraining)
   - Use minimal configurations from `config/test_grids/`

### Best Practices

1. **Use Fixtures**
   ```python
   def test_tokenization(tokenizer_config, sample_dataset):
       orchestrator = TokenizationOrchestrator(tokenizer_config)
       result = orchestrator.tokenize_dataset(sample_dataset)
       assert result is not None
   ```

2. **Mock Heavy Dependencies**
   ```python
   @patch("torch.cuda.is_available")
   def test_gpu_detection(mock_cuda):
       mock_cuda.return_value = False
       orchestrator = ContinualOrchestrator(config)
       assert orchestrator.devices == "cpu"
   ```

3. **Test Configuration Validation**
   ```python
   def test_invalid_config():
       with pytest.raises(ValueError):
           TokenizationOrchestrator(invalid_config)
   ```

### Creating New Test Suites

1. Add test file in appropriate directory
2. Create corresponding fixtures if needed
3. Add to test grid if running parameterized tests

## Test Grid System

The framework uses test grids for parameterized testing:

```yaml
# config/test_grids/minimal_test_grid.yaml
tokenization:
  - name: "basic"
    tokenizer: "gpt2"
    dataset: "sample"
  - name: "advanced"
    tokenizer: "custom"
    dataset: "large"
```

Run parameterized tests:
```bash
python scripts/run_parameterized_tests.py --grid minimal_test_grid.yaml
=======
# Run only unit tests
=======
# Run only unit tests (CPU)
>>>>>>> 05c94bb (Add GPU test and performance benchmark configuration files)
make test-unit
=======
# Run specific test suite
make test SUITE=unit
make test SUITE=integration
>>>>>>> a51282d (Add comprehensive configuration and developer guides with schema validation details)

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
make test-coverage
```

### Test Configuration

Test configurations are managed through:
- `pytest.ini` - Global pytest settings
- `tests/conftest.py` - Shared fixtures
- `config/test_grids/` - Parameterized test configurations

## Writing Tests

### Test Categories

1. **Unit Tests**: Test individual components in isolation
   - Location: `tests/unit/`
   - Naming: `test_*.py`
   - Use mocking for external dependencies

2. **Integration Tests**: Test complete workflows
   - Location: `tests/integration/`
   - Test full pipelines (tokenization, pretraining)
   - Use minimal configurations from `config/test_grids/`

### Best Practices

1. **Use Fixtures**
   ```python
   def test_tokenization(tokenizer_config, sample_dataset):
       orchestrator = TokenizationOrchestrator(tokenizer_config)
       result = orchestrator.tokenize_dataset(sample_dataset)
       assert result is not None
   ```

2. **Mock Heavy Dependencies**
   ```python
   @patch("torch.cuda.is_available")
   def test_gpu_detection(mock_cuda):
       mock_cuda.return_value = False
       orchestrator = ContinualOrchestrator(config)
       assert orchestrator.devices == "cpu"
   ```

3. **Test Configuration Validation**
   ```python
   def test_invalid_config():
       with pytest.raises(ValueError):
           TokenizationOrchestrator(invalid_config)
   ```

### Creating New Test Suites

1. Add test file in appropriate directory
2. Create corresponding fixtures if needed
3. Add to test grid if running parameterized tests

## Test Grid System

The framework uses test grids for parameterized testing:

```yaml
# config/test_grids/minimal_test_grid.yaml
tokenization:
  - name: "basic"
    tokenizer: "gpt2"
    dataset: "sample"
  - name: "advanced"
    tokenizer: "custom"
    dataset: "large"
```

Run parameterized tests:
```bash
<<<<<<< HEAD
# Run GPU-specific tests
make test-gpu

# Run distributed training tests with real GPUs
make test-distributed-gpu
```

Requirements for GPU testing:
- NVIDIA GPU with CUDA support
- NVIDIA Container Runtime installed
- Docker with GPU support configured
- Appropriate drivers and CUDA toolkit

### Parameterized Testing

For testing multiple configuration combinations:

```bash
# Run tests with a specific parameter grid
make test-grid GRID=config/test_grids/minimal_test_grid.yaml

# Run the minimal test grid (CPU only)
make test-minimal

# Run the comprehensive test grid (CPU only)
make test-comprehensive
```

### Writing GPU Tests

When writing tests that require GPU:

```python
from tests.conftest import requires_gpu

@requires_gpu
def test_gpu_feature():
    # This test will only run when GPUs are available
    pass
```

The test environment will:
1. Skip GPU tests when running in CI or CPU-only mode
2. Run GPU tests in a proper containerized environment with GPU access
3. Use real GPU operations for distributed training tests

## Continuous Integration

Our testing strategy separates tests into two categories:

1. **CI/CD Pipeline Tests** - Automatically run on GitHub:
   - CPU-only tests that can be safely mocked
   - Unit tests for individual components
   - Integration tests with appropriate mocks
   - Configuration validation tests 
   - No GPU requirements or hardware acceleration

2. **Manual GPU Tests** - Triggered by users with GPU access:
   - Tests that genuinely require GPU hardware
   - Distributed training tests (FSDP, DDP)
   - Performance benchmarks
   - Multi-GPU scaling tests

### Running CI Tests

These tests run automatically in GitHub Actions on every push and PR:

```bash
# Run all CI-compatible tests (CPU only)
make test

# Run specific CI test categories
make test-unit
make test-integration
```

### Running Manual GPU Tests 

These tests must be manually triggered by users with appropriate hardware:

```bash
# Basic GPU tests
make test-gpu

# Distributed training tests
make test-distributed-gpu

# Performance benchmark tests
make test-perf GPUS=2  # Specify number of GPUs
```

### GitHub Actions Workflows

Two workflow files are provided:

1. **ci.yml** - Automatic CI pipeline for CPU-only tests
2. **gpu-tests.yml** - Manual workflow for GPU tests (requires self-hosted runner with GPU)

To set up a self-hosted GPU runner:
1. Configure a machine with GPU(s)
2. Install GitHub Actions runner software
3. Register it as a self-hosted runner in your repository
4. Update the `runs-on` field in gpu-tests.yml to target your runner

## Creating Mock Data

For testing with mock data:

```python
from tests.fixtures.data_fixtures import create_mock_text_data, create_mock_tokenized_dataset

# Create mock text files for tokenization tests
data_dir = create_mock_text_data()

# Create mock tokenized dataset for pretraining tests
dataset_dir = create_mock_tokenized_dataset()
```

## Best Practices

<<<<<<< HEAD
```python
def test_example(mock_text_data, mock_tokenized_dataset):
    # Test using the mock data...
    pass
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)
```
=======
1. **Test Environment**:
   - CPU tests: Use CI and local development
   - GPU tests: Use dedicated test environment with proper hardware
   - Distributed tests: Use multi-GPU setup in containerized environment

2. **Test Organization**:
   - Mark GPU tests with @requires_gpu
   - Keep CPU and GPU tests separate
   - Use appropriate fixtures for each environment

3. **Test Data**:
   - Use small datasets for GPU tests
   - Mock heavy operations in CPU tests
   - Use real data only when necessary

4. **Performance Testing**:
   - Run performance tests only in GPU environment
   - Use consistent hardware for benchmarking
   - Document hardware requirements
>>>>>>> 05c94bb (Add GPU test and performance benchmark configuration files)
=======
python scripts/run_parameterized_tests.py --grid minimal_test_grid.yaml
```
>>>>>>> a51282d (Add comprehensive configuration and developer guides with schema validation details)
