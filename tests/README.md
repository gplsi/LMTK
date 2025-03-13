# Continual Pretraining Framework Testing Guide

This document provides information on the testing infrastructure for the Continual Pretraining Framework.

## Testing Philosophy

Our testing approach follows these principles:
1. **Comprehensive Coverage**: Test all critical components and their interactions
2. **Configuration Combinations**: Test various configuration combinations to ensure robustness
3. **Mock Data**: Use small, mock datasets to keep tests fast and reliable
4. **Proper Isolation**: Each test should be isolated and not depend on other tests
5. **Environment-Specific Testing**: GPU tests run only in appropriate environments

## Test Directory Structure

```
tests/
├── conftest.py                 # Pytest fixtures and configuration
├── fixtures/                   # Test fixtures for data and configs 
│   ├── configs.py              # Mock configuration objects
│   └── data_fixtures.py        # Mock data generation
├── integration/                # Integration tests
│   ├── test_pretraining_pipeline.py
│   └── test_tokenization_pipeline.py
└── unit/                       # Unit tests for individual components
    ├── test_config.py
    ├── test_dataset.py
    ├── test_fabric.py
    ├── test_model_configurations.py
    ├── test_pretraining_orchestrator.py
    ├── test_tokenization.py
    └── test_utils.py
```

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

```bash
# Run CPU-only tests
make test

# Run only unit tests (CPU)
make test-unit

# Run only integration tests (CPU)
make test-integration

# Run all tests with coverage report (CPU)
make test-all
```

### GPU Test Environment

GPU tests are run in a controlled environment using Docker with NVIDIA container runtime:

```bash
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

Tests are separated into two workflows:
1. **CPU Tests**: Run on every PR and push
   - Unit tests
   - Integration tests
   - Basic functionality
   - Configuration validation

2. **GPU Tests**: Run in a controlled environment
   - Distributed training tests
   - Performance tests
   - Multi-GPU synchronization tests
   - Mixed precision operations

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