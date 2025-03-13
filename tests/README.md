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

This document provides information on the testing infrastructure for the Continual Pretraining Framework.

## Testing Philosophy

Our testing approach follows these principles:
1. **Comprehensive Coverage**: Test all critical components and their interactions
2. **Configuration Combinations**: Test various configuration combinations to ensure robustness
3. **Mock Data**: Use small, mock datasets to keep tests fast and reliable
4. **Proper Isolation**: Each test should be isolated and not depend on other tests

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
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)
    ├── test_config.py
    ├── test_dataset.py
    ├── test_fabric.py
    ├── test_model_configurations.py
    ├── test_pretraining_orchestrator.py
    ├── test_tokenization.py
    └── test_utils.py
```

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

3. **Parameterized Tests**: Tests across multiple configuration combinations
   - Test grid configurations under `/config/test_grids/`

## Running Tests

### Basic Test Commands
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)

```bash
# Run all tests
make test

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
make test-unit

# Run only integration tests
make test-integration

# Run all tests with coverage report
make test-all
```

### Parameterized Testing

For testing multiple configuration combinations:

```bash
# Run tests with a specific parameter grid
make test-grid GRID=config/test_grids/minimal_test_grid.yaml

# Run the minimal test grid (faster)
make test-minimal

# Run the comprehensive test grid (more thorough but slower)
make test-comprehensive
```

### Custom Test Commands

For more control, you can use the parameterized test script directly:

```bash
python scripts/run_parameterized_tests.py --test-type=unit --test-pattern="tokenization" --param-file=config/test_grids/minimal_test_grid.yaml
```

## Creating New Tests

When adding new features, please follow these guidelines:

1. **Add Unit Tests**: For each new component/function
2. **Add Integration Tests**: If the component interacts with others
3. **Update Parameter Grids**: If adding configurable parameters
4. **Use Fixtures**: Leverage existing fixtures in `conftest.py`

## Continuous Integration

The project includes GitHub Actions workflows in `.github/workflows/ci.yml` that run tests automatically:

- On every push to the main branch
- On every pull request to the main branch
- Tests run with different Python versions (3.8, 3.9, 3.10)
- Includes separate workflow for distributed tests

## Creating Mock Data

For testing with mock data:

```python
from tests.fixtures.data_fixtures import create_mock_text_data, create_mock_tokenized_dataset

# Create mock text files for tokenization tests
data_dir = create_mock_text_data()

# Create mock tokenized dataset for pretraining tests
dataset_dir = create_mock_tokenized_dataset()
```

Or use the fixtures directly in tests:

```python
def test_example(mock_text_data, mock_tokenized_dataset):
    # Test using the mock data...
    pass
>>>>>>> cd01e49 (Add testing dependencies and configurations for unit and integration tests)
```