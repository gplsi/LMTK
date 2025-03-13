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

3. **Parameterized Tests**: Tests across multiple configuration combinations
   - Test grid configurations under `/config/test_grids/`

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

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
```