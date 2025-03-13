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
    ├── test_config.py
    ├── test_dataset.py
    ├── test_fabric.py
    ├── test_model_configurations.py
    ├── test_pretraining_orchestrator.py
    ├── test_tokenization.py
    └── test_utils.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
make test

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
```