"""
Pytest configuration file.
Contains fixtures that are available to all tests.
"""
import os
import pytest
import tempfile
from pathlib import Path
import shutil
import torch
from box import Box

from tests.fixtures.configs import get_base_config, get_tokenizer_config, get_pretraining_config
from tests.fixtures.data_fixtures import create_mock_text_data, create_mock_tokenized_dataset, MockGPT2

@pytest.fixture(scope="session")
def base_config():
    """Base configuration fixture"""
    return get_base_config()

@pytest.fixture(scope="session")
def tokenizer_config():
    """Tokenizer configuration fixture"""
    return get_tokenizer_config()

@pytest.fixture(scope="session")
def pretraining_config():
    """Pretraining configuration fixture"""
    return get_pretraining_config()

@pytest.fixture(scope="session")
def pretraining_fsdp_config():
    """FSDP Pretraining configuration fixture"""
    return get_pretraining_config(parallelization_strategy="fsdp")

@pytest.fixture(scope="session")
def pretraining_ddp_config():
    """DDP Pretraining configuration fixture"""
    return get_pretraining_config(parallelization_strategy="ddp")

@pytest.fixture(scope="session")
def mock_text_data():
    """Create mock text data for tokenization tests"""
    data_path = create_mock_text_data()
    yield data_path
    # Cleanup is handled at module level

@pytest.fixture(scope="session")
def mock_tokenized_dataset():
    """Create mock tokenized dataset for pretraining tests"""
    data_path = create_mock_tokenized_dataset()
    yield data_path
    # Cleanup is handled at module level

@pytest.fixture
def mock_gpt2_model():
    """Create a mock GPT2 model for testing"""
    return MockGPT2()

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix="test_output_")
    yield Path(temp_dir)
    # Clean up after test
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def disable_wandb(monkeypatch):
    """Disable wandb for testing"""
    monkeypatch.setenv("WANDB_MODE", "disabled")

@pytest.fixture
def mock_hf_dataset(monkeypatch):
    """Mock HuggingFace datasets for testing"""
    class MockDataset:
        def __init__(self, *args, **kwargs):
            self.data = {"input_ids": torch.randint(0, 50000, (10, 128)),
                         "attention_mask": torch.ones(10, 128)}
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}
        
        def __len__(self):
            return len(self.data["input_ids"])
    
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: MockDataset())
    monkeypatch.setattr("datasets.load_from_disk", lambda *args, **kwargs: MockDataset())