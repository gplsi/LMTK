"""
Unit tests for the configuration validation system.
"""
import pytest
from pathlib import Path
import tempfile
import os
import yaml

from src.config.config_loader import ConfigValidator


@pytest.mark.unit
@pytest.mark.config
class TestConfigValidation:
    
    def test_config_validator_initialization(self):
        """Test that the ConfigValidator initializes correctly"""
        validator = ConfigValidator()
        assert validator is not None
    
    def test_tokenization_config_validation(self, tokenizer_config):
        """Test tokenization config validation"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(tokenizer_config.to_dict(), f)
            config_path = f.name
        
        try:
            # Validate the config
            validator = ConfigValidator()
            config = validator.validate(config_path, "tokenization")
            
            # Check that essential fields are present
            assert config.task == "tokenization"
            assert config.tokenizer.name == "gpt2"
            assert config.tokenizer.context_length == 1024
            assert config.dataset.source == "local"
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_pretraining_config_validation(self, pretraining_config):
        """Test pretraining config validation"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(pretraining_config.to_dict(), f)
            config_path = f.name
        
        try:
            # Validate the config
            validator = ConfigValidator()
            config = validator.validate(config_path, "pretraining")
            
            # Check that essential fields are present
            assert config.task == "pretraining"
            assert config.model_name == "gpt2"
            assert config.batch_size == 2
            assert config.parallelization_strategy == "none"
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_invalid_config(self):
        """Test validation of an invalid configuration"""
        # Create an invalid config file (missing required fields)
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump({"task": "pretraining"}, f)  # Missing required fields
            config_path = f.name
        
        try:
            # Validate the config - should raise an exception
            validator = ConfigValidator()
            with pytest.raises(Exception):
                validator.validate(config_path, "pretraining")
        finally:
            # Clean up
            os.unlink(config_path)