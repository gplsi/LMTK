#!/usr/bin/env python3

"""
Unit tests for schema validation with the sequential naming convention.
This tests configurations for different combinations of task, framework, and strategy,
and validates them using the ConfigValidator.
"""

import unittest
import sys
import yaml
import tempfile
from pathlib import Path
from src.config.config_loader import ConfigValidator
from src.utils.logging import setup_logging, VerboseLevel, get_logger

logger = get_logger(__name__)

def create_test_config(task, framework=None, strategy=None, output_dir=None):
    """Create a test configuration for the given task, framework, and strategy."""
    config = {
        "experiment_name": f"test_{task}_{framework or 'none'}_{strategy or 'none'}",
        "task": task,
        "output": {
            "base_dir": "outputs",
            "logs_dir": "logs",
            "checkpoints_dir": "checkpoints"
        }
    }
    
    if framework:
        config["framework"] = framework
    
    if strategy:
        config["strategy"] = strategy
    
    if output_dir:
        config["output_dir"] = output_dir
    
    # Add task-specific configurations
    if task == "pretraining":
        config.update({
            "dataset": {
                "source": "huggingface",
                "nameOrPath": "wikitext",
                "format": "dataset",
                "split": "train"
            },
            "model": {
                "name_or_path": "gpt2"
            }
        })
    elif task == "instruction":
        config.update({
            "dataset": {
                "source": "huggingface",
                "nameOrPath": "tatsu-lab/alpaca",
                "format": "dataset",
                "split": "train"
            },
            "model": {
                "name_or_path": "gpt2"
            }
        })
    elif task == "tokenization":
        config.update({
            "dataset": {
                "source": "huggingface",
                "nameOrPath": "wikitext",
                "format": "dataset",
                "split": "train"
            },
            "tokenizer": {
                "name": "gpt2",
                "context_length": 1024,
                "task": "causal_pretraining"
            }
        })
    
    # Add framework-specific configurations
    if framework == "fabric":
        config.update({
            "precision": "bf16-mixed",
            "devices": 1,
            "max_epochs": 1
        })
    
    # Add strategy-specific configurations
    if strategy == "fsdp" and framework == "fabric":
        config.update({
            "fsdp": {
                "sharding_strategy": "full_shard",
                "auto_wrap_policy": "transformer"
            }
        })
    elif strategy == "deepspeed" and framework == "fabric":
        config.update({
            "deepspeed": {
                "zero_stage": 2
            }
        })
    elif strategy == "ddp" and framework == "fabric":
        config.update({
            "ddp": {
                "find_unused_parameters": False
            }
        })
    elif strategy == "dp" and framework == "fabric":
        config.update({
            "dp": {
                "sync_batch_norm": False
            }
        })
    
    return config


class TestSchemaValidation(unittest.TestCase):
    """Test cases for schema validation with the sequential naming convention."""
    
    def setUp(self):
        """Set up the test environment."""
        setup_logging(VerboseLevel.INFO)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)
        self.validator = ConfigValidator()
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def _validate_config(self, task, framework=None, strategy=None):
        """Helper method to validate a configuration for the given task, framework, and strategy."""
        # Create test config
        config = create_test_config(task, framework, strategy)
        
        # Create a temporary config file
        config_file = self.config_dir / f"{task}_{framework or 'none'}_{strategy or 'none'}.yaml"
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Validate the config
        try:
            # For training tasks, use the training schema
            if task in ["pretraining", "instruction"]:
                schema_name = "training"
            else:
                # For non-training tasks, use the task name directly
                schema_name = task
            
            self.validator.validate(config_file, schema_name)
            return True
        except Exception as e:
            logger.error(f"Validation failed for {config_file}: {e}")
            return False
    
    def test_tokenization(self):
        """Test validation for tokenization task."""
        self.assertTrue(self._validate_config("tokenization"))
    
    def test_pretraining(self):
        """Test validation for pretraining task."""
        self.assertTrue(self._validate_config("pretraining"))
    
    def test_instruction(self):
        """Test validation for instruction task."""
        self.assertTrue(self._validate_config("instruction"))
    
    def test_pretraining_fabric(self):
        """Test validation for pretraining task with fabric framework."""
        self.assertTrue(self._validate_config("pretraining", "fabric"))
    
    def test_instruction_fabric(self):
        """Test validation for instruction task with fabric framework."""
        self.assertTrue(self._validate_config("instruction", "fabric"))
    
    def test_pretraining_fabric_fsdp(self):
        """Test validation for pretraining task with fabric framework and fsdp strategy."""
        self.assertTrue(self._validate_config("pretraining", "fabric", "fsdp"))
    
    def test_pretraining_fabric_deepspeed(self):
        """Test validation for pretraining task with fabric framework and deepspeed strategy."""
        self.assertTrue(self._validate_config("pretraining", "fabric", "deepspeed"))
    
    def test_pretraining_fabric_ddp(self):
        """Test validation for pretraining task with fabric framework and ddp strategy."""
        self.assertTrue(self._validate_config("pretraining", "fabric", "ddp"))
    
    def test_pretraining_fabric_dp(self):
        """Test validation for pretraining task with fabric framework and dp strategy."""
        self.assertTrue(self._validate_config("pretraining", "fabric", "dp"))
    
    def test_instruction_fabric_fsdp(self):
        """Test validation for instruction task with fabric framework and fsdp strategy."""
        self.assertTrue(self._validate_config("instruction", "fabric", "fsdp"))
    
    def test_instruction_fabric_deepspeed(self):
        """Test validation for instruction task with fabric framework and deepspeed strategy."""
        self.assertTrue(self._validate_config("instruction", "fabric", "deepspeed"))
    
    def test_instruction_fabric_ddp(self):
        """Test validation for instruction task with fabric framework and ddp strategy."""
        self.assertTrue(self._validate_config("instruction", "fabric", "ddp"))
    
    def test_instruction_fabric_dp(self):
        """Test validation for instruction task with fabric framework and dp strategy."""
        self.assertTrue(self._validate_config("instruction", "fabric", "dp"))


if __name__ == "__main__":
    unittest.main()
