"""
Integration tests for the pretraining pipeline.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml
import torch
from box import Box
from datasets import Dataset

from src.tasks.pretraining import execute

# Import GPU requirement marker from conftest
from tests.conftest import requires_gpu


@pytest.mark.integration
class TestPretrainingPipeline:
    
    def setup_method(self):
        """Set up test environment before each test"""
        # Create temp directories for input and output
        self.input_dir = Path(tempfile.mkdtemp(prefix="test_input_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="test_output_"))
        
        # Create mock tokenized dataset
        os.makedirs(self.input_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up after each test"""
        shutil.rmtree(self.input_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.load_dataset")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.train")
    def test_pretraining_basic_pipeline(self, mock_train, mock_load_dataset, mock_model_from_pretrained):
        """Test basic pretraining pipeline execution with no distributed strategy"""
        # Create mock model
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        # Create config for test
        config = Box({
            "task": "pretraining",
            "output_dir": str(self.output_dir),
            "model_name": "gpt2",
            "precision": "16-mixed",
            "number_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "lr": 1e-5,
            "lr_decay": True,
            "lr_scheduler": "cosine",
            "warmup_proportion": 0.1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "dataset": {
                "source": "local",
                "nameOrPath": str(self.input_dir)
            },
            "parallelization_strategy": "none",
            "logging_config": "none"
        }, box_dots=True)
        
        # Execute the pretraining task
        with patch("src.tasks.pretraining.fabric.base.FabricTrainerBase.setup"):
            result = execute(config)
            
            # Verify model loading was called
            mock_model_from_pretrained.assert_called_with("gpt2")
            
            # Verify dataset loading and training were called
            mock_load_dataset.assert_called_once()
            mock_train.assert_called_once()
    
    @patch("lightning.fabric.Fabric")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.load_dataset")
    @patch("torch.cuda.device_count", return_value=2)  # Mock GPU count
    @patch("torch.cuda.is_available", return_value=True)  # Force GPU availability
    def test_pretraining_with_fsdp(self, mock_cuda_available, mock_device_count, 
                                  mock_load_dataset, mock_model_from_pretrained, mock_fabric):
        """Test pretraining with FSDP strategy"""
        # Create mock instances
        mock_fabric_instance = MagicMock()
        mock_fabric.return_value = mock_fabric_instance
        
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        # Create config for FSDP test
        config = Box({
            "task": "pretraining",
            "output_dir": str(self.output_dir),
            "model_name": "gpt2",
            "precision": "16-mixed",
            "number_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "lr": 1e-5,
            "lr_decay": True,
            "lr_scheduler": "cosine",
            "warmup_proportion": 0.1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "dataset": {
                "source": "local",
                "nameOrPath": str(self.input_dir)
            },
            "parallelization_strategy": "fsdp",
            "auto_wrap_policy": "gpt2",
            "sharding_strategy": "FULL_SHARD",
            "state_dict_type": "sharded",
            "limit_all_gathers": True,
            "cpu_offload": False,
            "num_workers": 1,
            "gradient_checkpointing": False,
            "logging_config": "none"
        }, box_dots=True)
        
        # Execute pretraining with FSDP
        with patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.train"):
            execute(config)
            
            # Verify FSDP setup
            mock_fabric.assert_called()
            mock_fabric_instance.launch.assert_called()
    
    @patch("lightning.fabric.Fabric")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.load_dataset")
    @patch("torch.cuda.device_count", return_value=2)  # Mock GPU count
    @patch("torch.cuda.is_available", return_value=True)  # Force GPU availability
    def test_pretraining_with_ddp(self, mock_cuda_available, mock_device_count, 
                                 mock_load_dataset, mock_model_from_pretrained, mock_fabric):
        """Test pretraining with DDP strategy"""
        # Create mock instances
        mock_fabric_instance = MagicMock()
        mock_fabric.return_value = mock_fabric_instance
        
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        # Create config for DDP test
        config = Box({
            "task": "pretraining",
            "output_dir": str(self.output_dir),
            "model_name": "gpt2",
            "precision": "16-mixed",
            "number_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "lr": 1e-5,
            "lr_decay": True,
            "lr_scheduler": "cosine",
            "warmup_proportion": 0.1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "dataset": {
                "source": "local",
                "nameOrPath": str(self.input_dir)
            },
            "parallelization_strategy": "ddp",
            "backend": "gloo",  # Using gloo backend for testing
            "num_workers": 1,
            "logging_config": "none"
        }, box_dots=True)
        
        # Execute pretraining with DDP
        with patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.train"):
            execute(config)
            
            # Verify DDP setup
            mock_fabric.assert_called()
            mock_fabric_instance.launch.assert_called()
    
    @patch("src.config.config_loader.ConfigValidator.validate")
    def test_pretraining_from_config_file(self, mock_validate):
        """Test pretraining from a config file using the main entry point"""
        # Create a config file
        config = {
            "task": "pretraining",
            "output_dir": str(self.output_dir),
            "model_name": "gpt2",
            "precision": "16-mixed",
            "number_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "lr": 1e-5,
            "lr_decay": True,
            "lr_scheduler": "cosine",
            "warmup_proportion": 0.1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "dataset": {
                "source": "local",
                "nameOrPath": str(self.input_dir)
            },
            "parallelization_strategy": "none",
            "logging_config": "none"
        }
        
        # Write config to a temporary file
        config_path = self.output_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Mock the validator and pretraining execution
        mock_validate.return_value = Box(config, box_dots=True)
        
        with patch("src.tasks.pretraining.execute") as mock_execute:
            # Import and execute the main function
            from src.main import execute_task
            execute_task(str(config_path))
            
            # Check that the pretraining was executed
            mock_execute.assert_called_once()
            
            # Verify that the config was validated
            mock_validate.assert_called_once_with(str(config_path), "pretraining")