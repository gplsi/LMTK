"""
Unit tests for the pretraining orchestrator system.
"""
import pytest
import os
import tempfile
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tasks.pretraining.orchestrator import ContinualOrchestrator


@pytest.mark.unit
class TestPretrainingOrchestrator:

    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_model")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_fabric")
    def test_orchestrator_initialization(self, mock_setup_fabric, mock_setup_model, pretraining_config):
        """Test that the orchestrator initializes correctly"""
        orchestrator = ContinualOrchestrator(pretraining_config)
        assert orchestrator.config == pretraining_config
        mock_setup_fabric.assert_not_called()  # Should not be called during init
        mock_setup_model.assert_not_called()  # Should not be called during init
    
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_model")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_fabric")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.load_dataset")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator.train")
    def test_execute_method(self, mock_train, mock_load_dataset, mock_setup_fabric, 
                           mock_setup_model, pretraining_config):
        """Test the orchestrator's execute method"""
        # Configure mocks
        mock_load_dataset.return_value = MagicMock()
        mock_train.return_value = None
        
        # Create orchestrator and call execute
        orchestrator = ContinualOrchestrator(pretraining_config)
        orchestrator.execute()
        
        # Verify that all necessary methods were called
        mock_setup_model.assert_called_once()
        mock_setup_fabric.assert_called_once()
        mock_load_dataset.assert_called_once()
        mock_train.assert_called_once()
    
    @patch("torch.load")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_model")
    @patch("src.tasks.pretraining.orchestrator.ContinualOrchestrator._setup_fabric")
    def test_load_checkpoint(self, mock_setup_fabric, mock_setup_model, mock_torch_load, 
                            pretraining_config, temp_output_dir):
        """Test loading from a checkpoint"""
        # Configure the config to use a checkpoint
        config = pretraining_config
        checkpoint_path = temp_output_dir / "checkpoint.pt"
        config.checkpoint = str(checkpoint_path)
        
        # Create a mock checkpoint file
        mock_checkpoint = {
            "model": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": 5,
            "global_step": 100
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock model and optimizer
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        # Create orchestrator with the mocks
        orchestrator = ContinualOrchestrator(config)
        orchestrator.model = mock_model
        orchestrator.optimizer = mock_optimizer
        
        # Call load_checkpoint
        epoch, global_step = orchestrator._load_checkpoint(checkpoint_path)
        
        # Verify checkpoint was loaded
        assert epoch == 5
        assert global_step == 100
        mock_torch_load.assert_called_once_with(checkpoint_path, map_location="cpu")
    
    @patch("src.utils.dataset.storage.DatasetHandler")
    def test_load_dataset(self, mock_dataset_handler_class, pretraining_config):
        """Test loading a dataset"""
        # Configure mock
        mock_dataset_handler = MagicMock()
        mock_dataset_handler_class.return_value = mock_dataset_handler
        
        # Mock the load_from_disk method
        mock_dataset = MagicMock()
        mock_dataset_handler.load_from_disk.return_value = mock_dataset
        
        orchestrator = ContinualOrchestrator(pretraining_config)
        dataset = orchestrator.load_dataset()
        
        # Verify dataset is loaded correctly
        assert dataset == mock_dataset
        mock_dataset_handler_class.assert_called_once()
        mock_dataset_handler.load_from_disk.assert_called_once_with(
            pretraining_config.dataset.nameOrPath
        )