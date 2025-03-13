"""
Unit tests for the fabric training components.
"""
import pytest
import torch
import lightning as L
from unittest.mock import patch, MagicMock, PropertyMock

from src.tasks.pretraining.fabric.distributed import setup_distributed_strategy
from src.tasks.pretraining.fabric.logger import setup_loggers


@pytest.mark.unit
class TestFabricComponents:
    
    @pytest.mark.parametrize("parallelization_strategy,expected_strategy_class", [
        ("none", "SingleDeviceStrategy"),
        ("ddp", "DDPStrategy"),
        ("fsdp", "FSDPStrategy"),
        ("deep_speed", "DeepSpeedStrategy")
    ])
    def test_setup_distributed_strategy(self, parallelization_strategy, expected_strategy_class):
        """Test that different parallelization strategies create the correct Lightning strategy objects"""
        config = MagicMock()
        config.parallelization_strategy = parallelization_strategy
        
        # For fsdp setup
        if parallelization_strategy == "fsdp":
            config.auto_wrap_policy = "gpt2"
            config.sharding_strategy = "FULL_SHARD"
            config.limit_all_gathers = True
            config.cpu_offload = False
        
        # For ddp setup
        if parallelization_strategy == "ddp":
            config.backend = "gloo"
        
        # For deepspeed setup
        if parallelization_strategy == "deep_speed":
            config.deepspeed_config_path = "config/deepspeed_config.json"
        
        with patch(f"lightning.fabric.strategies.{expected_strategy_class}") as mock_strategy:
            mock_strategy_instance = MagicMock()
            mock_strategy.return_value = mock_strategy_instance
            
            strategy = setup_distributed_strategy(config)
            
            # Check that the correct strategy is returned
            assert strategy is mock_strategy_instance
            assert mock_strategy.called
    
    @pytest.mark.parametrize("logging_config", ["none", "wandb"])
    def test_setup_loggers(self, logging_config):
        """Test logger setup for different configurations"""
        config = MagicMock()
        config.logging_config = logging_config
        
        # Setup for wandb
        if logging_config == "wandb":
            config.wandb_project = "test_project"
            config.wandb_entity = "test_entity"
            config.log_model = False
        
        with patch("lightning.pytorch.loggers.WandbLogger") as mock_wandb_logger:
            mock_wandb_logger_instance = MagicMock()
            mock_wandb_logger.return_value = mock_wandb_logger_instance
            
            loggers = setup_loggers(config)
            
            # Check logger setup
            if logging_config == "none":
                assert loggers == []
            else:
                assert mock_wandb_logger_instance in loggers
                mock_wandb_logger.assert_called_with(
                    project=config.wandb_project, 
                    entity=config.wandb_entity,
                    log_model=config.log_model
                )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fabric_strategy_gpu_setup(self):
        """Test fabric strategy setup with GPU"""
        with patch("lightning.fabric.Fabric") as mock_fabric:
            mock_fabric_instance = MagicMock()
            mock_fabric.return_value = mock_fabric_instance
            
            device_count_patch = patch("torch.cuda.device_count", return_value=2)
            
            with device_count_patch:
                from src.tasks.pretraining.fabric.base import FabricTrainerBase
                
                class TestTrainer(FabricTrainerBase):
                    def _pipeline(self):
                        pass
                
                config = MagicMock()
                config.parallelization_strategy = "ddp"
                config.backend = "nccl"
                config.precision = "16-mixed"
                config.logging_config = "none"
                
                trainer = TestTrainer(config)
                trainer.setup()
                
                # Check that fabric was set up correctly with GPU devices
                mock_fabric.assert_called_with(
                    devices=2,  # Should detect 2 GPUs
                    strategy=mock_fabric_instance._setup_strategy.return_value,
                    precision=config.precision,
                    loggers=[]
                )
                mock_fabric_instance.launch.assert_called_once()