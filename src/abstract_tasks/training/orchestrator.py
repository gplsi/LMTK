"""
Abstract training orchestrator.

This module provides the base orchestrator for all training tasks,
including pretraining and instruction fine-tuning. It handles common
functionality such as device detection, strategy selection, and
execution flow.
"""

import os
import logging
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple

# Import distributed training strategies from pretraining implementation
from src.tasks.pretraining.fabric.distributed import (
    FSDP,
    DeepSpeed,
    DistributedDataParallel as DDP,
    DataParallel as DP,
)

# Import metrics logger
from src.abstract_tasks.training.metrics_logger import create_metrics_logger

logger = logging.getLogger(__name__)


class TrainingOrchestrator(ABC):
    """
    Base orchestrator for training tasks.
    
    This abstract class provides the foundation for all training orchestrators,
    handling common functionality such as device detection, strategy selection,
    and execution flow. Task-specific orchestrators (like pretraining and
    instruction fine-tuning) should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize the training orchestrator.
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.devices = self._detect_devices()
        self.output_dir = self._get_output_dir()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metrics logger
        self.metrics_logger = create_metrics_logger(
            config=self.config,
            output_dir=self.output_dir,
            experiment_name=getattr(self.config, "experiment_name", None),
        )
    
    def _detect_devices(self) -> Union[int, List[int]]:
        """
        Detect available devices for training.
        
        Returns:
            Number of devices or list of device IDs
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU.")
            return 1
        
        # Get number of CUDA devices
        num_devices = torch.cuda.device_count()
        logger.info(f"Found {num_devices} CUDA device(s).")
        
        # Check if specific devices are requested
        if hasattr(self.config, "devices") and self.config.devices:
            devices = self.config.devices
            if isinstance(devices, list):
                logger.info(f"Using specified devices: {devices}")
                return devices
            elif isinstance(devices, int):
                if devices > num_devices:
                    logger.warning(
                        f"Requested {devices} devices, but only {num_devices} available. "
                        f"Using {num_devices} devices."
                    )
                    return num_devices
                logger.info(f"Using {devices} device(s).")
                return devices
        
        # Default: use all available devices
        logger.info(f"Using all {num_devices} available device(s).")
        return num_devices
    
    def _get_output_dir(self) -> str:
        """
        Get the output directory for training artifacts.
        
        Returns:
            Path to the output directory
        """
        # Default output directory
        output_dir = "output"
        
        # Check if output directory is specified in config
        if hasattr(self.config, "output_dir") and self.config.output_dir:
            output_dir = self.config.output_dir
        
        # Add experiment name if available
        if hasattr(self.config, "experiment_name") and self.config.experiment_name:
            output_dir = os.path.join(output_dir, self.config.experiment_name)
        
        return output_dir
    
    def execute(self) -> None:
        """
        Execute the training task.
        
        This method handles the execution flow based on the configuration,
        selecting the appropriate training strategy and running the task.
        """
        # Determine execution mode
        mode = getattr(self.config, "mode", "train")
        
        if mode == "process_dataset":
            # Process dataset mode
            self._process_dataset()
        elif mode == "train":
            # Training mode
            strategy = self._get_training_strategy()
            self._train(strategy)
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
    
    def _get_training_strategy(self) -> str:
        """
        Get the training strategy from the configuration.
        
        Returns:
            Name of the training strategy
        """
        # Default strategy
        strategy = "fsdp"
        
        # Check if strategy is specified in config
        if hasattr(self.config, "strategy") and self.config.strategy:
            strategy = self.config.strategy.lower()
        
        logger.info(f"Using {strategy.upper()} training strategy.")
        return strategy
    
    def _train(self, strategy: str) -> None:
        """
        Train the model using the specified strategy.
        
        Args:
            strategy: Name of the training strategy
        """
        # Select the appropriate trainer based on the strategy
        if strategy == "fsdp":
            trainer = self._setup_fsdp()
        elif strategy == "deepspeed":
            trainer = self._setup_deepspeed()
        elif strategy == "ddp":
            trainer = self._setup_ddp()
        elif strategy == "dp":
            trainer = self._setup_dp()
        else:
            raise ValueError(f"Unknown training strategy: {strategy}")
        
        # Set up the trainer
        trainer.setup()
        
        # Start training
        trainer.train()
    
    def _setup_fsdp(self) -> FSDP:
        """
        Set up the FSDP trainer.
        
        Returns:
            Configured FSDP trainer
        """
        return FSDP(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=logger,
        )
    
    def _setup_deepspeed(self) -> DeepSpeed:
        """
        Set up the DeepSpeed trainer.
        
        Returns:
            Configured DeepSpeed trainer
        """
        return DeepSpeed(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=logger,
        )
    
    def _setup_ddp(self) -> DDP:
        """
        Set up the DDP trainer.
        
        Returns:
            Configured DDP trainer
        """
        return DDP(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=logger,
        )
    
    def _setup_dp(self) -> DP:
        """
        Set up the DataParallel trainer.
        
        Returns:
            Configured DataParallel trainer
        """
        return DP(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=logger,
        )
    
    @abstractmethod
    def _process_dataset(self) -> None:
        """
        Process the dataset for training.
        
        This method should be implemented by subclasses to handle
        dataset-specific processing.
        """
        pass
