"""
Pretraining-specific implementations of Fabric-based trainers.

This module provides concrete implementations of the abstract Fabric-based
training strategies for pretraining tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.fabric.strategies import (
    FabricFSDPTrainer,
    FabricDeepSpeedTrainer,
    FabricDDPTrainer,
    FabricDataParallelTrainer,
)
from src.tasks.pretraining.fabric.base import PretrainingFabricBase
import lightning as L

logger = logging.getLogger(__name__)


class PretrainingFabricFSDPTrainer(FabricFSDPTrainer):
    """
    Pretraining-specific implementation of the Fabric FSDP trainer.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        dataset: Union[HFDataset, DatasetDict],
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the pretraining FSDP trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    # Implementation of abstract methods from FabricFSDPTrainer
    # ...


class PretrainingFabricDeepSpeedTrainer(FabricDeepSpeedTrainer):
    """
    Pretraining-specific implementation of the Fabric DeepSpeed trainer.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        dataset: Union[HFDataset, DatasetDict],
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the pretraining DeepSpeed trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    # Implementation of abstract methods from FabricDeepSpeedTrainer
    # ...


class PretrainingFabricDDPTrainer(FabricDDPTrainer):
    """
    Pretraining-specific implementation of the Fabric DDP trainer.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        dataset: Union[HFDataset, DatasetDict],
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the pretraining DDP trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    # Implementation of abstract methods from FabricDDPTrainer
    # ...


class PretrainingFabricDataParallelTrainer(FabricDataParallelTrainer):
    """
    Pretraining-specific implementation of the Fabric DataParallel trainer.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        dataset: Union[HFDataset, DatasetDict],
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the pretraining DataParallel trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    # Implementation of abstract methods from FabricDataParallelTrainer
    # ...
