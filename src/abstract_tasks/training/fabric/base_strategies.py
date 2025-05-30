"""
Base implementations for Fabric strategy classes.

This module provides base implementations for the different Fabric-based
training strategies, with common functionality that can be reused across tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import lightning as L
from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.fabric.strategies import (
    FabricFSDPTrainer,
    FabricDeepSpeedTrainer,
    FabricDDPTrainer,
    FabricDataParallelTrainer,
)

logger = logging.getLogger(__name__)


class BaseFabricFSDPStrategy(FabricFSDPTrainer):
    """
    Base implementation of the Fabric FSDP strategy.
    
    This class provides common functionality for FSDP-based training
    that can be reused across different tasks.
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
        Initialize the base Fabric FSDP strategy.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    def _initialize_fabric(self) -> None:
        """
        Initialize Fabric with the configured strategy.
        """
        # Set up loggers
        loggers = self._setup_loggers()
        
        # Initialize Fabric
        self.fabric = L.Fabric(
            devices=self.devices,
            strategy=self.strategy,
            precision=getattr(self.config, "precision", "32-true"),
            loggers=loggers,
        )
    
    def _setup_loggers(self) -> List[Any]:
        """
        Set up loggers for the trainer.
        
        Returns:
            A list of loggers
        """
        # Base implementation returns an empty list
        # Task-specific implementations can override this
        return []


class BaseFabricDeepSpeedStrategy(FabricDeepSpeedTrainer):
    """
    Base implementation of the Fabric DeepSpeed strategy.
    
    This class provides common functionality for DeepSpeed-based training
    that can be reused across different tasks.
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
        Initialize the base Fabric DeepSpeed strategy.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    def _initialize_fabric(self) -> None:
        """
        Initialize Fabric with the configured strategy.
        """
        # Set up loggers
        loggers = self._setup_loggers()
        
        # Initialize Fabric
        self.fabric = L.Fabric(
            devices=self.devices,
            strategy=self.strategy,
            precision=getattr(self.config, "precision", "32-true"),
            loggers=loggers,
        )
    
    def _setup_loggers(self) -> List[Any]:
        """
        Set up loggers for the trainer.
        
        Returns:
            A list of loggers
        """
        # Base implementation returns an empty list
        # Task-specific implementations can override this
        return []


class BaseFabricDDPStrategy(FabricDDPTrainer):
    """
    Base implementation of the Fabric DDP strategy.
    
    This class provides common functionality for DDP-based training
    that can be reused across different tasks.
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
        Initialize the base Fabric DDP strategy.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    def _initialize_fabric(self) -> None:
        """
        Initialize Fabric with the configured strategy.
        """
        # Set up loggers
        loggers = self._setup_loggers()
        
        # Initialize Fabric
        self.fabric = L.Fabric(
            devices=self.devices,
            strategy=self.strategy,
            precision=getattr(self.config, "precision", "32-true"),
            loggers=loggers,
        )
    
    def _setup_loggers(self) -> List[Any]:
        """
        Set up loggers for the trainer.
        
        Returns:
            A list of loggers
        """
        # Base implementation returns an empty list
        # Task-specific implementations can override this
        return []


class BaseFabricDataParallelStrategy(FabricDataParallelTrainer):
    """
    Base implementation of the Fabric DataParallel strategy.
    
    This class provides common functionality for DataParallel-based training
    that can be reused across different tasks.
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
        Initialize the base Fabric DataParallel strategy.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
    
    def _initialize_fabric(self) -> None:
        """
        Initialize Fabric with the configured strategy.
        """
        # Set up loggers
        loggers = self._setup_loggers()
        
        # Initialize Fabric
        self.fabric = L.Fabric(
            devices=self.devices,
            strategy=self.strategy,
            precision=getattr(self.config, "precision", "32-true"),
            loggers=loggers,
        )
    
    def _setup_loggers(self) -> List[Any]:
        """
        Set up loggers for the trainer.
        
        Returns:
            A list of loggers
        """
        # Base implementation returns an empty list
        # Task-specific implementations can override this
        return []
