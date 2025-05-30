"""
Abstract Fabric-based training strategies.

This module provides abstract base classes for different distributed training
strategies implemented with Lightning Fabric. These classes define the interface
that task-specific implementations should follow.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
from lightning.fabric.strategies import (
    FSDPStrategy,
    DeepSpeedStrategy,
    DDPStrategy,
    DataParallelStrategy,
)

from src.abstract_tasks.training.trainer import TrainerBase

logger = logging.getLogger(__name__)


class FabricTrainerBase(TrainerBase, ABC):
    """
    Base class for all Fabric-based trainers.
    
    This abstract class provides the foundation for all Fabric-based trainers,
    handling common functionality such as strategy setup and initialization.
    Task-specific trainers should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Fabric trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            cli_logger: Logger for CLI output
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.strategy = None
    
    def setup(self) -> None:
        """
        Set up the trainer.
        
        This method initializes the Fabric strategy, model, optimizer, scheduler,
        and dataloaders, and prepares them for training.
        """
        self.cli_logger.info("Setting up Fabric trainer")
        
        # Set up strategy
        self.strategy = self._setup_strategy()
        
        # Initialize Fabric (to be implemented by subclasses)
        self._initialize_fabric()
        
        # Continue with the standard setup
        super().setup()
    
    @abstractmethod
    def _setup_strategy(self) -> Any:
        """
        Set up and return the Fabric strategy.
        
        Returns:
            A configured Fabric strategy
        """
        pass
    
    @abstractmethod
    def _initialize_fabric(self) -> None:
        """
        Initialize Fabric with the configured strategy.
        
        This method should be implemented by subclasses to initialize
        the Fabric instance with the configured strategy.
        """
        pass


class FabricFSDPTrainer(FabricTrainerBase, ABC):
    """
    Abstract base class for FSDP-based trainers.
    
    This class defines the interface for trainers that use the
    Fully Sharded Data Parallel (FSDP) strategy.
    """
    
    def _setup_strategy(self) -> Union[FSDPStrategy, str]:
        """
        Set up and return the FSDP strategy.
        
        Returns:
            FSDPStrategy or str: A configured FSDP strategy when multiple devices are used,
            otherwise a string indicating an automatic strategy.
        """
        self.cli_logger.info("Setting up FSDP strategy.")
        if isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
            # Resolve FSDP configuration (to be implemented by subclasses)
            fsdp_config = self._resolve_fsdp_config()
            
            # Create FSDP strategy
            return FSDPStrategy(**fsdp_config)
        else:
            return "auto"
    
    @abstractmethod
    def _resolve_fsdp_config(self) -> Dict[str, Any]:
        """
        Resolve the FSDP configuration.
        
        Returns:
            A dictionary containing the FSDP configuration parameters
        """
        pass


class FabricDeepSpeedTrainer(FabricTrainerBase, ABC):
    """
    Abstract base class for DeepSpeed-based trainers.
    
    This class defines the interface for trainers that use the
    DeepSpeed strategy.
    """
    
    def _setup_strategy(self) -> DeepSpeedStrategy:
        """
        Set up and return the DeepSpeed strategy.
        
        Returns:
            DeepSpeedStrategy: A strategy object configured for DeepSpeed training.
        """
        self.cli_logger.info("Setting up DeepSpeed strategy.")
        if isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
            # Resolve DeepSpeed configuration (to be implemented by subclasses)
            ds_config = self._resolve_deepspeed_config()
            
            # Create DeepSpeed strategy
            return DeepSpeedStrategy(**ds_config)
        else:
            raise NotImplementedError(
                "DeepSpeed requires multiple devices. Use a different strategy for single-device training."
            )
    
    @abstractmethod
    def _resolve_deepspeed_config(self) -> Dict[str, Any]:
        """
        Resolve the DeepSpeed configuration.
        
        Returns:
            A dictionary containing the DeepSpeed configuration parameters
        """
        pass


class FabricDDPTrainer(FabricTrainerBase, ABC):
    """
    Abstract base class for DDP-based trainers.
    
    This class defines the interface for trainers that use the
    Distributed Data Parallel (DDP) strategy.
    """
    
    def _setup_strategy(self) -> Union[DDPStrategy, str]:
        """
        Set up and return the DDP strategy.
        
        Returns:
            DDPStrategy or str: A configured DDP strategy object for multiple devices,
            or a string indicating an automatic strategy if only one device is used.
        """
        self.cli_logger.info("Setting up DDP strategy.")
        if isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
            # Resolve DDP configuration (to be implemented by subclasses)
            ddp_config = self._resolve_ddp_config()
            
            # Create DDP strategy
            return DDPStrategy(**ddp_config)
        else:
            return "auto"
    
    @abstractmethod
    def _resolve_ddp_config(self) -> Dict[str, Any]:
        """
        Resolve the DDP configuration.
        
        Returns:
            A dictionary containing the DDP configuration parameters
        """
        pass


class FabricDataParallelTrainer(FabricTrainerBase, ABC):
    """
    Abstract base class for DataParallel-based trainers.
    
    This class defines the interface for trainers that use the
    DataParallel strategy.
    """
    
    def _setup_strategy(self) -> Union[DataParallelStrategy, str]:
        """
        Set up and return the DataParallel strategy.
        
        Returns:
            DataParallelStrategy or str: A DataParallel strategy object configured with the
            provided devices, or a string indicating an automatic strategy for single-device setups.
        """
        self.cli_logger.info("Setting up DataParallel strategy.")
        if isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
            # Create DataParallel strategy
            parallel_devices = self.devices if isinstance(self.devices, list) else list(range(self.devices))
            return DataParallelStrategy(
                parallel_devices=parallel_devices,
                output_device=parallel_devices[0],
            )
        else:
            return "auto"
