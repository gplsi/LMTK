"""
Abstract Fabric orchestrator.

This module provides the base orchestrator for Fabric-based training,
handling the setup and execution of distributed training using Lightning Fabric strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.orchestrator import TrainingOrchestrator

logger = logging.getLogger(__name__)


class FabricOrchestrator(TrainingOrchestrator, ABC):
    """
    Base orchestrator for Fabric-based training.
    
    This abstract class provides the foundation for all Fabric-based orchestrators,
    handling common functionality such as trainer creation and dataset management.
    Task-specific orchestrators should inherit from this class and implement
    the abstract methods for creating trainers.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the Fabric orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _create_trainer(self, strategy: str) -> Any:
        """
        Create a trainer based on the specified strategy.
        
        Args:
            strategy: The name of the strategy to use
            
        Returns:
            A configured trainer for the specified strategy
            
        Raises:
            ValueError: If the strategy is not supported
        """
        self._ensure_dataset_loaded()
        
        if strategy == "fsdp":
            return self._create_fsdp_strategy()
        elif strategy == "deepspeed":
            return self._create_deepspeed_strategy()
        elif strategy == "ddp":
            return self._create_ddp_strategy()
        elif strategy == "dataparallel":
            return self._create_dataparallel_strategy()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    @abstractmethod
    def _create_fsdp_strategy(self) -> Any:
        """
        Create an FSDP strategy.
        
        Returns:
            A configured FSDP strategy
        """
        pass
    
    @abstractmethod
    def _create_deepspeed_strategy(self) -> Any:
        """
        Create a DeepSpeed strategy.
        
        Returns:
            A configured DeepSpeed strategy
        """
        pass
    
    @abstractmethod
    def _create_ddp_strategy(self) -> Any:
        """
        Create a DDP strategy.
        
        Returns:
            A configured DDP strategy
        """
        pass
    
    @abstractmethod
    def _create_dataparallel_strategy(self) -> Any:
        """
        Create a DataParallel strategy.
        
        Returns:
            A configured DataParallel strategy
        """
        pass
    
    def _ensure_dataset_loaded(self) -> None:
        """
        Ensure that the dataset is loaded before creating a trainer.
        
        If the dataset is not already loaded, this method raises an error
        as the dataset should be loaded by the parent orchestrator.
        
        Raises:
            ValueError: If the dataset is not loaded
        """
        if self.processed_dataset is None:
            raise ValueError(
                "Dataset is not loaded. The task orchestrator should load "
                "the dataset before delegating to the framework-specific orchestrator."
            )
    
    def _process_dataset(self) -> None:
        """
        Process the dataset.
        
        This method should not be called directly on the framework-specific orchestrator.
        Dataset processing should be handled by the parent task orchestrator.
        
        Raises:
            NotImplementedError: Always, as this method should not be called
        """
        raise NotImplementedError(
            "Dataset processing should be handled by the task orchestrator, "
            "not the framework-specific orchestrator."
        )
    
    def _create_strategy_trainer(self, strategy_name: str, trainer_class: Any) -> Any:
        """
        Template method for creating strategy-specific trainers.
        
        Args:
            strategy_name: The name of the strategy
            trainer_class: The trainer class to instantiate
            
        Returns:
            An instance of the trainer class
        """
        logger.info(f"Creating Fabric {strategy_name} trainer")
        return trainer_class(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )