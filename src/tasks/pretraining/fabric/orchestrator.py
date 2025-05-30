"""
Pretraining-specific implementation of the Fabric orchestrator.

This module provides a concrete implementation of the abstract Fabric orchestrator
for pretraining tasks.
"""

import logging
from typing import Any, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.fabric.orchestrator import FabricOrchestrator
from src.tasks.pretraining.fabric.strategies import (
    PretrainingFabricFSDPStrategy,
    PretrainingFabricDeepSpeedStrategy,
    PretrainingFabricDDPStrategy,
    PretrainingFabricDataParallelStrategy,
)

logger = logging.getLogger(__name__)


class PretrainingFabricOrchestrator(FabricOrchestrator):
    """
    Pretraining-specific implementation of the Fabric orchestrator.
    
    This class extends the base Fabric orchestrator to implement pretraining-specific
    functionality, such as dataset processing and trainer creation.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the pretraining Fabric orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _create_fsdp_strategy(self) -> PretrainingFabricFSDPStrategy:
        """
        Create an FSDP strategy for pretraining.
        
        Returns:
            An instance of the pretraining FSDP strategy
        """
        self._ensure_dataset_loaded()
        return PretrainingFabricFSDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_deepspeed_strategy(self) -> PretrainingFabricDeepSpeedStrategy:
        """
        Create a DeepSpeed strategy for pretraining.
        
        Returns:
            An instance of the pretraining DeepSpeed strategy
        """
        self._ensure_dataset_loaded()
        return PretrainingFabricDeepSpeedStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_ddp_strategy(self) -> PretrainingFabricDDPStrategy:
        """
        Create a DDP strategy for pretraining.
        
        Returns:
            An instance of the pretraining DDP strategy
        """
        self._ensure_dataset_loaded()
        return PretrainingFabricDDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_dataparallel_strategy(self) -> PretrainingFabricDataParallelStrategy:
        """
        Create a DataParallel strategy for pretraining.
        
        Returns:
            An instance of the pretraining DataParallel strategy
        """
        self._ensure_dataset_loaded()
        return PretrainingFabricDataParallelStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _ensure_dataset_loaded(self) -> None:
        """
        Ensure that the dataset is loaded before creating a strategy.
        
        If the dataset is not already loaded, this method raises an error
        as the dataset should be loaded by the parent orchestrator.
        
        Raises:
            ValueError: If the dataset is not loaded
        """
        if self.processed_dataset is None:
            raise ValueError(
                "Dataset is not loaded. The PretrainingOrchestrator should load "
                "the dataset before delegating to the framework-specific orchestrator."
            )
    
    def _process_dataset(self) -> Union[HFDataset, DatasetDict]:
        """
        Process the dataset for pretraining.
        
        Returns:
            The processed dataset
        """
        # Implement pretraining-specific dataset processing
        # This would typically include tokenization, data augmentation, etc.
        pass
