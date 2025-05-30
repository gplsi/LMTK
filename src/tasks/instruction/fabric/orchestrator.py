"""
Fabric-based instruction orchestrator.

This module provides a Fabric-based implementation of the instruction orchestrator,
handling the setup and execution of distributed training using Lightning Fabric strategies.
"""

import logging
from typing import Any, Dict, List, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

# Import the abstract fabric orchestrator
from src.abstract_tasks.training.fabric.orchestrator import FabricOrchestrator

# Import instruction-specific trainers
from src.tasks.instruction.fabric.strategies import (
    InstructionFabricFSDPStrategy,
    InstructionFabricDeepSpeedStrategy,
    InstructionFabricDDPStrategy,
    InstructionFabricDataParallelStrategy,
)

logger = logging.getLogger(__name__)


class InstructionFabricOrchestrator(FabricOrchestrator):
    """
    Fabric-based orchestrator for instruction fine-tuning tasks.
    
    This class implements the abstract fabric orchestrator for instruction fine-tuning,
    providing concrete implementations of the trainer factory methods.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the instruction fabric orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _create_fsdp_trainer(self) -> InstructionFabricFSDPStrategy:
        """
        Create an FSDP trainer for instruction fine-tuning.
        
        Returns:
            Configured InstructionFabricFSDPStrategy
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric FSDP trainer for instruction fine-tuning")
        return InstructionFabricFSDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_deepspeed_trainer(self) -> InstructionFabricDeepSpeedStrategy:
        """
        Create a DeepSpeed trainer for instruction fine-tuning.
        
        Returns:
            Configured InstructionFabricDeepSpeedStrategy
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DeepSpeed trainer for instruction fine-tuning")
        return InstructionFabricDeepSpeedStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_ddp_trainer(self) -> InstructionFabricDDPStrategy:
        """
        Create a DDP trainer for instruction fine-tuning.
        
        Returns:
            Configured InstructionFabricDDPStrategy
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DDP trainer for instruction fine-tuning")
        return InstructionFabricDDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_dataparallel_trainer(self) -> InstructionFabricDataParallelStrategy:
        """
        Create a DataParallel trainer for instruction fine-tuning.
        
        Returns:
            Configured InstructionFabricDataParallelStrategy
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DataParallel trainer for instruction fine-tuning")
        return InstructionFabricDataParallelStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
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
                "Dataset is not loaded. The InstructionOrchestrator should load "
                "the dataset before delegating to the framework-specific orchestrator."
            )
    
    def _process_dataset(self) -> None:
        """
        Process the dataset for instruction fine-tuning.
        
        This method should not be called directly on the framework-specific orchestrator.
        Dataset processing should be handled by the parent InstructionOrchestrator.
        
        Raises:
            NotImplementedError: Always, as this method should not be called
        """
        raise NotImplementedError(
            "Dataset processing should be handled by the InstructionOrchestrator, "
            "not the framework-specific orchestrator."
        )
