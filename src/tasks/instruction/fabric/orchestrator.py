"""
Instruction-specific implementation of the Fabric orchestrator.

This module provides a concrete implementation of the abstract Fabric orchestrator
for instruction fine-tuning tasks.
"""

import logging
from typing import Any, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.fabric.orchestrator import FabricOrchestrator
from src.tasks.instruction.fabric.strategies import (
    InstructionFabricFSDPStrategy,
    InstructionFabricDeepSpeedStrategy,
    InstructionFabricDDPStrategy,
    InstructionFabricDataParallelStrategy,
)

logger = logging.getLogger(__name__)


class InstructionFabricOrchestrator(FabricOrchestrator):
    """
    Instruction-specific implementation of the Fabric orchestrator.
    
    This class extends the base Fabric orchestrator to implement instruction-specific
    functionality, such as dataset processing and strategy creation.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the instruction Fabric orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _create_fsdp_strategy(self) -> InstructionFabricFSDPStrategy:
        """
        Create an FSDP strategy for instruction fine-tuning.
        
        Returns:
            An instance of the instruction FSDP strategy
        """
        self._ensure_dataset_loaded()
        logger.info("Creating Fabric FSDP strategy for instruction fine-tuning")
        return InstructionFabricFSDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_deepspeed_strategy(self) -> InstructionFabricDeepSpeedStrategy:
        """
        Create a DeepSpeed strategy for instruction fine-tuning.
        
        Returns:
            An instance of the instruction DeepSpeed strategy
        """
        self._ensure_dataset_loaded()
        logger.info("Creating Fabric DeepSpeed strategy for instruction fine-tuning")
        return InstructionFabricDeepSpeedStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_ddp_strategy(self) -> InstructionFabricDDPStrategy:
        """
        Create a DDP strategy for instruction fine-tuning.
        
        Returns:
            An instance of the instruction DDP strategy
        """
        self._ensure_dataset_loaded()
        logger.info("Creating Fabric DDP strategy for instruction fine-tuning")
        return InstructionFabricDDPStrategy(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_dataparallel_strategy(self) -> InstructionFabricDataParallelStrategy:
        """
        Create a DataParallel strategy for instruction fine-tuning.
        
        Returns:
            An instance of the instruction DataParallel strategy
        """
        self._ensure_dataset_loaded()
        logger.info("Creating Fabric DataParallel strategy for instruction fine-tuning")
        return InstructionFabricDataParallelStrategy(
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
                "Dataset is not loaded. The InstructionOrchestrator should load "
                "the dataset before delegating to the framework-specific orchestrator."
            )
    
    def _process_dataset(self) -> Union[HFDataset, DatasetDict]:
        """
        Process the dataset for instruction fine-tuning.
        
        Returns:
            The processed dataset
        """
        # Implement instruction-specific dataset processing
        # This would typically include tokenization, formatting, etc.
        pass
