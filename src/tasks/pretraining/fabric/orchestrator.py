"""
Fabric-based pretraining orchestrator.

This module provides a Fabric-based implementation of the pretraining orchestrator,
handling the setup and execution of distributed training using Lightning Fabric strategies.
"""

import logging
from typing import Any, Dict, List, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

# Import the abstract fabric orchestrator
from src.abstract_tasks.training.fabric.orchestrator import FabricOrchestrator

# Import pretraining-specific trainers
from src.tasks.pretraining.fabric.trainers import (
    PretrainingFabricFSDPTrainer,
    PretrainingFabricDeepSpeedTrainer,
    PretrainingFabricDDPTrainer,
    PretrainingFabricDataParallelTrainer,
)

logger = logging.getLogger(__name__)


class PretrainingFabricOrchestrator(FabricOrchestrator):
    """
    Fabric-based orchestrator for pretraining tasks.
    
    This class implements the abstract fabric orchestrator for pretraining,
    providing concrete implementations of the trainer factory methods.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the pretraining fabric orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _create_fsdp_trainer(self) -> PretrainingFabricFSDPTrainer:
        """
        Create an FSDP trainer for pretraining.
        
        Returns:
            Configured PretrainingFabricFSDPTrainer
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric FSDP trainer for pretraining")
        return PretrainingFSDPTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_deepspeed_trainer(self) -> PretrainingFabricDeepSpeedTrainer:
        """
        Create a DeepSpeed trainer for pretraining.
        
        Returns:
            Configured PretrainingFabricDeepSpeedTrainer
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DeepSpeed trainer for pretraining")
        return PretrainingDeepSpeedTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_ddp_trainer(self) -> PretrainingFabricDDPTrainer:
        """
        Create a DDP trainer for pretraining.
        
        Returns:
            Configured PretrainingFabricDDPTrainer
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DDP trainer for pretraining")
        return PretrainingDDPTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            dataset=self.processed_dataset,
            cli_logger=logger,
        )
    
    def _create_dataparallel_trainer(self) -> PretrainingFabricDataParallelTrainer:
        """
        Create a DataParallel trainer for pretraining.
        
        Returns:
            Configured PretrainingFabricDataParallelTrainer
        """
        self._ensure_dataset_loaded()
        
        logger.info("Creating Fabric DataParallel trainer for pretraining")
        return PretrainingDataParallelTrainer(
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
                "Dataset is not loaded. The PretrainingOrchestrator should load "
                "the dataset before delegating to the framework-specific orchestrator."
            )
    
    def _process_dataset(self) -> None:
        """
        Process the dataset for pretraining.
        
        This method should not be called directly on the framework-specific orchestrator.
        Dataset processing should be handled by the parent PretrainingOrchestrator.
        
        Raises:
            NotImplementedError: Always, as this method should not be called
        """
        raise NotImplementedError(
            "Dataset processing should be handled by the PretrainingOrchestrator, "
            "not the framework-specific orchestrator."
        )
