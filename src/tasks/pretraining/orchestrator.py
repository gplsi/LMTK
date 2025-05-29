"""
Module for continual pretraining orchestration, including various parallelization strategies.

This module provides the ContinualOrchestrator class which is responsible for managing
the pretraining workflow using different distributed strategies such as FSDP, DDP, DP, and DeepSpeed.
It handles device detection, configuration validation, dataset loading, and training execution.
"""

from typing import Dict, Any, Optional, Union
from box import Box
from datasets import Dataset as HFDataset, DatasetDict
import torch

from src.utils.logging import get_logger, VerboseLevel
from src.utils.dataset import DatasetStorage
from src.abstract_tasks.training.orchestrator import TrainingOrchestrator
from src.tasks.pretraining.fabric.trainer import PretrainingTrainer
from utils import inherit_init_params


@inherit_init_params
class ContinualOrchestrator(TrainingOrchestrator):
    """
    Orchestrates the continual pretraining workflow.
    
    This class manages the entire lifecycle of continual pretraining,
    including device detection, configuration validation, dataset loading,
    and execution of various distributed training strategies.
    """

    def __init__(self, config: Box) -> None:
        """
        Initialize the ContinualOrchestrator with the given configuration.
        
        It performs the following steps:
          - Calls the superclass initializer with the provided configuration
          - Sets up logging and dataset storage
        
        Parameters:
            config (Box): Configuration object containing training parameters.
        """
        super().__init__(config)
        self.logger = get_logger(__name__, VerboseLevel.INFO)
        self.processed_dataset = None

    def validate_config(self) -> None:
        """
        Validate continual configuration.
        
        This method verifies that the configuration settings
        are properly defined and sufficient for running the pretraining tasks.
        
        Raises:
            ValueError: If required configuration elements are missing.
        """
        if not hasattr(self.config, "model") or not self.config.model:
            raise ValueError("Model configuration must be provided")
        if not hasattr(self.config.model, "name") or not self.config.model.name:
            raise ValueError("Model name must be provided")
        if not hasattr(self.config, "dataset") or not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")
        if not hasattr(self.config.dataset, "source") or not self.config.dataset.source:
            raise ValueError("Dataset source must be provided")
        if not hasattr(self.config, "output_dir") or not self.config.output_dir:
            raise ValueError("Output directory must be provided")

    def _validate_dataset_config(self) -> None:
        """
        Validate the dataset configuration.
        
        This method ensures that the dataset configuration is properly defined.
        
        Raises:
            ValueError: If the dataset configuration is missing.
        """
        if not hasattr(self.config, "dataset") or not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")

    def load_dataset(self) -> HFDataset:
        """
        Load the dataset based on the provided configuration.
        
        This method performs the following:
          - Instantiates a DatasetStorage with the appropriate verbosity level.
          - Validates that the dataset configuration meets the required settings.
          - Loads the dataset from a local disk if specified.
          - Wraps the dataset in a DatasetDict for maintainability if necessary.
        
        Returns:
            HFDataset: The loaded dataset in HuggingFace datasets format, potentially wrapped in a DatasetDict.
            
        Raises:
            NotImplementedError: If the dataset source is 'huggingface', as it is not implemented.
            ValueError: If the dataset source is invalid.
        """
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            )
        )
        
        self._validate_dataset_config()
        
        if self.config.dataset.source == "local":
            self.logger.info(f"Loading dataset from path '{self.config.dataset.nameOrPath}'")
            
            dataset = dataset_handler.load_from_disk(self.config.dataset.nameOrPath)
            # Ensure dataset is wrapped in a DatasetDict for consistency
            if isinstance(dataset, HFDataset):
                dataset = DatasetDict({"train": dataset})
            
            self.processed_dataset = dataset
            return dataset
        
        elif self.config.dataset.source == "huggingface":
            self.logger.info(f"Loading dataset from HuggingFace: '{self.config.dataset.nameOrPath}'")
            
            # Get dataset name and config
            dataset_name = self.config.dataset.nameOrPath
            config_name = self.config.dataset.get("config_name", None)
            split = self.config.dataset.get("split", "train")
            
            # Load dataset from HuggingFace
            dataset = HFDataset.load_dataset(
                dataset_name,
                name=config_name,
                split=split
            )
            
            # Ensure dataset is wrapped in a DatasetDict for consistency
            if isinstance(dataset, HFDataset):
                dataset = DatasetDict({"train": dataset})
            
            self.processed_dataset = dataset
            return dataset
        
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")

    def _process_dataset(self) -> None:
        """
        Process the dataset for pretraining.
        
        This method implements the abstract method from TrainingOrchestrator
        to handle dataset processing for pretraining.
        """
        self.logger.info("Processing dataset for pretraining")
        
        # Validate configuration
        self.validate_config()
        
        # Load dataset
        dataset = self.load_dataset()
        self.processed_dataset = dataset
        
        # Save processed dataset if configured
        if hasattr(self.config, "output") and self.config.output.get("save_processed", False):
            output_path = self.config.output.path
            self.logger.info(f"Saving processed dataset to {output_path}")
            dataset.save_to_disk(output_path)
        
        self.logger.info("Dataset processing completed successfully")

    def _setup_fsdp(self) -> PretrainingTrainer:
        """
        Set up the FSDP trainer for pretraining.
        
        This method overrides the parent method to use the PretrainingTrainer
        instead of the generic FSDP trainer.
        
        Returns:
            Configured PretrainingTrainer with FSDP strategy
        """
        self.logger.info("Setting up FSDP trainer for pretraining")
        
        return PretrainingTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=self.logger,
            dataset=self.processed_dataset,
        )
    
    def _setup_deepspeed(self) -> PretrainingTrainer:
        """
        Set up the DeepSpeed trainer for pretraining.
        
        This method overrides the parent method to use the PretrainingTrainer
        instead of the generic DeepSpeed trainer.
        
        Returns:
            Configured PretrainingTrainer with DeepSpeed strategy
        """
        self.logger.info("Setting up DeepSpeed trainer for pretraining")
        
        return PretrainingTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=self.logger,
            dataset=self.processed_dataset,
        )
    
    def _setup_ddp(self) -> PretrainingTrainer:
        """
        Set up the DDP trainer for pretraining.
        
        This method overrides the parent method to use the PretrainingTrainer
        instead of the generic DDP trainer.
        
        Returns:
            Configured PretrainingTrainer with DDP strategy
        """
        self.logger.info("Setting up DDP trainer for pretraining")
        
        return PretrainingTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=self.logger,
            dataset=self.processed_dataset,
        )
    
    def _setup_dp(self) -> PretrainingTrainer:
        """
        Set up the DataParallel trainer for pretraining.
        
        This method overrides the parent method to use the PretrainingTrainer
        instead of the generic DataParallel trainer.
        
        Returns:
            Configured PretrainingTrainer with DataParallel strategy
        """
        self.logger.info("Setting up DataParallel trainer for pretraining")
        
        return PretrainingTrainer(
            config=self.config,
            devices=self.devices,
            output_dir=self.output_dir,
            cli_logger=self.logger,
            dataset=self.processed_dataset,
        )

    def execute(self) -> None:
        """
        Execute the complete continual pretraining pipeline.
        
        This method overrides the execute method from TrainingOrchestrator
        to add pretraining-specific preprocessing before training.
        
        The execution flow includes:
          1. Validating configuration settings.
          2. Loading the dataset as per the specified source.
          3. Determining the parallelization strategy (FSDP, DDP, DP, DeepSpeed).
          4. Running the appropriate training task based on the strategy.
        
        Logs are generated at every major step. In case of errors, the exception is logged and re-raised.
        """
        self.logger.info("Starting pretraining pipeline")
        
        try:
            # Determine execution mode
            mode = getattr(self.config, "mode", "train")
            
            if mode == "process_dataset":
                # Process dataset mode
                self._process_dataset()
            elif mode == "train":
                # First process the dataset if needed
                if self.config.get("preprocess_dataset", True):
                    # Validate configuration
                    self.validate_config()
                    
                    # Load dataset
                    self.processed_dataset = self.load_dataset()
                    
                    # Save processed dataset if configured
                    if hasattr(self.config, "output") and self.config.output.get("save_processed", False):
                        output_path = self.config.output.path
                        self.logger.info(f"Saving processed dataset to {output_path}")
                        self.processed_dataset.save_to_disk(output_path)
                
                # Call parent method to handle training
                super().execute()
            else:
                raise ValueError(f"Unknown execution mode: {mode}")
            
            self.logger.info("Pretraining pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pretraining pipeline failed: {str(e)}")
            raise
