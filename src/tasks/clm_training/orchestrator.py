"""
Module for continual pretraining orchestration, including various parallelization strategies.

This module provides the ContinualOrchestrator class which is responsible for managing
the pretraining workflow using different distributed strategies such as FSDP, DDP, DP, and DeepSpeed.
It handles device detection, configuration validation, dataset loading, and training execution.
"""

# src/tasks/tokenization.py

from box import Box
from datasets import DatasetDict
import torch
from src.utils.logging import get_logger
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from src.tasks.clm_training.fabric.distributed import FSDP, DeepSpeed, DistributedDataParallel, DataParallel
from utils import inherit_init_params
from utils.orchestrator import BaseOrchestrator
from datasets import Dataset as HFDataset


class ContinualOrchestrator(BaseOrchestrator):
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
          - Gets the local rank from environment variables for distributed training
          - Calls the superclass initializer with the provided configuration and rank
          - Checks for CUDA availability and counts the available CUDA devices.
          - Logs the number of available CUDA devices. If none are found, a warning is 
            logged and training will fall back to using the CPU.
        
        Parameters:
            config (Box): Configuration object containing training parameters.
        """
        super().__init__(config)
        
        # get all devices available in torch so we can set them to torch modules
        if (torch.cuda.is_available()):
            self.devices = torch.cuda.device_count()
            self.logger.info(f"Found {self.devices} CUDA devices available for training")
            if self.devices > 0:
                self.logger.info(f"Found {self.devices} CUDA devices available for training")
                return
            
        self.logger.warning("No CUDA devices available for training. Training will be done on CPU")
        self.devices = "cpu"

    def validate_config(self) -> None:
        """
        Validate continual configuration.
        
        This method is intended to verify that the configuration settings
        are properly defined and sufficient for running the pretraining tasks.
        Currently, it is a placeholder for future validation implementation.
        
        Raises:
            NotImplementedError: If configuration validation logic is added and fails.
        """
        # TODO: Implement general configuration validation
        pass

    def fsdp(self, dataset: HFDataset) -> None:
        """
        Execute FSDP (Fully Sharded Data Parallel) continual pretraining task.
        
        This method initializes and sets up the FSDP trainer using the given dataset and configuration.
        It logs the start and finish of the FSDP training process.
        
        Parameters:
            dataset (HFDataset): The dataset to be used for pretraining.
        """
        self.logger.info("Starting FSDP continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = FSDP(
            devices=self.devices,
            config=self.config,
            dataset=dataset,
            checkpoint_path=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("FSDP training finished")
        
    def ddp(self, dataset: HFDataset) -> None:
        """
        Execute DDP (Distributed Data Parallel) continual pretraining task.
        
        This method sets up the trainer with DistributedDataParallel using the available devices.
        It logs the overall process of the DDP training.
        
        Parameters:
            dataset (HFDataset): The dataset to be used for pretraining.
        """        
        self.logger.info("Starting DDP continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DistributedDataParallel(
            devices=self.devices,
            config=self.config,
            dataset=dataset,
            checkpoint_path=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("DDP training finished")
        
    def dp(self, dataset: HFDataset) -> None:
        """
        Execute DP (Data Parallel) continual pretraining task.
        
        This method initializes and configures the DataParallel trainer using the provided dataset
        and configuration settings. It logs both the start and the completion of the DP training.
        
        Parameters:
            dataset (HFDataset): The dataset used for pretraining.
        """
        self.logger.info("Starting DP continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DataParallel(
            devices=self.devices,
            config=self.config,
            dataset=dataset,
            checkpoint_path=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("DP training finished")

    def deep_speed(self, dataset: HFDataset) -> None:
        """
        Execute DeepSpeed continual pretraining task.
        
        This method sets up the DeepSpeed trainer by using the chosen configuration,
        dataset, and available devices. It logs the commencement and the completion of the
        DeepSpeed training process.
        
        Parameters:
            dataset (HFDataset): The dataset to be used for pretraining.
        """
        self.logger.info("Starting Deep Speed continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DeepSpeed(
            devices=self.devices,
            config=self.config,
            dataset=dataset,
            checkpoint_path=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("Deep Speed training finished")


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
            # TODO: IMPROVE THIS FOR MAINTAINABILITY
            if isinstance(dataset, HFDataset):
                dataset = DatasetDict({"train": dataset})
            return dataset
        
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")

    def execute(self) -> None:
        """
        Execute the complete continual pretraining pipeline.
        
        The execution flow includes:
          1. Validating configuration settings.
          2. Loading the dataset as per the specified source.
          3. Determining the parallelization strategy (FSDP, DDP, DP, DeepSpeed).
          4. Running the appropriate training task based on the strategy.
        
        Logs are generated at every major step. In case of errors, the exception is logged and re-raised.
        """
        self.logger.info("Starting training pipeline")
        try:
            # Debug: Log configuration types that might cause issues
            config_keys_to_check = ['gradient_accumulation_steps', 'validate_after_k_steps', 'max_epochs', 'max_steps', 'batch_size', 'eval_batch_size']
            for key in config_keys_to_check:
                if hasattr(self.config, key):
                    value = getattr(self.config, key)
                    self.logger.debug(f"Orchestrator config {key}: value={value}, type={type(value)}")
            
            # 1. Validate configuration
            self.validate_config()
            
            # 2. Load dataset
            dataset = self.load_dataset()

            # Select specific continual method
            strategy = self.config.get("parallelization_strategy", "fsdp")
            if strategy == "fsdp":
                self.fsdp(dataset)
            elif strategy == "ddp":
                self.ddp(dataset)
            elif strategy == "deep_speed":
                self.deep_speed(dataset)
            elif strategy == "dp":
                self.dp(dataset)
            else:
                raise ValueError(f"Invalid parallelization strategy: {strategy}")
            
            self.logger.info("Continual Pretraining completed successfully")

        except Exception as e:
            import traceback
            self.logger.error(f"Continual Pretraining pipeline failed: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
