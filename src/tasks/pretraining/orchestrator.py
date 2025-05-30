"""
Module for continual pretraining orchestration, supporting multiple distributed training frameworks.

This module provides the PretrainingOrchestrator class which is responsible for managing
the pretraining workflow using different distributed strategies and frameworks.
It handles device detection, configuration validation, dataset loading, and training execution.
"""

from typing import Dict, Any, Optional, Union, Type
from box import Box
from datasets import Dataset as HFDataset, DatasetDict
import importlib
import logging

from src.utils.logging import get_logger, VerboseLevel
from src.utils.dataset import DatasetStorage
from src.abstract_tasks.training.orchestrator import TrainingOrchestrator


class PretrainingOrchestrator(TrainingOrchestrator):
    """
    Orchestrates the pretraining workflow with support for multiple frameworks.
    
    This class manages the entire lifecycle of pretraining, including device detection,
    configuration validation, dataset loading, and execution of various distributed 
    training strategies across different frameworks (Fabric, PyTorch native, etc.).
    """

    def __init__(self, config: Box) -> None:
        """
        Initialize the PretrainingOrchestrator with the given configuration.
        
        It performs the following steps:
          - Calls the superclass initializer with the provided configuration
          - Sets up logging and dataset storage
        
        Parameters:
            config (Box): Configuration object containing training parameters.
        """
        super().__init__(config)
        self.logger = get_logger(__name__, VerboseLevel.INFO)
        self.processed_dataset = None
        self._framework_orchestrator = None

    def validate_config(self) -> None:
        """
        Validate pretraining configuration.
        
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
        
        # Validate framework configuration
        if not hasattr(self.config, "framework"):
            self.logger.warning("No framework specified, defaulting to 'fabric'")
            self.config.framework = "fabric"

    def _validate_dataset_config(self) -> None:
        """
        Validate the dataset configuration.
        
        This method ensures that the dataset configuration is properly defined.
        
        Raises:
            ValueError: If the dataset configuration is missing.
        """
        if not hasattr(self.config, "dataset") or not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")

    def load_dataset(self) -> Union[HFDataset, DatasetDict]:
        """
        Load the dataset based on the provided configuration.
        
        This method performs the following:
          - Instantiates a DatasetStorage with the appropriate verbosity level.
          - Validates that the dataset configuration meets the required settings.
          - Loads the dataset from a local disk if specified.
          - Wraps the dataset in a DatasetDict for maintainability if necessary.
        
        Returns:
            Union[HFDataset, DatasetDict]: The loaded dataset in HuggingFace datasets format.
            
        Raises:
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

    def _get_framework_orchestrator(self) -> TrainingOrchestrator:
        """
        Get the appropriate framework-specific orchestrator based on configuration.
        
        This method dynamically imports and instantiates the appropriate framework-specific
        orchestrator based on the configuration.
        
        Returns:
            TrainingOrchestrator: A framework-specific orchestrator instance.
            
        Raises:
            ImportError: If the specified framework module cannot be imported.
            AttributeError: If the specified orchestrator class cannot be found.
        """
        if self._framework_orchestrator is not None:
            return self._framework_orchestrator
        
        framework = getattr(self.config, "framework", "fabric").lower()
        
        try:
            # Dynamically import the framework-specific orchestrator
            module_path = f"src.tasks.pretraining.{framework}.orchestrator"
            self.logger.info(f"Attempting to import orchestrator from {module_path}")
            
            module = importlib.import_module(module_path)
            
            # Get the orchestrator class - handle both naming conventions
            class_names = [
                f"Pretraining{framework.capitalize()}Orchestrator",  # PretrainingFabricOrchestrator
                f"{framework.capitalize()}Orchestrator",             # FabricOrchestrator
            ]
            
            orchestrator_class = None
            for class_name in class_names:
                self.logger.debug(f"Looking for orchestrator class: {class_name}")
                if hasattr(module, class_name):
                    orchestrator_class = getattr(module, class_name)
                    self.logger.info(f"Found orchestrator class: {class_name}")
                    break
            
            if orchestrator_class is None:
                available_classes = [name for name in dir(module) if not name.startswith('_') and name.endswith('Orchestrator')]
                raise AttributeError(f"Could not find orchestrator class for framework '{framework}'. Available classes: {available_classes}")
            
            # Create an instance of the orchestrator
            self._framework_orchestrator = orchestrator_class(self.config)
            
            # Pass the processed dataset to the framework orchestrator
            if hasattr(self._framework_orchestrator, "processed_dataset"):
                self.logger.info(f"Passing processed dataset to {framework} orchestrator")
                self._framework_orchestrator.processed_dataset = self.processed_dataset
            
            return self._framework_orchestrator
            
        except ImportError as e:
            self.logger.error(f"Failed to import framework module '{framework}': {str(e)}")
            if framework == "fabric":
                self.logger.error("The fabric framework is required but could not be imported.")
            raise ImportError(f"Framework '{framework}' is not supported. Please check your configuration.") from e
        except AttributeError as e:
            self.logger.error(f"Failed to find orchestrator class for framework '{framework}': {str(e)}")
            raise AttributeError(f"Orchestrator for framework '{framework}' is not implemented.") from e

    def _create_trainer(self, strategy: str) -> Any:
        """
        Create a trainer for the specified strategy.
        
        This method delegates to the framework-specific orchestrator to create
        the appropriate trainer for the specified strategy.
        
        Args:
            strategy: Name of the training strategy
            
        Returns:
            A configured trainer for the specified strategy
        """
        framework_orchestrator = self._get_framework_orchestrator()
        return framework_orchestrator._create_trainer(strategy)

    def execute(self) -> None:
        """
        Execute the complete pretraining pipeline.
        
        This method orchestrates the entire pretraining workflow, including
        dataset processing, framework selection, and training execution.
        
        The execution flow includes:
          1. Validating configuration settings.
          2. Loading the dataset as per the specified source.
          3. Selecting the appropriate framework orchestrator.
          4. Delegating to the framework orchestrator for training.
        
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
                
                # Get the framework-specific orchestrator
                framework = getattr(self.config, "framework", "fabric").lower()
                self.logger.info(f"Using '{framework}' framework for training")
                
                # Get the framework-specific orchestrator
                framework_orchestrator = self._get_framework_orchestrator()
                
                # Ensure the processed dataset is passed to the framework orchestrator
                if hasattr(framework_orchestrator, "processed_dataset") and self.processed_dataset is not None:
                    framework_orchestrator.processed_dataset = self.processed_dataset
                
                # Determine the training strategy
                strategy = self._get_training_strategy()
                self.logger.info(f"Using '{strategy}' training strategy")
                
                # Train using the framework-specific orchestrator's train method directly
                trainer = framework_orchestrator._create_trainer(strategy)
                
                # Set up the trainer
                trainer.setup()
                
                # Start training
                trainer.train()
            else:
                raise ValueError(f"Unknown execution mode: {mode}")
            
            self.logger.info("Pretraining pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pretraining pipeline failed: {str(e)}")
            raise
