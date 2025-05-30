"""
Base task orchestrator.

This module provides a base implementation for task-specific orchestrators,
handling common functionality such as framework selection and execution flow.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Union

from box import Box
from datasets import Dataset as HFDataset, DatasetDict

from src.abstract_tasks.training.orchestrator import TrainingOrchestrator

logger = logging.getLogger(__name__)


class BaseTaskOrchestrator(TrainingOrchestrator):
    """
    Base orchestrator for specific tasks.
    
    This class provides common functionality for task-specific orchestrators,
    such as framework selection, strategy determination, and execution flow.
    Task-specific orchestrators should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, config: Box) -> None:
        """
        Initialize the base task orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.processed_dataset = None
    
    def _get_framework_orchestrator(self):
        """
        Get the framework-specific orchestrator based on the configuration.
        
        Returns:
            An instance of the framework-specific orchestrator
        
        Raises:
            ValueError: If the specified framework is not supported
        """
        # Get the framework from the configuration, defaulting to "fabric"
        framework = getattr(self.config, "framework", "fabric").lower()
        task_name = self.__class__.__name__.replace("Orchestrator", "").lower()
        
        try:
            # Dynamically import the framework-specific orchestrator
            module_path = f"src.tasks.{task_name}.{framework}.orchestrator"
            module = importlib.import_module(module_path)
            
            # Get the orchestrator class name based on the framework and task
            class_name = f"{task_name.capitalize()}{framework.capitalize()}Orchestrator"
            orchestrator_class = getattr(module, class_name)
            
            # Create an instance of the orchestrator
            logger.info(f"Using {framework} framework for {task_name}")
            return orchestrator_class(self.config)
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load {framework} orchestrator for {task_name}: {str(e)}")
            raise ValueError(f"Unsupported framework: {framework}. Please check your configuration.")
    
    def _get_training_strategy(self) -> str:
        """
        Determine the training strategy based on the configuration.
        
        Returns:
            The name of the training strategy to use
        """
        # Get the strategy from the configuration, defaulting to "fsdp"
        strategy = getattr(self.config, "strategy", "fsdp").lower()
        
        # Map strategy names to standardized values
        strategy_mapping = {
            "fsdp": "fsdp",
            "deepspeed": "deepspeed",
            "ddp": "ddp",
            "dataparallel": "dataparallel",
            "dp": "dataparallel",
        }
        
        # Return the standardized strategy name
        return strategy_mapping.get(strategy, strategy)
    
    def execute(self) -> None:
        """
        Execute the task workflow.
        
        This method orchestrates the complete workflow, including dataset processing,
        framework selection, and training execution.
        """
        task_name = self.__class__.__name__.replace("Orchestrator", "")
        logger.info(f"Starting {task_name} workflow")
        
        try:
            # Determine execution mode
            mode = getattr(self.config, "mode", "train")
            
            if mode == "process_dataset":
                # Process dataset mode
                self._process_dataset()
            elif mode == "train":
                # First process the dataset if needed
                if self.config.get("preprocess_dataset", True):
                    self._process_dataset()
                
                # Get the framework-specific orchestrator
                framework_orchestrator = self._get_framework_orchestrator()
                
                # Ensure the processed dataset is passed to the framework orchestrator
                if hasattr(framework_orchestrator, "processed_dataset") and self.processed_dataset is not None:
                    framework_orchestrator.processed_dataset = self.processed_dataset
                
                # Determine the training strategy
                strategy = self._get_training_strategy()
                logger.info(f"Using '{strategy}' training strategy")
                
                # Train using the framework-specific orchestrator's train method directly
                trainer = framework_orchestrator._create_trainer(strategy)
                
                # Set up the trainer
                trainer.setup()
                
                # Start training
                trainer.train()
            else:
                raise ValueError(f"Unknown execution mode: {mode}")
            
            logger.info(f"{task_name} workflow completed successfully")
            
        except Exception as e:
            logger.error(f"{task_name} workflow failed: {str(e)}")
            raise
