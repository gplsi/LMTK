"""
Instruction Orchestrator Module

This module provides the InstructionOrchestrator class, which handles the orchestration
of the instruction-following process for language models. The workflow includes validating configuration,
loading datasets, integrating language model templates with specific instructions, and processing
the datasets for training or inference.
"""

from typing import Dict, Any, Optional, Union
import importlib
import logging
from pathlib import Path

from box import Box
from datasets import Dataset as HFDataset, DatasetDict
import torch
from transformers import AutoTokenizer

from src.utils.logging import get_logger, VerboseLevel
from src.abstract_tasks.training.orchestrator import TrainingOrchestrator
from src.tasks.instruction.instructions.manager import InstructionManager
from src.tasks.instruction.instructions.datasets import DatasetHandler
from src.tasks.instruction.templates.composer import PromptComposer
from src.tasks.instruction.templates.base import PromptStyle
from utils import inherit_init_params


@inherit_init_params
class InstructionOrchestrator(TrainingOrchestrator):
    """
    Orchestrates the instruction-following workflow.

    This class handles the complete process of integrating language model templates with
    specific instructions and datasets. It validates the provided configuration, loads
    the dataset, processes it with appropriate instructions, and prepares it for training
    or inference.
    """

    def __init__(self, config: Box) -> None:
        """
        Initialize the InstructionOrchestrator.

        Args:
            config: Configuration object containing instruction task parameters.
        """
        super().__init__(config)
        self.instruction_manager = None
        self.tokenizer = None
        self.composer = None
        self.logger = get_logger(__name__, VerboseLevel.INFO)
        self.processed_dataset = None

    def validate_config(self) -> None:
        """
        Validate the instruction configuration.

        This method ensures that the essential parts of the configuration are provided:
          - Model configuration including tokenizer name
          - Dataset configuration
          - Output configuration including a valid path

        Raises:
            ValueError: If any required configuration element is missing.
        """
        if not hasattr(self.config, "model") or not self.config.model:
            raise ValueError("Model configuration must be provided")
        if not hasattr(self.config.model, "tokenizer") or not self.config.model.tokenizer:
            raise ValueError("Tokenizer name must be provided")
        if not hasattr(self.config, "dataset") or not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")
        if not hasattr(self.config, "output") or not self.config.output or not self.config.output.path:
            raise ValueError("Output path must be provided")

    def setup_tokenizer_and_composer(self) -> None:
        """
        Set up the tokenizer and prompt composer based on configuration.

        This method initializes the tokenizer and prompt composer with the
        appropriate configuration settings.
        """
        self.logger.info(f"Loading tokenizer: {self.config.model.tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.tokenizer,
            trust_remote_code=self.config.model.get("trust_remote_code", True)
        )

        # Set up prompt composer with configured styles or defaults
        system_style = None
        user_style = None
        assistant_style = None
        final_style = None

        # Get max length from config or use default
        max_length = self.config.model.get("max_length", 2048)
        truncation_side = self.config.model.get("truncation_side", "right")

        self.composer = PromptComposer(
            tokenizer=self.tokenizer,
            system_style=system_style,
            user_style=user_style,
            assistant_style=assistant_style,
            final_style=final_style,
            max_length=max_length,
            truncation_side=truncation_side
        )

        # Set up instruction manager with the composer
        model_name = self.config.model.get("name", None)
        self.instruction_manager = InstructionManager(
            composer=self.composer,
            model_name=model_name
        )

    def process_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Process the dataset with instructions and templates.

        This method applies the appropriate instructions to the dataset
        and formats it according to the model's template requirements.

        Args:
            dataset: The dataset to process.

        Returns:
            The processed dataset ready for training or inference.
        """
        # Get column mappings from config
        input_column = self.config.dataset.get("input_column", "input")
        instruction_column = self.config.dataset.get("instruction_column", None)
        label_column = self.config.dataset.get("label_column", "output")
        task_type_column = self.config.dataset.get("task_type_column", None)
        combined_column = self.config.dataset.get("combined_column", None)
        
        # Get task name if specified
        task_name = self.config.get("task_name", None)
        
        # Get output column names
        output_column = self.config.output.get("input_column", "processed_input")
        output_label_column = self.config.output.get("label_column", "processed_label")
        
        # Process the dataset
        self.logger.info("Processing dataset with instructions")
        return self.instruction_manager.process_dataset(
            dataset_source=dataset,
            instruction_column=instruction_column,
            input_column=input_column,
            label_column=label_column,
            task_type_column=task_type_column,
            combined_column=combined_column,
            task_name=task_name,
            output_column=output_column,
            output_label_column=output_label_column
        )

    def load_dataset(self) -> HFDataset:
        """
        Load the dataset according to the configuration.

        This method loads the dataset from the specified source and format,
        handling different dataset types (HuggingFace, JSON, etc.).

        Returns:
            The loaded dataset.

        Raises:
            ValueError: If the dataset configuration is invalid.
        """
        dataset_type = self.config.dataset.get("type", "")
        
        if dataset_type == "huggingface":
            # Load from HuggingFace
            dataset_name = self.config.dataset.get("name", "")
            if not dataset_name:
                raise ValueError("HuggingFace dataset name must be provided")
            
            config_name = self.config.dataset.get("config_name", None)
            split = self.config.dataset.get("split", "train")
            
            self.logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            return HFDataset.load_dataset(
                dataset_name,
                name=config_name,
                split=split
            )
        elif dataset_type == "json":
            # Load from JSON files
            files = self.config.dataset.get("files", {})
            if not files or not files.get("train"):
                raise ValueError("JSON dataset files must be provided")
            
            train_files = files.get("train")
            if isinstance(train_files, str):
                train_files = [train_files]
            
            self.logger.info(f"Loading JSON dataset from {len(train_files)} file(s)")
            return HFDataset.from_json(train_files)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def _process_dataset(self) -> None:
        """
        Process the dataset for instruction fine-tuning.

        This method implements the abstract method from TrainingOrchestrator
        to handle dataset processing for instruction fine-tuning.
        """
        self.logger.info("Processing dataset for instruction fine-tuning")
        
        # Validate configuration
        self.validate_config()
        
        # Set up tokenizer and composer
        self.setup_tokenizer_and_composer()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Process dataset with instructions
        processed_dataset = self.process_dataset(dataset)
        self.processed_dataset = processed_dataset
        
        # Save processed dataset
        output_path = self.config.output.path
        self.logger.info(f"Saving processed dataset to {output_path}")
        processed_dataset.save_to_disk(output_path)
        
        self.logger.info("Dataset processing completed successfully")
    
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
        
        try:
            # Dynamically import the framework-specific orchestrator
            module_path = f"src.tasks.instruction.{framework}.orchestrator"
            module = importlib.import_module(module_path)
            
            # Get the orchestrator class name based on the framework
            class_name = f"Instruction{framework.capitalize()}Orchestrator"
            orchestrator_class = getattr(module, class_name)
            
            # Create an instance of the orchestrator
            self.logger.info(f"Using {framework} framework for instruction fine-tuning")
            return orchestrator_class(self.config)
            
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load {framework} orchestrator: {str(e)}")
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
        Execute the instruction workflow.

        This method overrides the execute method from TrainingOrchestrator
        to add instruction-specific preprocessing before training.
        """
        self.logger.info("Starting instruction workflow")
        
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
                    
                    # Set up tokenizer and composer
                    self.setup_tokenizer_and_composer()
                    
                    # Load dataset
                    dataset = self.load_dataset()
                    
                    # Process dataset with instructions
                    self.processed_dataset = self.process_dataset(dataset)
                    
                    # Save processed dataset if configured
                    if self.config.output.get("save_processed", True):
                        output_path = self.config.output.path
                        self.logger.info(f"Saving processed dataset to {output_path}")
                        self.processed_dataset.save_to_disk(output_path)
                
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
            
            self.logger.info("Instruction workflow completed successfully")
            
        except Exception as e:
            self.logger.error(f"Instruction workflow failed: {str(e)}")
            raise