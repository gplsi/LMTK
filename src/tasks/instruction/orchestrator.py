"""
Instruction Orchestrator Module

This module provides the InstructionOrchestrator class, which handles the orchestration
of the instruction-following process for language models. The workflow includes validating configuration,
loading datasets, integrating language model templates with specific instructions, and processing
the datasets for training or inference.
"""

from typing import Dict, Any, Optional, Union
from box import Box
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from src.utils.logging import get_logger, VerboseLevel
from src.utils.orchestrator import BaseOrchestrator
from src.tasks.instruction.instructions.manager import InstructionManager
from src.tasks.instruction.instructions.datasets import DatasetHandler
from src.tasks.instruction.templates.composer import PromptComposer
from src.tasks.instruction.templates.base import PromptStyle
from utils import inherit_init_params


@inherit_init_params
class InstructionOrchestrator(BaseOrchestrator):
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
        if not self.config.model:
            raise ValueError("Model configuration must be provided")
        if not self.config.model.tokenizer:
            raise ValueError("Tokenizer name must be provided")
        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")
        if not self.config.output or not self.config.output.path:
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
            trust_remote_code=True
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

    def execute(self) -> None:
        """
        Execute the complete instruction workflow.

        The process is carried out in the following steps:
          1. Validate instruction configuration.
          2. Set up tokenizer and prompt composer.
          3. Load the dataset.
          4. Process the dataset with instructions and templates.
          5. Save the processed dataset to disk.

        Logging is used throughout the process to provide debugging information.
        In the event of an error, a log message is produced and the exception is re-raised.

        Raises:
            Exception: Propagates any exceptions encountered during the workflow execution.
        """
        self.logger.info("Starting instruction workflow")
        try:
            # 1. Validate configuration
            self.validate_config()
            
            # 2. Set up tokenizer and composer
            self.setup_tokenizer_and_composer()
            
            # 3. Load dataset
            dataset = self.load_dataset()
            
            # 4. Process dataset with instructions
            processed_dataset = self.process_dataset(dataset)
            
            # 5. Save results
            self.logger.info(f"Saving processed dataset to {self.config.output.path}")
            self.storage.save_to_disk(processed_dataset, self.config.output.path)
            
            self.logger.info("Instruction workflow completed successfully")
            
        except Exception as e:
            self.logger.error(f"Instruction workflow failed: {str(e)}")
            raise