"""
Tokenization Instruction Orchestrator Module

This module provides the TokenizationInstructionOrchestrator class, which handles the orchestration 
of the tokenization process for instruction datasets. The workflow includes validating configuration, 
loading multi-language datasets, tokenizing them using instruction-specific preprocessing with 
chat templates, and saving the tokenized data with train/validation splits.
"""

from box import Box
from datasets import DatasetDict, concatenate_datasets
from datasets import Dataset as HFDataset
from src.utils.logging import get_logger, set_logger_level
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from src.utils.orchestrator import BaseOrchestrator
from src.tasks.tokenization_instruction.src.prepare_data.preprocess import tokenizer_dataset_multiTurn
from transformers import AutoTokenizer
import os
import json
from utils import inherit_init_params

@inherit_init_params
class TokenizationInstructionOrchestrator(BaseOrchestrator):
    """
    Orchestrates the instruction tokenization workflow with multi-language support.

    This class handles the complete process of tokenizing instruction datasets across multiple languages.
    It validates the provided configuration, loads multi-language datasets, processes them using
    instruction-specific preprocessing with chat templates, and saves train/validation splits.
    """

    def validate_config(self) -> None:
        """
        Validate the tokenization instruction configuration.

        This method ensures that the essential parts of the configuration are provided:
          - Tokenizer configuration and the tokenizer name.
          - Languages list for multi-language processing.
          - Data path configuration.
          - Output configuration including valid paths for train and validation sets.

        Raises:
            ValueError: If any required configuration element is missing.
        """
        if not self.config.tokenizer:
            raise ValueError("Tokenizer configuration must be provided")
        if not self.config.tokenizer.tokenizer_name:
            raise ValueError("Tokenizer tokenizer_name must be provided")
        if not self.config.get("languages"):
            raise ValueError("Languages configuration must be provided")
        if not self.config.get("dataPath"):
            raise ValueError("Data path must be provided")
        if not self.config.output:
            raise ValueError("Output configuration must be provided")
        if not self.config.output.get("training_tokenized"):
            raise ValueError("Training output path must be provided")
        if not self.config.output.get("eval_tokenized"):
            raise ValueError("Evaluation output path must be provided")

    def process_multi_language_datasets(self) -> tuple[HFDataset, HFDataset]:
        """
        Process multi-language instruction datasets and return train/validation splits.

        This method processes datasets for each configured language using the instruction-specific
        tokenization with chat templates. It concatenates all language datasets and splits them
        into training and validation sets.

        Returns:
            tuple[HFDataset, HFDataset]: Training and validation datasets.
        """
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_name)
        
        # Log the chat template if available
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            self.logger.info(f"Using chat template: {tokenizer.chat_template}")
        else:
            self.logger.warning("No chat template found in tokenizer")

        languages = self.config.languages
        self.logger.info(f"Processing {len(languages)} languages: {languages}")        # Get max sequence length and other config parameters
        max_seq_length = self.config.tokenizer.get("context_length", 
                                                   self.config.train_args.get("max_seq_length", 2048))
        train_size = self.config.get("train_size", 0.7)
        test_size = 1.0 - train_size  # Convert train_size to test_size for split
        random_seed = self.config.get("random_seed", 42)

        train_dataset = None
        val_dataset = None

        for lang in languages:
            self.logger.info(f"Processing language: {lang}")
            
            # Construct the path for this language
            lang_path = os.path.join(self.config.dataPath, lang)
            
            if not os.path.exists(lang_path):
                self.logger.warning(f"Path does not exist for language {lang}: {lang_path}")
                continue

            # Process the dataset for this language using instruction tokenization
            try:
                tokenized_data = tokenizer_dataset_multiTurn(
                    lang_path, 
                    tokenizer, 
                    self.config, 
                    max_seq_length
                )
                
                self.logger.info(f"Tokenized {len(tokenized_data)} examples for language {lang}")

                # Split the dataset into train and validation sets
                train_data = tokenized_data.train_test_split(
                    test_size=test_size, 
                    seed=random_seed
                )

                # Concatenate with existing datasets
                if train_dataset is None and val_dataset is None:
                    train_dataset = train_data['train']
                    val_dataset = train_data['test']
                else:
                    train_dataset = concatenate_datasets([train_dataset, train_data['train']])
                    val_dataset = concatenate_datasets([val_dataset, train_data['test']])
                    
                self.logger.info(f"Accumulated train size: {len(train_dataset)}, val size: {len(val_dataset)}")
                
            except Exception as e:
                self.logger.error(f"Failed to process language {lang}: {str(e)}")
                continue

        if train_dataset is None or val_dataset is None:
            raise ValueError("No datasets were successfully processed")

        self.logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        return train_dataset, val_dataset

    def execute(self) -> None:
        """
        Execute the complete instruction tokenization workflow.

        The process is carried out in the following steps:
          1. Validate tokenization instruction configuration.
          2. Process multi-language datasets using instruction-specific tokenization.
          3. Save the training and validation datasets to disk.

        Logging is used throughout the process to provide debugging information. In the event 
        of an error, a log message is produced and the exception is re-raised.

        Raises:
            Exception: Propagates any exceptions encountered during the workflow execution.
        """
        
        self.logger.info("Starting instruction tokenization workflow")
        try:
            # 1. Validate configuration
            self.validate_config()

            # 2. Process multi-language datasets 
            train_dataset, val_dataset = self.process_multi_language_datasets()            # 3. Save results
            self.logger.info(f"Saving training dataset to: {self.config.output.train_path}")
            train_dataset.save_to_disk(self.config.output.train_path)
            
            self.logger.info(f"Saving validation dataset to: {self.config.output.validation_path}")
            val_dataset.save_to_disk(self.config.output.validation_path)

            self.logger.info("Instruction tokenization workflow completed successfully")

        except Exception as e:
            self.logger.error(f"Instruction tokenization workflow failed: {str(e)}")
            raise
