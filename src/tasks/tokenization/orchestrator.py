"""
Tokenization Orchestrator Module

This module provides the TokenizationOrchestrator class, which handles the orchestration 
of the tokenization process for datasets. The workflow includes validating configuration, 
loading the dataset, tokenizing it via a specified tokenizer, and saving the tokenized data 
to a designated output location.
"""

from box import Box
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from src.utils.logging import get_logger, set_logger_level
from src.tasks.tokenization.tokenizer import CausalLMTokenizer
from src.tasks.tokenization.tokenizer.instruction import InstructionTokenizer
from src.tasks.tokenization.tokenizer.mlm import MaskedLMTokenizer
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from src.utils.orchestrator import BaseOrchestrator
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
from utils import inherit_init_params

@inherit_init_params
class TokenizationOrchestrator(BaseOrchestrator):
    """
    Orchestrates the tokenization workflow.

    This class handles the complete process of tokenizing a dataset. It validates the 
    provided configuration, loads the dataset, tokenizes it using a causal language model 
    tokenizer, and saves the tokenized dataset to disk using the specified output path.
    """

    def validate_config(self) -> None:
        """
        Validate the tokenization configuration.

        This method ensures that the essential parts of the configuration are provided:
          - Tokenizer configuration and the tokenizer name.
          - Dataset configuration.
          - Output configuration including a valid path.

        Raises:
            ValueError: If any required configuration element is missing.
        """
        if not self.config.tokenizer:
            raise ValueError("Tokenizer configuration must be provided")
        if not self.config.tokenizer.tokenizer_name:
            raise ValueError("Tokenizer tokenizer_name must be provided")
        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")
        if not self.config.output or not self.config.output.path:
            raise ValueError("Output path must be provided")

    def tokenize_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Tokenize the provided HuggingFace dataset.

        This method creates a tokenizer configuration from self.config and instantiates a 
        CausalLMTokenizer with that configuration. It then applies the tokenization process 
        on the input dataset.

        Args:
            dataset (HFDataset): The dataset to be tokenized.

        Returns:
            HFDataset: The tokenized version of the input dataset.
        """        
        # Derive context length robustly: accept either 'context_length' or legacy 'max_sequence_length'
        context_length = (
            self.config.tokenizer.get("context_length")
            or self.config.tokenizer.get("max_sequence_length")
        )
        if context_length is None:
            raise ValueError("Tokenizer context length not provided. Set tokenizer.context_length or tokenizer.max_sequence_length.")
        overlap = self.config.tokenizer.get("overlap")
        tokenizer_name = self.config.tokenizer.tokenizer_name
        batch_size = self.config.tokenizer.get("batch_size", 2000)  # Default batch size if not specified
        num_proc = self.config.tokenizer.get("num_proc", None)
        show_progress = self.config.tokenizer.get("show_progress", True)
        
        # Extract instruction-specific parameters
        padding_strategy = self.config.tokenizer.get("padding_strategy", "fixed")
        masking_strategy = self.config.tokenizer.get("masking_strategy", "context_aware")
        mask_prompt = self.config.tokenizer.get("mask_prompt", True)
        ignore_index = self.config.tokenizer.get("ignore_index", -100)
        # Support both 'max_seq_length' and 'max_sequence_length' keys
        max_seq_length = self.config.tokenizer.get(
            "max_seq_length",
            self.config.tokenizer.get("max_sequence_length", None),
        )
        test_size = self.config.get("test_size", 0.3)
        seed = self.config.get("seed", 1234)

        
        # Create the tokenizer configuration using parameters from the orchestrator's configuration.
        tokenizer_config = TokenizerConfig(
            context_length=context_length,
            overlap=overlap,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            num_proc=num_proc,
            show_progress=show_progress,
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            ),
            # Instruction-specific parameters
            padding_strategy=padding_strategy,
            masking_strategy=masking_strategy,
            mask_prompt=mask_prompt,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
            test_size=test_size,
            seed=seed
        )

        task = self.config.tokenizer.get("task", "clm_training")
        if task == "clm_training":
            tokenizer = CausalLMTokenizer(tokenizer_config)
        elif task == "instruction":
            tokenizer = InstructionTokenizer(tokenizer_config)
        elif task == "mlm_training":
            # Enrich TokenizerConfig with MLM-specific fields if present in config
            # Using Box-style access (self.config.tokenizer.get) to keep optionality
            tokenizer_config.mlm_probability = self.config.tokenizer.get("mlm_probability", tokenizer_config.mlm_probability)
            tokenizer_config.mask_token = self.config.tokenizer.get("mask_token", tokenizer_config.mask_token)
            tokenizer_config.mask_token_id = self.config.tokenizer.get("mask_token_id", tokenizer_config.mask_token_id)
            tokenizer_config.mask_special_tokens = self.config.tokenizer.get("mask_special_tokens", tokenizer_config.mask_special_tokens)
            tokenizer_config.exclude_token_ids = self.config.tokenizer.get("exclude_token_ids", tokenizer_config.exclude_token_ids)
            tokenizer_config.whole_word_masking = self.config.tokenizer.get("whole_word_masking", tokenizer_config.whole_word_masking)
            # Masking strategy for MLM uses a dict with three probabilities
            mlm_masking_strategy = self.config.tokenizer.get("masking_strategy", None)
            if isinstance(mlm_masking_strategy, dict):
                tokenizer_config.masking_strategy = mlm_masking_strategy
            tokenizer = MaskedLMTokenizer(tokenizer_config)
        # TODO: add more tasks here like the mlm_training...
        else:
            raise ValueError(f"Unsupported tokenization task: {task}")
        
        return tokenizer.tokenize(dataset)

    def execute(self) -> None:
        """
        Execute the complete tokenization workflow.

        The process is carried out in the following steps:
          1. Validate tokenization configuration.
          2. Load the dataset (by invoking self.load_dataset()).
          3. Tokenize the dataset using the tokenize_dataset() method.
          4. Save the tokenized dataset to disk using self.storage.

        Logging is used throughout the process to provide debugging information. In the event 
        of an error, a log message is produced and the exception is re-raised.

        Raises:
            Exception: Propagates any exceptions encountered during the workflow execution.
        """
        
        self.logger.info("Starting tokenization workflow")
        try:
            # 1. Validate configuration
            self.validate_config()

            # 2. Load dataset
            dataset = self.load_dataset()
            
            # 3. Tokenize dataset
            tokenized_dataset = self.tokenize_dataset(dataset)

            # 4. Save results
            self.storage.save_to_disk(tokenized_dataset, self.config.output.path)

            self.logger.info("Tokenization workflow completed successfully")

        except Exception as e:
            self.logger.error(f"Tokenization workflow failed: {str(e)}")
            raise
