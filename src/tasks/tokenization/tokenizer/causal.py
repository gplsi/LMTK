# src/tokenization/causal.py
from typing import Dict, List, Optional, Union
from datasets import Features, Sequence, Value
from datasets import Dataset as HFDataset
import os
from datasets import DatasetDict, Dataset
from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.utils import build_causal_lm_outputs
from src.tasks.tokenization.tokenizer.config import TokenizerConfig

class CausalLMTokenizer(BaseTokenizer):
    """
    Tokenizer implementation for casual language modeling tasks.

    This class extends BaseTokenizer to provide tokenization methods specifically designed
    for causal language modeling. It supports both single HFDataset instances and multiple splits
    encapsulated in a DatasetDict. The tokenization process generates sequences for input IDs,
    attention masks, and labels, with the labels mirroring the input IDs for training purposes.
    """
    
    def __init__(self, config: TokenizerConfig) -> None:
        """
        Initialize the CausalLMTokenizer instance with the given configuration.

        This constructor sets up the tokenizer by invoking the base constructor and defines
        the expected output structure (features) of tokenization using Hugging Face Datasets.

        Args:
            config (TokenizerConfig): Configuration parameters for the tokenizer, including
                                      context length and overlap settings.
        """
        super().__init__(config)
        # Define the expected structure of tokenized outputs:
        # - input_ids: sequence of token identifiers (int32)
        # - attention_mask: sequence indicating which tokens should be attended to (int32)
        # - labels: sequence of token identifiers (int32) used as training labels
        self._features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32"))
        })
    
    def tokenize(self, dataset: Union[HFDataset, DatasetDict]) -> HFDataset:
        """
        Tokenize the input dataset for causal language modeling.

        This method applies tokenization to either a single dataset (HFDataset) or a dictionary 
        of datasets (DatasetDict) containing multiple splits. It utilizes the Hugging Face's map 
        function to efficiently process data in batches. Post tokenization, it removes unnecessary 
        columns from the original dataset.

        Args:
            dataset (Union[HFDataset, DatasetDict]): The dataset or dataset dictionary to be tokenized.

        Returns:
            Union[HFDataset, DatasetDict]: A tokenized dataset if a single HFDataset is provided, or
                                           a DatasetDict if multiple splits were tokenized. In the case
                                           of a DatasetDict with only one split, the single tokenized dataset
                                           is returned directly.
        """
        self.logger.info("Initializing tokenizer")
        # Initialize the underlying tokenizer (typically set up in BaseTokenizer)
        self._initialize_tokenizer()
        
        # Handle the case where dataset is a dictionary of splits (e.g., train/validation/test)
        if isinstance(dataset, DatasetDict):
            self.logger.info("Detected 'DatasetDict' instance")
            tokenized_datasets = DatasetDict()
            for split, split_dataset in dataset.items():
                # Map the _tokenize_function over the current split dataset
                tokenized_datasets[split] = split_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    features=self._features,
                    remove_columns=split_dataset.column_names  # Remove columns not needed post tokenization
                )
            
            # If there is only one split, return the single tokenized dataset directly
            if len(tokenized_datasets) == 1:
                self.logger.debug("Only one dataset split found, returning single tokenized dataset")
                return tokenized_datasets[list(tokenized_datasets.keys())[0]]
            
            self.logger.debug("Multiple dataset splits found, returning tokenized 'DatasetDict'")
            return tokenized_datasets
        else:
            self.logger.info("Detected regular 'Dataset' instance")
            # Handle a single HFDataset instance by tokenizing it directly
            tokenized_dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                features=self._features,
                remove_columns=dataset.column_names  # Clean up raw data columns after tokenization
            )
            self.logger.debug("Returning tokenized dataset")
            return tokenized_dataset
    
    def _tokenize_function(self, batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """
        Internal helper function to tokenize a batch of text data.

        This function processes the 'text' field of a batch using the underlying tokenizer.
        It applies operations such as truncation, padding, and handling of overlapping tokens.
        The generated outputs include token IDs, attention masks, and labels, where the labels
        are a copy of the input_ids to support causal language modeling training configurations.

        Args:
            batch (Dict[str, List[str]]): A dictionary with a key "text" containing a list of text strings.

        Returns:
            Dict[str, List[int]]: A dictionary with keys:
                - "input_ids": The tokenized representation of the input text.
                - "attention_mask": Binary mask indicating real tokens versus padding.
                - "labels": Copy of input_ids, used as training labels for language modeling.
        """
        outputs = self._tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.context_length,
            return_overflowing_tokens=True,
            return_length=True,
            stride=self.config.overlap,
            padding=True
        )
        
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": outputs["input_ids"].copy()  # Duplicate input_ids for use as labels in LM training
        }
