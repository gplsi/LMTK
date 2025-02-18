# src/tokenization/causal.py
from typing import Dict, List, Optional, Union
from datasets import Dataset, Features, Sequence, Value
import os
from datasets import DatasetDict, Dataset
from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.utils import build_causal_lm_outputs
from src.tasks.tokenization.tokenizer.config import TokenizerConfig

class CausalLMTokenizer(BaseTokenizer):
    """Tokenizer for causal language modeling tasks."""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self._features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32"))
        })
    
    def tokenize(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for causal language modeling."""
        self.logger.info("Initializing tokenizer")
        self._initialize_tokenizer()
        
        if isinstance(dataset, DatasetDict):
            self.logger.info("Detected 'DatasetDict' instance")
            # Handle DatasetDict case
            tokenized_datasets = DatasetDict()
            for split, split_dataset in dataset.items():
                tokenized_datasets[split] = split_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    features=self._features,
                    remove_columns=split_dataset.column_names
                )
            
            if len(tokenized_datasets) == 1:
                self.logger.debug("Only one dataset split found, returning single tokenized dataset")
                return tokenized_datasets[list(tokenized_datasets.keys())[0]]
            
            self.logger.debug("Multiple dataset splits found, returning tokenized 'DatasetDict'")
            return tokenized_datasets
        else:
            self.logger.info("Detected regular 'Dataset' instance")
            # Handle single Dataset case
            tokenized_dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                features=self._features,
                remove_columns=dataset.column_names
            )
            self.logger.debug("Returning tokenized dataset")
            return tokenized_dataset
    
    def _tokenize_function(self, batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Internal tokenization function."""
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
            "labels": outputs["input_ids"].copy()
        }
