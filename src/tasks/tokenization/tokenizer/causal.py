# src/tokenization/causal.py
from typing import Dict, List, Optional, Union
from datasets import Dataset, Features, Sequence, Value
import os

from src.tasks.tokenization.tokenizer.base import BaseTokenizer, TokenizerConfig
from src.tasks.tokenization.tokenizer.utils import build_causal_lm_outputs

class CausalLMTokenizer(BaseTokenizer):
    """Tokenizer for causal language modeling tasks."""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self._features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32"))
        })
    
    def tokenize(self, dataset: Dataset, output_path: Optional[str] = None) -> str:
        """
        Tokenize dataset for causal language modeling.
        
        Args:
            dataset: Input dataset containing text
            output_path: Optional path to save tokenized dataset
            
        Returns:
            Path to tokenized dataset
        """
        self._initialize_tokenizer()
        
        tokenized_datasets = dataset.map(
            self._tokenize_function,
            batched=True,
            features=self._features,
            remove_columns=dataset.column_names
        )
        
        if output_path:
            tokenized_datasets.save_to_disk(output_path)
            return output_path
            
        return tokenized_datasets
    
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
        
        return build_causal_lm_outputs(outputs)
