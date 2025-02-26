# src/tokenization/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
from src.utils.logging import get_logger

class BaseTokenizer(ABC):
    """Base tokenizer class implementing common functionality."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.logger = get_logger(__name__, config.verbose_level)
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        
    def _initialize_tokenizer(self) -> None:
        """Initialize the HuggingFace tokenizer."""
        if not self.config.tokenizer_name:
            raise ValueError("Tokenizer name must be specified")
            
        self.logger.info(f"Initializing tokenizer: {self.config.tokenizer_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, List[int]]:
        """Tokenize input texts."""
        pass
