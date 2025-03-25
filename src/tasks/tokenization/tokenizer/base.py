"""
Module: base.py
This module defines the BaseTokenizer abstract class, which provides a framework
to initialize and utilize a HuggingFace tokenizer. It ensures that any tokenizer
implementation adheres to a common interface, aiding in scalability and maintainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer, AutoTokenizer

from src.tasks.tokenization.tokenizer.config import TokenizerConfig
from src.utils.logging import get_logger

class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    This class implements the basic functionality required to initialize a HuggingFace tokenizer 
    using a configuration object. Subclasses are expected to provide a concrete implementation
    for the `tokenize` method.
    
    Attributes:
        config (TokenizerConfig): Configuration containing parameters for tokenizer initialization.
        logger: Logger instance to log informational or debugging messages.
        _tokenizer (Optional[PreTrainedTokenizer]): The actual HuggingFace tokenizer instance, 
            initialized on demand.
    """
    def __init__(self, config: TokenizerConfig) -> None:
        """
        Initialize the BaseTokenizer with the provided configuration.

        This constructor sets the configuration, initializes the logger based on the 
        verbosity level specified in the configuration, and prepares a placeholder for 
        the actual tokenizer instance.

        Args:
            config (TokenizerConfig): Configuration object with tokenizer parameters.
        """
        self.config = config
        self.logger = get_logger(__name__, config.verbose_level)
        # Initialize the tokenizer instance as None. It will be set during explicit initialization.
        self._tokenizer: Optional[PreTrainedTokenizer] = None

    def _initialize_tokenizer(self) -> None:
        """
        Initialize the HuggingFace tokenizer based on the tokenizer name in the configuration.

        This method uses the AutoTokenizer to load a pre-trained tokenizer model by name.
        It also sets the padding token to the end-of-sequence token to ensure consistency 
        in tokenization.

        Raises:
            ValueError: If the tokenizer name is not specified in the configuration.
        """
        if not self.config.tokenizer_name:
            raise ValueError("Tokenizer name must be specified")

        self.logger.info(f"Initializing tokenizer: {self.config.tokenizer_name}")
        # Load the tokenizer from HuggingFace's pre-trained models.
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        # Configure the padding token to be the same as the end-of-sequence token.
        self._tokenizer.pad_token = self._tokenizer.eos_token

    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, List[int]]:
        """
        Abstract method to tokenize input texts into token IDs.

        This method must be implemented by subclasses to convert a given text or list of texts 
        into a dictionary where keys represent a tokenizer output type (e.g., 'input_ids') and 
        values are the corresponding token IDs.

        Args:
            texts (Union[str, List[str]]): A single text string or a list of text strings to be tokenized.

        Returns:
            Dict[str, List[int]]: A dictionary mapping tokenization features to lists of token IDs.
        """
        pass