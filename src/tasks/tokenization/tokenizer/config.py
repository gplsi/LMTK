"""
Module: config
Path: src/tokenization/config.py

This module defines the configuration for tokenization tasks. It encapsulates 
the parameters required during tokenization in a dataclass, ensuring that the 
configuration is both clear and immutable throughout its usage.

Note:
    This module assumes that a VerboseLevel enum is defined in 'src/utils/logging'
    to control the verbosity levels of logging output.
"""

from dataclasses import dataclass
from typing import Optional
from src.utils.logging import VerboseLevel

@dataclass
class TokenizerConfig:
    """
    Dataclass representing configuration settings for tokenization.

    Attributes:
        context_length (int): 
            The maximum allowed length (in tokens) of the input context. 
            This parameter is essential for defining how many tokens the tokenizer 
            should consider at one time.

        overlap (Optional[int]): 
            The number of tokens that should overlap when processing segments of text.
            This is particularly useful when using sliding window techniques where 
            context continuity is required. Defaults to None if not specified.

        tokenizer_name (Optional[str]): 
            Identifier or name of the specific tokenizer to be used. 
            This allows for flexibility when multiple tokenizer implementations are available.
            Defaults to None if not provided.

        verbose_level (VerboseLevel): 
            Determines the verbosity of logging during the tokenization process.
            Typical levels are defined in the VerboseLevel enum, with a default set to INFO.

    .. attribute:: overlap
       :no-index:
    .. attribute:: tokenizer_name
       :no-index:
    .. attribute:: verbose_level
       :no-index:
    """

    context_length: int
    overlap: Optional[int] = None
    tokenizer_name: Optional[str] = None
    verbose_level: VerboseLevel = VerboseLevel.INFO