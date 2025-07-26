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
            
        batch_size (Optional[int]):
            The batch size for processing datasets. If None, uses the dataset's default batch size.
            Larger batch sizes can improve performance but require more memory.
              num_proc (Optional[int]):
            Number of processes to use for parallel processing. If None, uses intelligent defaults:
            - For fast tokenizers: None (relies on internal Rust parallelism)
            - For slow tokenizers: Half of available CPU cores
            Setting to 1 forces single-process mode. Setting to a value > 1 enables Python multiprocessing
            for slow tokenizers (ignored for fast tokenizers).
              show_progress (bool):
            Whether to show progress bars during tokenization. Useful for monitoring progress
            on large datasets. Defaults to True.

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
    batch_size: Optional[int] = None
    num_proc: Optional[int] = None
    show_progress: bool = True
    
    # Instruction-specific parameters
    padding_strategy: str = "fixed"  # "fixed" or "dynamic"
    masking_strategy: str = "context_aware"  # "context_aware" or "response_only"
    mask_prompt: bool = True
    ignore_index: int = -100
    max_seq_length: Optional[int] = None
