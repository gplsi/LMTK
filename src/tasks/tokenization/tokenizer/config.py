# src/tokenization/config.py
from dataclasses import dataclass
from typing import Optional
from src.utils.logging import VerboseLevel

@dataclass
class TokenizerConfig:
    """Configuration for tokenization tasks."""
    context_length: int
    overlap: Optional[int] = None
    tokenizer_name: Optional[str] = None
    verbose_level: VerboseLevel = VerboseLevel.INFO
