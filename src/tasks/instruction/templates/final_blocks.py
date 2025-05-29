"""
Final prompt templates for various language models.

This module contains classes that define the final part of prompts for different
language models. These templates handle the end of prompt tokens and stopping criteria
for text generation.

Classes:
    DefaultFinal: Default final prompt format
    StableLMFinal: StableLM final format
    StableLMZephyrFinal: StableLM Zephyr final format
    TogetherComputerChatFinal: Together Computer Chat final format
    TogetherComputerInstructFinal: Together Computer Instruct final format
    FalconFinal: Falcon final format
    Llama3Final: Llama 3 final format
    Phi1Final: Phi-1 final format
"""

from typing import List, Tuple
from transformers import PreTrainedTokenizerBase
from src.tasks.instruction.templates.base import PromptStyle

class DefaultFinal(PromptStyle):
    """Default final prompt format with basic EOS token handling."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply default final formatting (returns empty string)."""
        self.logger.debug("Applying default final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for default format (EOS token)."""
        for attr in ('eos_token_id', 'eos_id', 'eos_token'):
            try:
                if hasattr(tokenizer, attr):
                    self.logger.debug(f"Using {attr} as stop token")
                    return ([getattr(tokenizer, attr)],)
            except AttributeError:
                pass
        raise ValueError("Tokenizer does not have eos_token_id, eos_id or eos_token")

class StableLMFinal(PromptStyle):
    """StableLM final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply StableLM final formatting (returns empty string)."""
        self.logger.debug("Applying StableLM final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for StableLM format."""
        self.logger.debug("Getting StableLM final stop tokens")
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )

class StableLMZephyrFinal(PromptStyle):
    """StableLM Zephyr final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply StableLM Zephyr final formatting (returns empty string)."""
        self.logger.debug("Applying StableLM Zephyr final block")
        return ""

class TogetherComputerChatFinal(PromptStyle):
    """Together Computer Chat final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Together Computer Chat final formatting (returns empty string)."""
        self.logger.debug("Applying TogetherComputerChat final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for Together Computer Chat format."""
        lt, gt = tokenizer.token_to_id("<"), tokenizer.token_to_id(">:")
        self.logger.debug(f"Using {lt} and {gt} as stop tokens")
        return (
            [tokenizer.eos_id],
            [lt, tokenizer.token_to_id("human"), gt],
            [lt, tokenizer.token_to_id("bot"), gt],
        )

class TogetherComputerInstructFinal(PromptStyle):
    """Together Computer Instruct final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Together Computer Instruct final formatting (returns empty string)."""
        self.logger.debug("Applying TogetherComputerInstruct final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for Together Computer Instruct format."""
        colon = tokenizer.token_to_id(":")
        self.logger.debug(f"Using {colon} as stop token")
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Q"), colon],
            [tokenizer.token_to_id("Question")],
            [tokenizer.token_to_id("A"), colon],
            [tokenizer.token_to_id("Label"), colon],
            [187, 187],  # '\n', '\n'
            [535],  # '\n\n'
            [2756],  # '\n\n\n'
        )

class FalconFinal(PromptStyle):
    """Falcon final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Falcon final formatting (returns empty string)."""
        self.logger.debug("Applying Falcon final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for Falcon format."""
        self.logger.debug("Getting Falcon final stop tokens")
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )

class Llama3Final(PromptStyle):
    """Llama 3 final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Llama 3 final formatting (returns empty string)."""
        self.logger.debug("Applying Llama3 final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for Llama 3 format."""
        self.logger.debug("Getting Llama3 final stop tokens")
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|eot_id|>")],
        )

class Phi1Final(PromptStyle):
    """Phi-1 final prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Phi-1 final formatting (returns empty string)."""
        self.logger.debug("Applying Phi1 final block")
        return ""
    
    def stop_tokens(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], ...]:
        """Get stop tokens for Phi-1 format."""
        self.logger.debug("Getting Phi1 final stop tokens")
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
        )