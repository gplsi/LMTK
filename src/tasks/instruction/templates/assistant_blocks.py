"""
Assistant prompt templates for various language models.

This module contains classes that define the assistant's response format for different
language models. Each class implements the `apply` method which formats the assistant's
response according to the specific model's requirements.

Classes:
    DefaultAssistant: Default assistant format (empty string)
    StableLMAssistant: StableLM assistant format
    StableLMZephyrAssistant: StableLM Zephyr assistant format
    TogetherComputerChatAssistant: Together Computer Chat assistant format
    TogetherComputerInstructAssistant: Together Computer Instruct assistant format
    FalconAssistant: Falcon assistant format
    VicunaAssistant: Vicuna assistant format
    Llama2Assistant: Llama 2 assistant format
    Llama3Assistant: Llama 3 assistant format
    FreeWilly2Assistant: FreeWilly 2 assistant format
    PlatypusAssistant: Platypus assistant format
    CodeLlamaAssistant: CodeLlama assistant format
    Phi1Assistant: Phi-1 assistant format
    Phi2Assistant: Phi-2 assistant format
    TinyLlamaAssistant: TinyLlama assistant format
    GemmaAssistant: Gemma assistant format
"""

from typing import Any, Dict, Optional
from src.tasks.instruction.templates.base import PromptStyle

class DefaultAssistant(PromptStyle):
    """Default assistant prompt format (empty string)."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply default assistant formatting (returns empty string)."""
        self.logger.debug("Applying default assistant block")
        return ""

class StableLMAssistant(PromptStyle):
    """StableLM assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply StableLM assistant formatting (returns empty string)."""
        self.logger.debug("Applying StableLM assistant block")
        return ""

class StableLMZephyrAssistant(PromptStyle):
    """StableLM Zephyr assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply StableLM Zephyr assistant formatting."""
        self.logger.debug("Applying StableLM Zephyr assistant block")
        return "<|assistant|>"

class TogetherComputerChatAssistant(PromptStyle):
    """Together Computer Chat assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Together Computer Chat assistant formatting."""
        self.logger.debug("Applying TogetherComputerChat assistant block")
        return "\n<bot>:"

class TogetherComputerInstructAssistant(PromptStyle):
    """Together Computer Instruct assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Together Computer Instruct assistant formatting."""
        self.logger.debug("Applying TogetherComputerInstruct assistant block")
        return "\nA:"

class FalconAssistant(PromptStyle):
    """Falcon assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Falcon assistant formatting (returns empty string)."""
        self.logger.debug("Applying Falcon assistant block")
        return ""

class VicunaAssistant(PromptStyle):
    """Vicuna assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Vicuna assistant formatting."""
        self.logger.debug("Applying Vicuna assistant block")
        return " ASSISTANT:"

class Llama2Assistant(PromptStyle):
    """Llama 2 assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Llama 2 assistant formatting (returns empty string)."""
        self.logger.debug("Applying Llama2 assistant block")
        return ""

class Llama3Assistant(PromptStyle):
    """Llama 3 assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Llama 3 assistant formatting."""
        self.logger.debug("Applying Llama3 assistant block")
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

class FreeWilly2Assistant(PromptStyle):
    """FreeWilly 2 assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply FreeWilly 2 assistant formatting."""
        self.logger.debug("Applying FreeWilly2 assistant block")
        return "### Assistant:\n"

class PlatypusAssistant(PromptStyle):
    """Platypus assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Platypus assistant formatting."""
        self.logger.debug("Applying Platypus assistant block")
        return "### Response:\n"

class CodeLlamaAssistant(PromptStyle):
    """CodeLlama assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply CodeLlama assistant formatting (returns empty string)."""
        self.logger.debug("Applying CodeLlama assistant block")
        return ""

class Phi1Assistant(PromptStyle):
    """Phi-1 assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Phi-1 assistant formatting (returns empty string)."""
        self.logger.debug("Applying Phi1 assistant block")
        return ""

class Phi2Assistant(PromptStyle):
    """Phi-2 assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Phi-2 assistant formatting (returns empty string)."""
        self.logger.debug("Applying Phi2 assistant block")
        return ""

class TinyLlamaAssistant(PromptStyle):
    """TinyLlama assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply TinyLlama assistant formatting."""
        self.logger.debug("Applying TinyLlama assistant block")
        return "<|assistant|>\n"

class GemmaAssistant(PromptStyle):
    """Gemma assistant prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Gemma assistant formatting."""
        self.logger.debug("Applying Gemma assistant block")
        return "<start_of_turn>model\n"
