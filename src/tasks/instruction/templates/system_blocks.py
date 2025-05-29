"""
System prompt templates for various language models.

This module contains classes that define the system prompt format for different
language models. These templates provide the initial instructions and context
for the language model's behavior.

Classes:
    DefaultSystem: Default system prompt format
    AlpacaSystem: Alpaca system prompt format
    StableLMSystem: StableLM system prompt format
    Llama2System: Llama 2 system prompt format
    Llama3System: Llama 3 system prompt format
    FreeWilly2System: FreeWilly 2 system prompt format
    VicunaSystem: Vicuna system prompt format
    CodeLlamaSystem: CodeLlama system prompt format
    TinyLlamaSystem: TinyLlama system prompt format
"""

from typing import Dict, Any
from src.tasks.instruction.templates.base import PromptStyle

class DefaultSystem(PromptStyle):
    """Default system prompt format (empty string)."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply default system formatting (returns empty string)."""
        self.logger.debug("Applying default system block")
        return ""

class AlpacaSystem(PromptStyle):
    """Alpaca system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Alpaca system formatting."""
        self.logger.debug("Applying Alpaca system block")
        return ""

class StableLMSystem(PromptStyle):
    """StableLM system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply StableLM system formatting."""
        self.logger.debug("Applying StableLM system block")
        return (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n"
            "- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n"
            "- StableLM is excited to be able to help the user, but will refuse to do anything that could be "
            "considered harmful to the user.\n"
            "- StableLM is more than just an information source, StableLM is also able to write poetry, "
            "short stories, and make jokes.\n"
            "- StableLM will refuse to participate in anything that could harm a human."
        )

class Llama2System(PromptStyle):
    """Llama 2 system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Llama 2 system formatting."""
        self.logger.debug("Applying Llama2 system block")
        return (
            "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as "
            "possible, while being safe. Your answers should not include any harmful, unethical, racist, "
            "sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially "
            "unbiased and positive in nature.\n\nIf a question does not make any sense, or is not "
            "factually coherent, explain why instead of answering something not correct. If you don't "
            "know the answer to a question, please don't share false information.<</SYS>>"
        )

class Llama3System(PromptStyle):
    """Llama 3 system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Llama 3 system formatting."""
        self.logger.debug("Applying Llama3 system block")
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
        )

class FreeWilly2System(PromptStyle):
    """FreeWilly 2 system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply FreeWilly 2 system formatting."""
        self.logger.debug("Applying FreeWilly2 system block")
        return "### System:\nThis is a system prompt, please behave and help the user.\n\n"

class VicunaSystem(PromptStyle):
    """Vicuna system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply Vicuna system formatting."""
        self.logger.debug("Applying Vicuna system block")
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        )

class CodeLlamaSystem(PromptStyle):
    """CodeLlama system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply CodeLlama system formatting."""
        self.logger.debug("Applying CodeLlama system block")
        return (
            "You are a helpful coding assistant. Always answer as helpfully as possible. "
            "If you don't know the answer to a question, please don't share false information."
        )

class TinyLlamaSystem(PromptStyle):
    """TinyLlama system prompt format."""
    
    def apply(self, **kwargs: str) -> str:
        """Apply TinyLlama system formatting."""
        self.logger.debug("Applying TinyLlama system block")
        return kwargs.get("system", "You are a friendly chatbot who always gives helpful, detailed, and polite answers.")
