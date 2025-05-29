"""
User prompt templates for various language models.

This module contains classes that format user inputs according to different
language model specifications.
"""

from src.tasks.instruction.templates.base import PromptStyle

class DefaultUser(PromptStyle):
    """Default user prompt format (passes through input)."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Return the prompt as-is.
        
        Args:
            prompt: The input text
            **kwargs: Ignored
        """
        self.logger.debug("Applying default user block")
        return prompt

class AlpacaUser(PromptStyle):
    """Alpaca-style instruction format with optional input."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format as Alpaca instruction with optional input.
        
        Args:
            prompt: The instruction text
            **kwargs: May contain 'input' for additional context
        """
        self.logger.debug("Applying Alpaca user block")
        if kwargs.get("input"):
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Input:\n{kwargs['input']}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )

class FlanUser(PromptStyle):
    """Flan-style instruction format with multilingual support."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format as Flan instruction with language support.
        
        Args:
            prompt: The instruction text
            **kwargs: May contain 'lang' (default: 'en') for language selection
        """
        self.logger.debug("Applying Flan user block")
        lang = kwargs.get("lang", "en")
        prompt_dict = {
            'en': "Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:\n{prompt}\n\n### Response:\n",
            'es': "A continuación se muestra una instrucción que describe una tarea. "
                  "Por favor escriba una respuesta que complete adecuadamente la solicitud.\n\n"
                  f"### Instrucción:\n{prompt}\n\n### Respuesta:\n",
            'va': "A continuació es mostra una instrucció que descriu una tasca.  "
                  "Per favor escriga una resposta que complete adequadament la sol·licitud.\n\n"
                  f"### Instrucció:\n{prompt}\n\n### Resposta:\n"
        }
        return prompt_dict[lang]

class StableLMUser(PromptStyle):
    """StableLM user prompt format with role tags."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format with StableLM role tags.
        
        Args:
            prompt: The user's message
            **kwargs: Ignored
        """
        self.logger.debug("Applying StableLM user block")
        return f"<|USER|>{prompt}<|ASSISTANT|>"

class StableLMZephyrUser(PromptStyle):
    """StableLM Zephyr user prompt format with chat roles."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format with Zephyr chat role tags.
        
        Args:
            prompt: The user's message
            **kwargs: Ignored
        """
        self.logger.debug("Applying StableLM Zephyr user block")
        return f"<|user>{prompt}<|assistant>"

class Llama2User(PromptStyle):
    """Llama 2 user prompt format with instruction tags."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format with Llama 2 instruction tags.
        
        Args:
            prompt: The user's instruction
            **kwargs: Ignored
        """
        self.logger.debug("Applying Llama2 user block")
        return f"[INST] {prompt} [/INST] "

class Llama3User(PromptStyle):
    """Llama 3 user prompt format with header tags."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format with Llama 3 header tags.
        
        Args:
            prompt: The user's message
            **kwargs: Ignored
        """
        self.logger.debug("Applying Llama3 user block")
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n"
