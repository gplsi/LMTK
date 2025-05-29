"""
Prompt template composer for instruction-following language models.

This module provides functionality to compose prompts from different template components
and handle tokenization with Hugging Face tokenizers.
"""

from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from .base import PromptStyle
from .system_blocks import DefaultSystem
from .user_blocks import DefaultUser
from .assistant_blocks import DefaultAssistant
from .final_blocks import DefaultFinal
from utils.logger import get_logger

class PromptComposer:
    """Composes prompts from template components and handles tokenization.
    
    This class provides methods to create properly formatted prompts for both
    training and inference, with support for different template styles and
    proper handling of special tokens.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_style: Optional[PromptStyle] = None,
        user_style: Optional[PromptStyle] = None,
        assistant_style: Optional[PromptStyle] = None,
        final_style: Optional[PromptStyle] = None,
        max_length: int = 2048,
        truncation_side: str = "right"
    ):
        """Initialize the prompt composer.
        
        Args:
            tokenizer: Hugging Face tokenizer instance
            system_style: Template for system prompt (default: DefaultSystem)
            user_style: Template for user input (default: DefaultUser)
            assistant_style: Template for assistant response (default: DefaultAssistant)
            final_style: Template for final block (default: DefaultFinal)
            max_length: Maximum sequence length
            truncation_side: Which side to truncate from ("left" or "right")
        """
        self.tokenizer = tokenizer
        self.system_style = system_style or DefaultSystem()
        self.user_style = user_style or DefaultUser()
        self.assistant_style = assistant_style or DefaultAssistant()
        self.final_style = final_style or DefaultFinal()
        self.max_length = max_length
        self.truncation_side = truncation_side
        
        # Set up logging from any of the styles
        self.logger = get_logger()
    
    def compose(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        label: Optional[str] = None,
        **template_kwargs
    ) -> Dict[str, Union[str, List[int]]]:
        """Compose a complete prompt with optional label for training.
        
        Args:
            user_input: The user's input/instruction
            system_prompt: Optional system prompt (uses default if None)
            label: Optional label/response for training
            **template_kwargs: Additional arguments to pass to template styles
            
        Returns:
            Dictionary containing:
            - 'text': The complete prompt string
            - 'input_ids': Tokenized input IDs
            - 'attention_mask': Attention mask
            - 'labels': Tokenized labels (if label provided)
        """
        # Apply templates
        system = self.system_style.apply(system=system_prompt or "", **template_kwargs)
        self.logger.debug(f"System prompt: {system}")
        user = self.user_style.apply(prompt=user_input, **template_kwargs)
        self.logger.debug(f"User prompt: {user}")
        
        # For training, include the label with assistant formatting
        self.logger.debug(f"Generating the full text for train-test mode")
        if label is not None:
            assistant = self.assistant_style.apply(response=label, **template_kwargs)
            self.logger.debug(f"Assistant response: {assistant}")
            full_text = f"{system}{user}{assistant}"
            self.logger.debug(f"Full text: {full_text}")
        else:
            self.logger.debug(f"Generating the full text for inference mode")
            # For inference, just include the assistant start token
            assistant_start = getattr(self.assistant_style, 'start_token', '')
            full_text = f"{system}{user}{assistant_start}"
            self.logger.debug(f"Full text: {full_text}")
        
        # Apply final formatting (e.g., EOS token)
        final_text = f"{full_text}{self.final_style.apply(**template_kwargs)}"
        self.logger.debug(f"Final text: {final_text}")
        

        # Tokenize
        tokenized = self.tokenizer(
            final_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length' if label is not None else False,
            return_tensors='pt' if label is not None else None,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        self.logger.debug(f"Tokenized input_ids: {tokenized['input_ids']}")
        self.logger.debug(f"Tokenized attention_mask: {tokenized['attention_mask']}")
        
        # Prepare output
        result = {
            'text': final_text,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        
        # For training, prepare labels (shifted right)
        if label is not None:
            self.logger.debug(f"Generating the labels for training mode")
            # Tokenize the full prompt + response
            prompt_without_response = f"{system}{user}"
            response_text = f"{self.assistant_style.apply(response=label, **template_kwargs)}"
            self.logger.debug(f"Prompt without response: {prompt_without_response}")
            self.logger.debug(f"Response text: {response_text}")


            # Tokenize prompt and response separately to create proper labels
            prompt_tokens = self.tokenizer(
                prompt_without_response,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True
            )
            self.logger.debug(f"Prompt tokens: {prompt_tokens}")
            
            response_tokens = self.tokenizer(
                response_text,
                truncation=True,
                max_length=self.max_length - len(prompt_tokens['input_ids']),
                add_special_tokens=False
            )
            self.logger.debug(f"Response tokens: {response_tokens}")
            
            # Create labels with -100 for prompt tokens
            labels = [-100] * len(prompt_tokens['input_ids']) + response_tokens['input_ids']
            
            
            # Pad to max length
            padding_length = self.max_length - len(labels)
            if padding_length > 0:
                labels = labels + [-100] * padding_length
            else:
                labels = labels[:self.max_length]
            
            result['labels'] = labels
            self.logger.debug(f"Labels: {labels}")
        return result
    
    def get_stop_tokens(self) -> Tuple[List[int], ...]:
        """Get stop tokens from the final style."""
        if hasattr(self.final_style, 'stop_tokens'):
            self.logger.debug(f"Stop tokens: {self.final_style.stop_tokens(self.tokenizer)}")
            return self.final_style.stop_tokens(self.tokenizer)
        self.logger.debug(f"Stop tokens: {[self.tokenizer.eos_token_id]}")
        return ([self.tokenizer.eos_token_id],)
    
    def __call__(self, *args, **kwargs):
        """Alias for compose method."""
        return self.compose(*args, **kwargs)
