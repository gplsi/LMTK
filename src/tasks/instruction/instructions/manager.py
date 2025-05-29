"""
Instruction manager for integrating instructions with template composer.

This module provides the InstructionManager class that coordinates between
instruction sets and the template composer to create complete prompts.
"""
from typing import Dict, Any, Optional, Union, List
from transformers import PreTrainedTokenizerBase

from .base import BaseInstruction, InstructionSet
from .model_instructions import get_model_instruction
from ..templates.composer import PromptComposer

class InstructionManager:
    """Manages instructions and integrates them with the template composer.
    
    This class serves as the bridge between the instruction module and the
    template composer, allowing for flexible composition of prompts with
    appropriate instructions for different models and tasks.
    """
    
    def __init__(
        self,
        composer: PromptComposer,
        default_instruction: Optional[Union[BaseInstruction, InstructionSet]] = None,
        model_name: Optional[str] = None
    ):
        """Initialize the instruction manager.
        
        Args:
            composer: The PromptComposer instance to use
            default_instruction: Default instruction to use if none specified
            model_name: Name of the model to get optimized instructions for
        """
        self.composer = composer
        self.model_name = model_name
        
        # Set up default instruction based on model if provided
        if default_instruction is None and model_name is not None:
            self.default_instruction = get_model_instruction(model_name)
        else:
            self.default_instruction = default_instruction or get_model_instruction("default")
        
        # Dictionary to store task-specific instruction sets
        self.task_instructions: Dict[str, Union[BaseInstruction, InstructionSet]] = {}
        
        # Get logger from composer
        self.logger = composer.logger
    
    def register_task_instruction(
        self, 
        task_name: str, 
        instruction: Union[BaseInstruction, InstructionSet]
    ) -> None:
        """Register an instruction set for a specific task.
        
        Args:
            task_name: Name of the task
            instruction: Instruction or instruction set for the task
        """
        self.task_instructions[task_name] = instruction
        self.logger.debug(f"Registered instruction for task '{task_name}'")
    
    def get_instruction_for_task(
        self, 
        task_name: Optional[str] = None
    ) -> Union[BaseInstruction, InstructionSet]:
        """Get the instruction for a specific task.
        
        Args:
            task_name: Name of the task to get instruction for
            
        Returns:
            The instruction for the task or the default instruction
        """
        if task_name is not None and task_name in self.task_instructions:
            return self.task_instructions[task_name]
        return self.default_instruction
    
    def apply_instruction(
        self,
        user_input: str,
        task_name: Optional[str] = None,
        instruction_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Apply the appropriate instruction to the user input.
        
        Args:
            user_input: The raw user input
            task_name: Optional task name to select instruction
            instruction_kwargs: Additional arguments for the instruction
            **kwargs: Additional arguments for both instruction and composer
            
        Returns:
            The input with instruction applied
        """
        instruction_kwargs = instruction_kwargs or {}
        
        # Get the appropriate instruction
        instruction = self.get_instruction_for_task(task_name)
        
        # Apply the instruction
        instruction_text = instruction.apply(**instruction_kwargs, **kwargs)
        self.logger.debug(f"Applied instruction for task '{task_name}': {instruction_text}")
        
        # Combine instruction with user input
        if instruction_text:
            # The instruction is prepended to the user input
            combined_input = f"{instruction_text}\n\n{user_input}"
            self.logger.debug(f"Combined input: {combined_input}")
            return combined_input
        
        # If no instruction text, return the original input
        return user_input
    
    def compose_with_instruction(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        task_name: Optional[str] = None,
        instruction_kwargs: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply instruction and compose the complete prompt.
        
        This is the main method that combines instruction application
        with template composition.
        
        Args:
            user_input: The raw user input
            system_prompt: Optional system prompt
            task_name: Optional task name to select instruction
            instruction_kwargs: Additional arguments for the instruction
            label: Optional label/response for training
            **kwargs: Additional arguments for both instruction and composer
            
        Returns:
            The composed prompt dictionary from the composer
        """
        # Apply the instruction to the user input
        instructed_input = self.apply_instruction(
            user_input=user_input,
            task_name=task_name,
            instruction_kwargs=instruction_kwargs,
            **kwargs
        )
        
        # Use the composer to create the final prompt
        return self.composer.compose(
            user_input=instructed_input,
            system_prompt=system_prompt,
            label=label,
            **kwargs
        )
    
    def __call__(self, *args, **kwargs):
        """Alias for compose_with_instruction method."""
        return self.compose_with_instruction(*args, **kwargs)
