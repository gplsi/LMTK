"""
Base classes for instruction handling.

This module provides the base classes for creating and managing instructions
that can be combined with templates to create complete prompts.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

class BaseInstruction(ABC):
    """Base class for all instruction types.
    
    This class defines the interface that all instruction classes must implement.
    """
    
    def __init__(self):
        """Initialize the instruction with a logger."""
        self.logger = get_logger()
    
    @abstractmethod
    def apply(self, **kwargs: Any) -> str:
        """Apply the instruction to the given context.
        
        Args:
            **kwargs: Context variables that can be used in the instruction.
            
        Returns:
            The formatted instruction as a string.
        """
        pass
    
    def __call__(self, **kwargs: Any) -> str:
        """Allow the instruction to be called as a function."""
        return self.apply(**kwargs)


class InstructionSet:
    """A collection of instructions that can be applied together."""
    
    def __init__(self, *instructions: BaseInstruction):
        """Initialize with a sequence of instructions.
        
        Args:
            *instructions: One or more BaseInstruction instances.
        """
        self.instructions = list(instructions)
        self.logger = get_logger()
    
    def add(self, instruction: BaseInstruction) -> None:
        """Add an instruction to the set.
        
        Args:
            instruction: The instruction to add.
        """
        self.instructions.append(instruction)
    
    def apply(self, **kwargs: Any) -> str:
        """Apply all instructions in sequence.
        
        Args:
            **kwargs: Context variables for the instructions.
            
        Returns:
            The combined result of all instructions.
        """
        self.logger.debug("Applying instruction set")
        results = []
        for instruction in self.instructions:
            result = instruction.apply(**kwargs)
            if result:  # Only add non-empty results
                results.append(result)
        return "\n\n".join(results)
    
    def __call__(self, **kwargs: Any) -> str:
        """Allow the instruction set to be called as a function."""
        return self.apply(**kwargs)
