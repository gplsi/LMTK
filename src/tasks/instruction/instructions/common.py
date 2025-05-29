"""
Common instruction types for language model prompting.

This module provides implementations of common instruction patterns
that can be used with different language models.
"""
from typing import Dict, Any, Optional, List
from .base import BaseInstruction

class SimpleInstruction(BaseInstruction):
    """A simple text-based instruction."""
    
    def __init__(self, text: str):
        """Initialize with fixed instruction text.
        
        Args:
            text: The instruction text to use.
        """
        super().__init__()
        self.text = text
    
    def apply(self, **kwargs: Any) -> str:
        """Return the instruction text.
        
        Args:
            **kwargs: Ignored
            
        Returns:
            The instruction text.
        """
        self.logger.debug(f"Applying simple instruction: {self.text}")
        return self.text


class TemplatedInstruction(BaseInstruction):
    """An instruction that uses string formatting with variables."""
    
    def __init__(self, template: str, required_vars: Optional[List[str]] = None):
        """Initialize with a template string.
        
        Args:
            template: String template with {variable} placeholders
            required_vars: List of variable names that must be provided
        """
        super().__init__()
        self.template = template
        self.required_vars = required_vars or []
    
    def apply(self, **kwargs: Any) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            The formatted instruction.
            
        Raises:
            ValueError: If a required variable is missing.
        """
        # Check for required variables
        for var in self.required_vars:
            if var not in kwargs:
                raise ValueError(f"Required variable '{var}' not provided for instruction")
        
        self.logger.debug(f"Applying templated instruction with vars: {kwargs.keys()}")
        return self.template.format(**kwargs)


class ConditionalInstruction(BaseInstruction):
    """An instruction that changes based on conditions."""
    
    def __init__(self, conditions: Dict[str, str], default: str = ""):
        """Initialize with condition-to-instruction mapping.
        
        Args:
            conditions: Dictionary mapping condition names to instruction texts
            default: Default instruction if no condition matches
        """
        super().__init__()
        self.conditions = conditions
        self.default = default
    
    def apply(self, **kwargs: Any) -> str:
        """Apply the instruction based on condition.
        
        Args:
            **kwargs: Must contain 'condition' key with value matching
                     one of the condition names
            
        Returns:
            The instruction for the matching condition or default.
        """
        condition = kwargs.get('condition', '')
        instruction = self.conditions.get(condition, self.default)
        self.logger.debug(f"Applying conditional instruction for '{condition}'")
        return instruction


class TaskSpecificInstruction(BaseInstruction):
    """An instruction tailored to a specific task type."""
    
    def __init__(self, task_instructions: Dict[str, str], default: str = ""):
        """Initialize with task-to-instruction mapping.
        
        Args:
            task_instructions: Dictionary mapping task types to instruction texts
            default: Default instruction if task type not found
        """
        super().__init__()
        self.task_instructions = task_instructions
        self.default = default
    
    def apply(self, **kwargs: Any) -> str:
        """Apply the instruction based on task type.
        
        Args:
            **kwargs: Must contain 'task_type' key
            
        Returns:
            The instruction for the specified task or default.
        """
        task_type = kwargs.get('task_type', '')
        instruction = self.task_instructions.get(task_type, self.default)
        self.logger.debug(f"Applying task-specific instruction for '{task_type}'")
        return instruction


class MultiPartInstruction(BaseInstruction):
    """An instruction composed of multiple parts."""
    
    def __init__(self, parts: List[BaseInstruction], separator: str = "\n\n"):
        """Initialize with multiple instruction parts.
        
        Args:
            parts: List of instruction components
            separator: Text to use between parts
        """
        super().__init__()
        self.parts = parts
        self.separator = separator
    
    def apply(self, **kwargs: Any) -> str:
        """Apply all parts and join with separator.
        
        Args:
            **kwargs: Passed to each instruction part
            
        Returns:
            The combined instruction.
        """
        self.logger.debug(f"Applying multi-part instruction with {len(self.parts)} parts")
        results = []
        for part in self.parts:
            result = part.apply(**kwargs)
            if result:  # Only add non-empty results
                results.append(result)
        return self.separator.join(results)
