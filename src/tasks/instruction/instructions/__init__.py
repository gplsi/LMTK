"""
Instructions module for language model prompting.

This module provides a framework for creating, managing, and applying
instructions to language model inputs, working alongside the template system.
"""

from .base import BaseInstruction, InstructionSet
from .common import (
    SimpleInstruction,
    TemplatedInstruction,
    ConditionalInstruction,
    TaskSpecificInstruction,
    MultiPartInstruction
)
from .model_instructions import (
    TASK_TYPES,
    DefaultInstruction,
    AlpacaInstruction,
    LlamaInstruction,
    MistralInstruction,
    get_model_instruction,
    get_summarization_instructions,
    get_qa_instructions,
    get_translation_instructions
)
from .manager import InstructionManager
from .dataset_handler import DatasetHandler

__all__ = [
    # Base classes
    'BaseInstruction',
    'InstructionSet',
    
    # Common instruction types
    'SimpleInstruction',
    'TemplatedInstruction',
    'ConditionalInstruction',
    'TaskSpecificInstruction',
    'MultiPartInstruction',
    
    # Model-specific instructions
    'TASK_TYPES',
    'DefaultInstruction',
    'AlpacaInstruction',
    'LlamaInstruction',
    'MistralInstruction',
    'get_model_instruction',
    
    # Task-specific instruction sets
    'get_summarization_instructions',
    'get_qa_instructions',
    'get_translation_instructions',
    
    # Manager
    'InstructionManager',
    
    # Dataset handling
    'DatasetHandler',
]
