"""
Model-specific instructions for different language models.

This module provides predefined instructions optimized for different
language model architectures and families.
"""
from typing import Dict, Any, Optional
from .base import BaseInstruction, InstructionSet
from .common import SimpleInstruction, TemplatedInstruction, TaskSpecificInstruction

# Standardized task types that can be referenced by all model instructions
TASK_TYPES = {
    'SUMMARIZE': 'summarize',
    'TRANSLATE': 'translate',
    'ANSWER': 'answer',
    'GENERATE': 'generate',
    'CLASSIFY': 'classify',
    'EXTRACT': 'extract',
    'CODE': 'code',
    'CHAT': 'chat',
    'REWRITE': 'rewrite',
    'ANALYZE': 'analyze',
    'EXPLAIN': 'explain',
    'COMPARE': 'compare',
    'CRITIQUE': 'critique',
    'BRAINSTORM': 'brainstorm'
}

class DefaultInstruction(BaseInstruction):
    """Default generic instruction that works with most models."""
    
    def apply(self, **kwargs: Any) -> str:
        """Apply a generic instruction.
        
        Args:
            **kwargs: May contain 'task' for task-specific instructions
            
        Returns:
            A generic instruction.
        """
        self.logger.debug("Applying default instruction")
        task = kwargs.get('task', '')
        if task:
            return f"Please perform the following task: {task}"
        return "Please respond to the following:"


class AlpacaInstruction(BaseInstruction):
    """Instructions optimized for Alpaca-style models."""
    
    def apply(self, **kwargs: Any) -> str:
        """Apply Alpaca-optimized instruction.
        
        Args:
            **kwargs: May contain 'task_type' for specific formatting
            
        Returns:
            Instruction formatted for Alpaca models.
        """
        self.logger.debug("Applying Alpaca instruction")
        task_type = kwargs.get('task_type', '')
        
        task_instructions = {
            TASK_TYPES['SUMMARIZE']: "Summarize the following text concisely while preserving the key information:",
            TASK_TYPES['TRANSLATE']: "Translate the following text to {target_language}:",
            TASK_TYPES['ANSWER']: "Answer the following question accurately and concisely:",
            TASK_TYPES['GENERATE']: "Generate {content_type} based on the following description:",
            TASK_TYPES['CLASSIFY']: "Classify the following text into one of these categories: {categories}:",
            TASK_TYPES['EXTRACT']: "Extract the {entity_type} from the following text:",
            TASK_TYPES['CODE']: "Write code in {programming_language} that does the following:",
            TASK_TYPES['EXPLAIN']: "Explain the following concept in simple terms:",
            TASK_TYPES['REWRITE']: "Rewrite the following text to be more {style}:",
        }
        
        if task_type in task_instructions:
            template = task_instructions[task_type]
            # Format the template with any provided variables
            try:
                return template.format(**kwargs)
            except KeyError as e:
                self.logger.warning(f"Missing variable for task template: {e}")
                return "Please complete the following task:"
        
        return "Please complete the following task:"


class LlamaInstruction(BaseInstruction):
    """Instructions optimized for Llama family models."""
    
    def apply(self, **kwargs: Any) -> str:
        """Apply Llama-optimized instruction.
        
        Args:
            **kwargs: May contain 'task_type' and other formatting variables
            
        Returns:
            Instruction formatted for Llama models.
        """
        self.logger.debug("Applying Llama instruction")
        task_type = kwargs.get('task_type', '')
        
        # Llama models perform well with clear, direct instructions
        task_instructions = {
            TASK_TYPES['SUMMARIZE']: "Summarize the following text in a concise way:",
            TASK_TYPES['TRANSLATE']: "Translate this text from {source_language} to {target_language}:",
            TASK_TYPES['ANSWER']: "Answer this question with accurate information:",
            TASK_TYPES['GENERATE']: "Write {content_type} about the following topic:",
            TASK_TYPES['CLASSIFY']: "Classify this text into one of these categories ({categories}):",
            TASK_TYPES['EXTRACT']: "Extract all {entity_type} from this text:",
            TASK_TYPES['CODE']: "Write code in {programming_language} that accomplishes the following:",
            TASK_TYPES['CHAT']: "You are having a conversation. Respond naturally to the following message:",
            TASK_TYPES['BRAINSTORM']: "Generate a list of creative ideas about the following topic:",
        }
        
        if task_type in task_instructions:
            template = task_instructions[task_type]
            try:
                return template.format(**kwargs)
            except KeyError as e:
                self.logger.warning(f"Missing variable for Llama task template: {e}")
                return "Follow these instructions:"
        
        return "Follow these instructions:"


class MistralInstruction(BaseInstruction):
    """Instructions optimized for Mistral family models."""
    
    def apply(self, **kwargs: Any) -> str:
        """Apply Mistral-optimized instruction.
        
        Args:
            **kwargs: May contain 'task_type' and other formatting variables
            
        Returns:
            Instruction formatted for Mistral models.
        """
        self.logger.debug("Applying Mistral instruction")
        task_type = kwargs.get('task_type', '')
        
        # Mistral models work well with detailed instructions
        task_instructions = {
            TASK_TYPES['SUMMARIZE']: "Create a comprehensive summary of the following text, highlighting the main points and key details:",
            TASK_TYPES['TRANSLATE']: "Translate the following text from {source_language} to {target_language}, maintaining the original meaning and tone:",
            TASK_TYPES['ANSWER']: "Provide a detailed and accurate answer to the following question, citing relevant information:",
            TASK_TYPES['GENERATE']: "Create {content_type} based on the following specifications. Be creative yet precise:",
            TASK_TYPES['CLASSIFY']: "Analyze the following text and classify it into one of these categories: {categories}. Explain your reasoning:",
            TASK_TYPES['EXTRACT']: "Extract all instances of {entity_type} from the following text and list them in order of appearance:",
            TASK_TYPES['CODE']: "Write efficient and well-documented code in {programming_language} that implements the following functionality:",
            TASK_TYPES['ANALYZE']: "Perform a thorough analysis of the following information, considering all relevant factors:",
            TASK_TYPES['COMPARE']: "Compare and contrast the following items, highlighting similarities and differences:",
            TASK_TYPES['CRITIQUE']: "Provide a constructive critique of the following, identifying strengths and areas for improvement:",
        }
        
        if task_type in task_instructions:
            template = task_instructions[task_type]
            try:
                return template.format(**kwargs)
            except KeyError as e:
                self.logger.warning(f"Missing variable for Mistral task template: {e}")
                return "Please follow these instructions carefully:"
        
        return "Please follow these instructions carefully:"


# Factory function to get appropriate instruction for a model
def get_model_instruction(model_name: str) -> BaseInstruction:
    """Get the appropriate instruction for a specific model.
    
    Args:
        model_name: Name or identifier of the model
        
    Returns:
        An instruction instance optimized for the model.
    """
    model_name = model_name.lower()
    
    if "alpaca" in model_name:
        return AlpacaInstruction()
    elif "llama-2" in model_name or "llama2" in model_name:
        return LlamaInstruction()
    elif "llama-3" in model_name or "llama3" in model_name:
        return LlamaInstruction()  # Using same base but could be specialized
    elif "mistral" in model_name:
        return MistralInstruction()
    else:
        return DefaultInstruction()


# Common task-specific instruction sets
def get_summarization_instructions() -> InstructionSet:
    """Get instructions for summarization tasks."""
    return InstructionSet(
        SimpleInstruction("Read the following text carefully."),
        TaskSpecificInstruction({
            'concise': "Create a brief summary capturing the main points.",
            'detailed': "Create a comprehensive summary including key details and supporting points.",
            'bullet': "Extract the main points and list them as bullet points."
        }, default="Create a summary of the following text.")
    )


def get_qa_instructions() -> InstructionSet:
    """Get instructions for question-answering tasks."""
    return InstructionSet(
        SimpleInstruction("Answer the following question accurately and completely."),
        TemplatedInstruction("If the answer is not contained in the provided context, respond with '{unknown_response}'.", 
                            required_vars=["unknown_response"])
    )


def get_translation_instructions() -> InstructionSet:
    """Get instructions for translation tasks."""
    return InstructionSet(
        TemplatedInstruction("Translate the following text from {source_language} to {target_language}.", 
                            required_vars=["source_language", "target_language"]),
        SimpleInstruction("Maintain the original meaning, tone, and style as much as possible.")
    )
