"""
Instruction manager for integrating instructions with template composer.

This module provides the InstructionManager class that coordinates between
instruction sets and the template composer to create complete prompts.
"""
from typing import Dict, Any, Optional, Union, List
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .base import BaseInstruction, InstructionSet
from .model_instructions import get_model_instruction, TASK_TYPES
from ..templates.composer import PromptComposer
from .datasets import DatasetHandler

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
        model_name: Optional[str] = None,
        dataset_handler: Optional[DatasetHandler] = None
    ):
        """Initialize the instruction manager.
        
        Args:
            composer: The PromptComposer instance to use
            default_instruction: Default instruction to use if none specified
            model_name: Name of the model to get optimized instructions for
            dataset_handler: Optional DatasetHandler instance (creates one if None)
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
        
        # Set up dataset handler
        self.dataset_handler = dataset_handler or DatasetHandler()
        
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
    
    def process_dataset(
        self,
        dataset_source: Union[str, Dataset],
        instruction_column: Optional[str] = None,
        input_column: Optional[str] = None,
        label_column: Optional[str] = None,
        task_type_column: Optional[str] = None,
        combined_column: Optional[str] = None,
        task_name: Optional[str] = None,
        instruction_kwargs: Optional[Dict[str, Any]] = None,
        output_column: str = "processed_input",
        output_label_column: str = "processed_label",
        map_task_types: bool = True,
        **kwargs
    ) -> Dataset:
        """Process a dataset with instructions and prepare it for training or inference.
        
        This method handles both Hugging Face and local JSON datasets.
        
        Args:
            dataset_source: Path to JSON file, Hugging Face dataset name, or Dataset object
            instruction_column: Column containing instructions
            input_column: Column containing inputs
            label_column: Optional column containing labels/responses
            task_type_column: Optional column containing task types
            combined_column: Optional column containing both instruction and input
            task_name: Optional task name to select instruction
            instruction_kwargs: Additional arguments for the instruction
            output_column: Column to store processed inputs
            output_label_column: Column to store processed labels
            map_task_types: Whether to map dataset task types to standardized TASK_TYPES
            **kwargs: Additional arguments for dataset loading and processing
            
        Returns:
            Processed dataset ready for training or inference
        """
        # Load dataset if source is a string
        if isinstance(dataset_source, str):
            dataset = self.dataset_handler.load_dataset(dataset_source, **kwargs)
        else:
            dataset = dataset_source
        
        self.logger.info(f"Processing dataset with {len(dataset)} items")
        
        # Map task types if requested
        if map_task_types and task_type_column and task_type_column in dataset.features:
            # Infer mapping from dataset task types to standardized ones
            mapping = self.dataset_handler.infer_task_type_mapping(
                dataset, task_type_column
            )
            
            # Apply mapping
            if mapping:
                self.logger.info(f"Mapping {len(mapping)} task types to standardized TASK_TYPES")
                if isinstance(dataset_source, str) and dataset_source.endswith('.json'):
                    dataset = self.dataset_handler.json_handler.map_custom_task_types(
                        dataset, task_type_column, mapping
                    )
                else:
                    dataset = self.dataset_handler.huggingface_handler.map_task_types(
                        dataset, task_type_column, mapping
                    )
        
        # Get the appropriate instruction
        instruction = self.get_instruction_for_task(task_name)
        
        # Process dataset based on available columns
        if label_column:
            # Training mode with labels
            self.logger.info("Processing dataset for training (with labels)")
            dataset = self.dataset_handler.prepare_for_training(
                dataset=dataset,
                instruction_column=instruction_column,
                input_column=input_column,
                label_column=label_column,
                instruction=instruction,
                task_type_column=task_type_column,
                additional_kwargs=instruction_kwargs
            )
            
            # Rename output columns if needed
            if 'processed_input' != output_column:
                dataset = dataset.rename_column('processed_input', output_column)
            if 'processed_label' != output_label_column:
                dataset = dataset.rename_column('processed_label', output_label_column)
                
        else:
            # Inference mode without labels
            self.logger.info("Processing dataset for inference (no labels)")
            dataset = self.dataset_handler.apply_instruction_to_dataset(
                dataset=dataset,
                instruction=instruction,
                instruction_column=instruction_column,
                input_column=input_column,
                output_column=output_column,
                task_type_column=task_type_column,
                additional_kwargs=instruction_kwargs
            )
        
        return dataset
    
    def batch_compose(
        self,
        dataset: Dataset,
        input_column: str,
        system_prompt: Optional[str] = None,
        label_column: Optional[str] = None,
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Compose prompts for a batch of dataset items.
        
        Args:
            dataset: The dataset to process
            input_column: Column containing processed inputs
            system_prompt: Optional system prompt
            label_column: Optional column containing labels
            batch_size: Number of items to process at once
            **kwargs: Additional arguments for the composer
            
        Returns:
            List of composed prompt dictionaries
        """
        self.logger.info(f"Batch composing prompts for {len(dataset)} items")
        
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            
            for item in batch:
                user_input = item[input_column]
                label = item[label_column] if label_column and label_column in item else None
                
                # Use the composer to create the prompt
                result = self.composer.compose(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    label=label,
                    **kwargs
                )
                
                results.append(result)
        
        return results
    
    def __call__(self, *args, **kwargs):
        """Alias for compose_with_instruction method."""
        return self.compose_with_instruction(*args, **kwargs)
