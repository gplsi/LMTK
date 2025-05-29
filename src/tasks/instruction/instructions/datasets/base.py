"""
Base dataset handler with common functionality.

This module provides the base class and common utilities for dataset handlers.
"""
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from abc import ABC, abstractmethod
from datasets import Dataset
from ..base import BaseInstruction, InstructionSet
from utils.logger import get_logger

class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers."""
    
    def __init__(self):
        """Initialize the dataset handler."""
        self.logger = get_logger()
    
    @abstractmethod
    def load_dataset(self, *args, **kwargs) -> Dataset:
        """Load a dataset from a source.
        
        This method must be implemented by subclasses.
        """
        pass
    
    def extract_instruction_input_pairs(
        self,
        dataset: Dataset,
        instruction_column: Optional[str] = None,
        input_column: Optional[str] = None,
        combined_column: Optional[str] = None,
        split_function: Optional[Callable[[str], Tuple[str, str]]] = None,
        instruction_prefix: Optional[str] = None,
        input_prefix: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Extract instruction-input pairs from dataset.
        
        Args:
            dataset: The dataset to process
            instruction_column: Column containing instructions
            input_column: Column containing inputs
            combined_column: Column containing both instruction and input
            split_function: Function to split combined text into instruction and input
            instruction_prefix: Prefix to identify instruction in combined text
            input_prefix: Prefix to identify input in combined text
            max_samples: Maximum number of samples to process
            
        Returns:
            List of dictionaries with 'instruction' and 'input' keys
            
        Raises:
            ValueError: If neither separate columns nor combined column with split info is provided
        """
        self.logger.info("Extracting instruction-input pairs from dataset")
        
        # Validate input configuration
        if not ((instruction_column and input_column) or 
                (combined_column and (split_function or (instruction_prefix and input_prefix)))):
            raise ValueError(
                "Must provide either: "
                "1) both instruction_column and input_column, or "
                "2) combined_column with either split_function or both instruction_prefix and input_prefix"
            )
        
        # Limit dataset size if requested
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            self.logger.info(f"Limited dataset to {max_samples} samples")
        
        result = []
        
        # Process dataset based on column configuration
        if instruction_column and input_column:
            self.logger.info(f"Using separate columns: {instruction_column} and {input_column}")
            for item in dataset:
                if instruction_column in item and input_column in item:
                    result.append({
                        'instruction': item[instruction_column],
                        'input': item[input_column]
                    })
                else:
                    self.logger.warning(f"Missing columns in item: {item.keys()}")
        
        elif combined_column and split_function:
            self.logger.info(f"Using combined column with split function: {combined_column}")
            for item in dataset:
                if combined_column in item:
                    try:
                        instruction, input_text = split_function(item[combined_column])
                        result.append({
                            'instruction': instruction,
                            'input': input_text
                        })
                    except Exception as e:
                        self.logger.warning(f"Error splitting text: {e}")
        
        elif combined_column and instruction_prefix and input_prefix:
            self.logger.info(f"Using combined column with prefixes: {combined_column}")
            for item in dataset:
                if combined_column in item:
                    text = item[combined_column]
                    # Simple split by prefixes
                    try:
                        parts = text.split(instruction_prefix, 1)
                        if len(parts) > 1:
                            instruction_with_input = parts[1]
                            instruction_input_parts = instruction_with_input.split(input_prefix, 1)
                            if len(instruction_input_parts) > 1:
                                instruction = instruction_input_parts[0].strip()
                                input_text = instruction_input_parts[1].strip()
                                result.append({
                                    'instruction': instruction,
                                    'input': input_text
                                })
                            else:
                                # No input part found, treat everything as instruction
                                result.append({
                                    'instruction': instruction_with_input.strip(),
                                    'input': ""
                                })
                    except Exception as e:
                        self.logger.warning(f"Error splitting text with prefixes: {e}")
        
        self.logger.info(f"Extracted {len(result)} instruction-input pairs")
        return result
    
    def apply_instruction_to_dataset(
        self,
        dataset: Dataset,
        instruction: Union[BaseInstruction, InstructionSet],
        instruction_column: str,
        input_column: str,
        output_column: str = "processed_input",
        task_type_column: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Apply instruction to dataset items.
        
        Args:
            dataset: The dataset to process
            instruction: The instruction to apply
            instruction_column: Column containing instructions
            input_column: Column containing inputs
            output_column: Column to store processed inputs
            task_type_column: Optional column containing task types
            additional_kwargs: Additional arguments for instruction.apply
            
        Returns:
            Dataset with processed inputs
        """
        self.logger.info(f"Applying instruction to dataset with {len(dataset)} items")
        
        def process_item(item):
            kwargs = additional_kwargs.copy() if additional_kwargs else {}
            
            # Add task_type if available
            if task_type_column and task_type_column in item:
                kwargs['task_type'] = item[task_type_column]
            
            # Apply instruction to input
            instruction_text = instruction.apply(**kwargs)
            if instruction_text:
                item[output_column] = f"{instruction_text}\n\n{item[input_column]}"
            else:
                item[output_column] = item[input_column]
            
            return item
        
        return dataset.map(process_item)
    
    def prepare_for_training(
        self,
        dataset: Dataset,
        instruction_column: str,
        input_column: str,
        label_column: str,
        instruction: Optional[Union[BaseInstruction, InstructionSet]] = None,
        task_type_column: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Prepare dataset for training with instructions.
        
        Args:
            dataset: The dataset to process
            instruction_column: Column containing instructions
            input_column: Column containing inputs
            label_column: Column containing labels/responses
            instruction: Optional instruction to apply (if None, uses raw instruction from column)
            task_type_column: Optional column containing task types
            additional_kwargs: Additional arguments for instruction.apply
            
        Returns:
            Dataset prepared for training
        """
        self.logger.info(f"Preparing dataset for training with {len(dataset)} items")
        
        def prepare_item(item):
            # Get instruction text
            if instruction:
                kwargs = additional_kwargs.copy() if additional_kwargs else {}
                
                # Add task_type if available
                if task_type_column and task_type_column in item:
                    kwargs['task_type'] = item[task_type_column]
                
                instruction_text = instruction.apply(**kwargs)
            else:
                instruction_text = item[instruction_column]
            
            # Combine instruction with input
            if instruction_text:
                item['processed_input'] = f"{instruction_text}\n\n{item[input_column]}"
            else:
                item['processed_input'] = item[input_column]
            
            # Keep the label
            item['processed_label'] = item[label_column]
            
            return item
        
        return dataset.map(prepare_item)
