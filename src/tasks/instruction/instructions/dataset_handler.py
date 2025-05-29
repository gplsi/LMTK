"""
Dataset handler for working with Hugging Face datasets and local JSON datasets.

This module provides functionality to process datasets and extract
instruction-input pairs from different column configurations.
"""
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Set
import json
import os
import datasets
from datasets import Dataset, DatasetDict
from .base import BaseInstruction, InstructionSet
from .model_instructions import TASK_TYPES
from utils.logger import get_logger

class DatasetHandler:
    """Handler for processing datasets and extracting instruction-input pairs."""
    
    def __init__(self):
        """Initialize the dataset handler."""
        self.logger = get_logger()
    
    def load_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        **dataset_kwargs
    ) -> Union[Dataset, DatasetDict]:
        """Load a dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            split: Optional dataset split to load
            **dataset_kwargs: Additional arguments for datasets.load_dataset
            
        Returns:
            The loaded dataset
        """
        self.logger.info(f"Loading dataset: {dataset_name}")
        try:
            if split:
                return datasets.load_dataset(dataset_name, split=split, **dataset_kwargs)
            return datasets.load_dataset(dataset_name, **dataset_kwargs)
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def load_json_dataset(
        self,
        file_path: str,
        instruction_key: str = "instruction",
        input_key: str = "input",
        label_key: Optional[str] = "output",
        task_type_key: Optional[str] = "task_type",
        task_type_mapping: Optional[Dict[str, str]] = None
    ) -> Dataset:
        """Load a dataset from a local JSON file.
        
        Args:
            file_path: Path to the JSON file
            instruction_key: Key for instruction field in JSON
            input_key: Key for input field in JSON
            label_key: Optional key for label/output field in JSON
            task_type_key: Optional key for task type field in JSON
            task_type_mapping: Optional mapping from custom task types to TASK_TYPES
            
        Returns:
            The loaded dataset as a Hugging Face Dataset
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file is invalid
        """
        self.logger.info(f"Loading JSON dataset from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        
        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            if isinstance(data, dict) and any(isinstance(data.get(k), list) for k in data):
                # Handle case where data is wrapped in an object with a list property
                for k, v in data.items():
                    if isinstance(v, list):
                        data = v
                        self.logger.info(f"Extracted list from key '{k}' with {len(data)} items")
                        break
            else:
                raise ValueError("JSON data must be a list of objects or an object with a list property")
        
        # Process and validate the data
        processed_data = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                self.logger.warning(f"Skipping non-object item at index {idx}")
                continue
            
            processed_item = {}
            
            # Check required fields
            if instruction_key not in item:
                self.logger.warning(f"Missing instruction key '{instruction_key}' at index {idx}")
                continue
            
            processed_item[instruction_key] = item[instruction_key]
            
            # Input field (may be empty)
            processed_item[input_key] = item.get(input_key, "")
            
            # Optional label field
            if label_key and label_key in item:
                processed_item[label_key] = item[label_key]
            
            # Process task type if present
            if task_type_key and task_type_key in item:
                original_task_type = item[task_type_key]
                
                # Map custom task type to standardized task type if mapping provided
                if task_type_mapping and original_task_type in task_type_mapping:
                    processed_item[task_type_key] = task_type_mapping[original_task_type]
                    self.logger.debug(f"Mapped task type '{original_task_type}' to '{processed_item[task_type_key]}'")
                else:
                    processed_item[task_type_key] = original_task_type
            
            # Add any additional fields
            for k, v in item.items():
                if k not in processed_item:
                    processed_item[k] = v
            
            processed_data.append(processed_item)
        
        self.logger.info(f"Loaded {len(processed_data)} items from JSON file")
        
        # Convert to Hugging Face Dataset
        return Dataset.from_list(processed_data)
    
    def map_custom_task_types(
        self,
        dataset: Dataset,
        task_type_column: str,
        mapping: Dict[str, str],
        target_column: Optional[str] = None
    ) -> Dataset:
        """Map custom task types to standardized TASK_TYPES.
        
        Args:
            dataset: The dataset to process
            task_type_column: Column containing task types
            mapping: Mapping from custom task types to TASK_TYPES values
            target_column: Optional column to store mapped task types
                          (if None, overwrites the original column)
            
        Returns:
            Dataset with mapped task types
        """
        target_column = target_column or task_type_column
        
        def map_task_type(item):
            if task_type_column in item and item[task_type_column] in mapping:
                item[target_column] = mapping[item[task_type_column]]
            return item
        
        return dataset.map(map_task_type)
    
    def get_task_type_mapping(
        self,
        dataset: Dataset,
        task_type_column: str,
        case_insensitive: bool = True
    ) -> Dict[str, str]:
        """Generate a mapping from custom task types to standardized TASK_TYPES.
        
        This attempts to match custom task types to the standardized ones based on
        string similarity.
        
        Args:
            dataset: The dataset to analyze
            task_type_column: Column containing task types
            case_insensitive: Whether to ignore case when matching
            
        Returns:
            Mapping from custom task types to standardized TASK_TYPES
        """
        # Get unique task types from the dataset
        custom_types = set()
        for item in dataset:
            if task_type_column in item and item[task_type_column]:
                custom_types.add(item[task_type_column])
        
        # Create mapping
        mapping = {}
        standard_types = {v: k for k, v in TASK_TYPES.items()}
        
        for custom_type in custom_types:
            # Try exact match first
            custom_lower = custom_type.lower() if case_insensitive else custom_type
            
            # Check for exact match
            for std_type, std_key in standard_types.items():
                std_lower = std_type.lower() if case_insensitive else std_type
                if custom_lower == std_lower:
                    mapping[custom_type] = std_type
                    break
            
            # If no exact match, try substring match
            if custom_type not in mapping:
                for std_type, std_key in standard_types.items():
                    std_lower = std_type.lower() if case_insensitive else std_type
                    if custom_lower in std_lower or std_lower in custom_lower:
                        mapping[custom_type] = std_type
                        break
        
        self.logger.info(f"Generated mapping for {len(mapping)}/{len(custom_types)} custom task types")
        return mapping
    
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
