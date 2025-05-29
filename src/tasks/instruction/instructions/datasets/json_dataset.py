"""
JSON dataset handler.

This module provides functionality to load and process datasets from local JSON files.
"""
from typing import Dict, Any, Optional, List
import json
import os
from datasets import Dataset
from .base import BaseDatasetHandler
from ..model_instructions import TASK_TYPES

class JsonDatasetHandler(BaseDatasetHandler):
    """Handler for local JSON datasets."""
    
    def load_dataset(
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
    
    def save_dataset_to_json(
        self,
        dataset: Dataset,
        output_path: str,
        pretty_print: bool = True
    ) -> None:
        """Save a dataset to a JSON file.
        
        Args:
            dataset: The dataset to save
            output_path: Path to save the JSON file
            pretty_print: Whether to format the JSON with indentation
        """
        self.logger.info(f"Saving dataset with {len(dataset)} items to {output_path}")
        
        # Convert dataset to list of dictionaries
        data = [dict(item) for item in dataset]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        self.logger.info(f"Dataset saved successfully to {output_path}")