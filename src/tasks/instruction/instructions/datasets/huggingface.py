"""
Hugging Face dataset handler.

This module provides functionality to load and process datasets from Hugging Face.
"""
from typing import Dict, Any, Optional, Union
import datasets
from datasets import Dataset, DatasetDict
from .base import BaseDatasetHandler
from ..model_instructions import TASK_TYPES

class HuggingFaceDatasetHandler(BaseDatasetHandler):
    """Handler for Hugging Face datasets."""
    
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
        self.logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        try:
            if split:
                return datasets.load_dataset(dataset_name, split=split, **dataset_kwargs)
            return datasets.load_dataset(dataset_name, **dataset_kwargs)
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def map_task_types(
        self,
        dataset: Dataset,
        task_type_column: str,
        mapping: Dict[str, str],
        target_column: Optional[str] = None
    ) -> Dataset:
        """Map dataset task types to standardized TASK_TYPES.
        
        Args:
            dataset: The dataset to process
            task_type_column: Column containing task types
            mapping: Mapping from dataset task types to TASK_TYPES values
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