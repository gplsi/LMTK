"""
Dataset handlers for instruction datasets.

This module provides functionality to load and process datasets from various sources.
"""
from .base import BaseDatasetHandler
from .huggingface import HuggingFaceDatasetHandler
from .json_dataset import JsonDatasetHandler

__all__ = [
    'BaseDatasetHandler',
    'HuggingFaceDatasetHandler',
    'JsonDatasetHandler',
    'DatasetHandler',
]

class DatasetHandler:
    """Factory class that provides access to different dataset handlers."""
    
    def __init__(self):
        """Initialize the dataset handler."""
        self.huggingface_handler = HuggingFaceDatasetHandler()
        self.json_handler = JsonDatasetHandler()
    
    def load_huggingface_dataset(self, *args, **kwargs):
        """Load a dataset from Hugging Face.
        
        See HuggingFaceDatasetHandler.load_dataset for parameters.
        """
        return self.huggingface_handler.load_dataset(*args, **kwargs)
    
    def load_json_dataset(self, *args, **kwargs):
        """Load a dataset from a local JSON file.
        
        See JsonDatasetHandler.load_dataset for parameters.
        """
        return self.json_handler.load_dataset(*args, **kwargs)
    
    def load_dataset(self, source, **kwargs):
        """Load a dataset from a source.
        
        This method automatically determines the appropriate handler based on the source.
        
        Args:
            source: Path to a JSON file or name of a Hugging Face dataset
            **kwargs: Additional arguments for the specific handler
            
        Returns:
            The loaded dataset
        """
        import os
        
        if os.path.exists(source) and source.endswith('.json'):
            return self.load_json_dataset(source, **kwargs)
        else:
            return self.load_huggingface_dataset(source, **kwargs)
    
    def extract_instruction_input_pairs(self, dataset, **kwargs):
        """Extract instruction-input pairs from a dataset.
        
        This delegates to the base handler's implementation.
        
        See BaseDatasetHandler.extract_instruction_input_pairs for parameters.
        """
        return self.huggingface_handler.extract_instruction_input_pairs(dataset, **kwargs)
    
    def apply_instruction_to_dataset(self, dataset, instruction, **kwargs):
        """Apply an instruction to a dataset.
        
        This delegates to the base handler's implementation.
        
        See BaseDatasetHandler.apply_instruction_to_dataset for parameters.
        """
        return self.huggingface_handler.apply_instruction_to_dataset(
            dataset, instruction, **kwargs
        )
    
    def prepare_for_training(self, dataset, **kwargs):
        """Prepare a dataset for training.
        
        This delegates to the base handler's implementation.
        
        See BaseDatasetHandler.prepare_for_training for parameters.
        """
        return self.huggingface_handler.prepare_for_training(dataset, **kwargs)
    
    def save_dataset_to_json(self, dataset, output_path, **kwargs):
        """Save a dataset to a JSON file.
        
        This delegates to the JSON handler's implementation.
        
        See JsonDatasetHandler.save_dataset_to_json for parameters.
        """
        return self.json_handler.save_dataset_to_json(dataset, output_path, **kwargs)
