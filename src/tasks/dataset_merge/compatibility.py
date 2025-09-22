"""
Dataset compatibility validation utilities.
"""

from typing import List, Dict, Union
from datasets import Dataset, DatasetDict, Features
import logging


class DatasetCompatibilityChecker:
    """Handles all dataset compatibility validation logic."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_compatibility(self, datasets: List[Union[Dataset, DatasetDict]]) -> Dict:
        """
        Check if datasets are compatible for merging.
        
        Args:
            datasets: List of datasets to check
            
        Returns:
            Dict with compatibility info
            
        Raises:
            ValueError: If datasets are incompatible
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        # Use first dataset as reference
        reference = datasets[0]
        reference_info = self._extract_dataset_info(reference)
        
        self.logger.info(f"Reference dataset: {reference_info['type']}")
        self.logger.info(f"Reference splits: {reference_info['splits']}")
        self.logger.info(f"Reference features: {list(reference_info['features'].keys())}")
        
        # Validate all other datasets against reference
        for i, dataset in enumerate(datasets[1:], 1):
            current_info = self._extract_dataset_info(dataset)
            self._validate_against_reference(current_info, reference_info, i)
        
        self.logger.info(f"✓ All {len(datasets)} datasets are compatible")
        return reference_info
    
    def _extract_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict:
        """Extract key information from a dataset."""
        if isinstance(dataset, DatasetDict):
            # For DatasetDict, use 'train' split as reference, or first available split
            reference_split = dataset.get('train') or list(dataset.values())[0]
            return {
                'type': 'DatasetDict',
                'splits': sorted(dataset.keys()),
                'features': reference_split.features,
                'total_examples': sum(len(split) for split in dataset.values())
            }
        else:
            return {
                'type': 'Dataset',
                'splits': ['train'],  # Single dataset treated as 'train' split
                'features': dataset.features,
                'total_examples': len(dataset)
            }
    
    def _validate_against_reference(self, current_info: Dict, reference_info: Dict, dataset_idx: int):
        """Validate current dataset against reference dataset."""
        
        # Check dataset type compatibility
        if current_info['type'] != reference_info['type']:
            raise ValueError(
                f"Dataset {dataset_idx} type mismatch: "
                f"expected {reference_info['type']}, got {current_info['type']}"
            )
        
        # Check feature compatibility
        if current_info['features'] != reference_info['features']:
            ref_keys = set(reference_info['features'].keys())
            curr_keys = set(current_info['features'].keys())
            
            if ref_keys != curr_keys:
                raise ValueError(
                    f"Dataset {dataset_idx} feature keys mismatch: "
                    f"expected {sorted(ref_keys)}, got {sorted(curr_keys)}"
                )
            
            # Check feature types
            for key in ref_keys:
                if reference_info['features'][key] != current_info['features'][key]:
                    raise ValueError(
                        f"Dataset {dataset_idx} feature '{key}' type mismatch: "
                        f"expected {reference_info['features'][key]}, "
                        f"got {current_info['features'][key]}"
                    )
        
        # For DatasetDict, we allow different splits - they'll be merged appropriately
        if current_info['type'] == 'DatasetDict':
            self.logger.info(f"Dataset {dataset_idx} splits: {current_info['splits']}")
        
        self.logger.info(f"✓ Dataset {dataset_idx} compatible ({current_info['total_examples']:,} examples)")
