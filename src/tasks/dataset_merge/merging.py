"""
Dataset merge operations for combining multiple datasets.
"""

from typing import List, Union
from datasets import Dataset, DatasetDict, concatenate_datasets
import logging


class DatasetMerger:
    """Handles the actual merging of datasets."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def merge_datasets(self, datasets: List[Union[Dataset, DatasetDict]], 
                      final_shuffle_seed: int = None) -> Union[Dataset, DatasetDict]:
        """
        Merge multiple datasets.
        
        Args:
            datasets: List of datasets to merge
            final_shuffle_seed: Seed for final shuffle
            
        Returns:
            Merged dataset
        """
        if not datasets:
            raise ValueError("No datasets to merge")
        
        # Check if all datasets are the same type
        first_type = type(datasets[0])
        if not all(isinstance(d, first_type) for d in datasets):
            raise ValueError("All datasets must be the same type")
        
        if isinstance(datasets[0], DatasetDict):
            return self._merge_dataset_dicts(datasets, final_shuffle_seed)
        else:
            return self._merge_single_datasets(datasets, final_shuffle_seed)
    
    def _merge_dataset_dicts(self, dataset_dicts: List[DatasetDict], 
                           final_shuffle_seed: int) -> DatasetDict:
        """Merge multiple DatasetDicts by combining corresponding splits."""
        
        # Collect all unique split names
        all_splits = set()
        for dataset_dict in dataset_dicts:
            all_splits.update(dataset_dict.keys())
        
        merged_splits = {}
        
        for split_name in sorted(all_splits):
            self.logger.info(f"Merging split: {split_name}")
            
            # Collect all datasets that have this split
            split_datasets = []
            contributions = []
            
            for i, dataset_dict in enumerate(dataset_dicts):
                if split_name in dataset_dict:
                    split_data = dataset_dict[split_name]
                    split_datasets.append(split_data)
                    contributions.append(f"Dataset {i+1}: {len(split_data):,}")
            
            if split_datasets:
                self.logger.info(f"  Contributions: {contributions}")
                
                # Concatenate datasets for this split
                merged_split = concatenate_datasets(split_datasets)
                
                # Shuffle the merged split
                if final_shuffle_seed is not None:
                    split_seed = final_shuffle_seed + hash(split_name) % 1000
                    merged_split = merged_split.shuffle(seed=split_seed)
                
                merged_splits[split_name] = merged_split
                self.logger.info(f"  Result: {len(merged_split):,} examples")
        
        total_examples = sum(len(split) for split in merged_splits.values())
        self.logger.info(f"✓ DatasetDict merged: {total_examples:,} total examples across {len(merged_splits)} splits")
        
        return DatasetDict(merged_splits)
    
    def _merge_single_datasets(self, datasets: List[Dataset], 
                             final_shuffle_seed: int) -> Dataset:
        """Merge multiple single datasets."""
        
        total_before = sum(len(d) for d in datasets)
        self.logger.info(f"Merging {len(datasets)} single datasets ({total_before:,} total examples)")
        
        # Concatenate all datasets
        merged = concatenate_datasets(datasets)
        
        # Final shuffle
        if final_shuffle_seed is not None:
            merged = merged.shuffle(seed=final_shuffle_seed)
        
        self.logger.info(f"✓ Single datasets merged: {len(merged):,} examples")
        return merged
