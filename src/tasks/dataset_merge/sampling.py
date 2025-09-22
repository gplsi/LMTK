"""
Dataset sampling utilities for merge operations.
"""

from typing import Union
from datasets import Dataset, DatasetDict
import logging


class DatasetSampler:
    """Handles dataset sampling operations."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def sample_dataset(self, dataset: Union[Dataset, DatasetDict], 
                      percentage: float, shuffle_seed: int = 42) -> Union[Dataset, DatasetDict]:
        """
        Sample a percentage from a dataset.
        
        Args:
            dataset: Dataset to sample from
            percentage: Percentage to sample (can be > 1.0 for oversampling)
            shuffle_seed: Seed for reproducible shuffling
            
        Returns:
            Sampled dataset
        """
        if isinstance(dataset, DatasetDict):
            return self._sample_dataset_dict(dataset, percentage, shuffle_seed)
        else:
            return self._sample_single_dataset(dataset, percentage, shuffle_seed)
    
    def _sample_dataset_dict(self, dataset_dict: DatasetDict, 
                           percentage: float, shuffle_seed: int) -> DatasetDict:
        """Sample from a DatasetDict by sampling each split."""
        sampled_splits = {}
        
        for split_name, split_dataset in dataset_dict.items():
            sampled_split = self._sample_single_dataset(split_dataset, percentage, shuffle_seed)
            sampled_splits[split_name] = sampled_split
            
            self.logger.info(
                f"Split '{split_name}': {len(split_dataset):,} → {len(sampled_split):,} "
                f"({percentage*100:.1f}%)"
            )
        
        return DatasetDict(sampled_splits)
    
    def _sample_single_dataset(self, dataset: Dataset, 
                             percentage: float, shuffle_seed: int) -> Dataset:
        """Sample from a single dataset."""
        total_size = len(dataset)
        target_size = int(total_size * percentage)
        
        # Shuffle first
        shuffled = dataset.shuffle(seed=shuffle_seed)
        
        if target_size >= total_size:
            # No sampling needed or oversampling
            if target_size == total_size:
                return shuffled
            else:
                # Oversample by repeating and taking subset
                num_repeats = (target_size // total_size) + 1
                repeated_indices = (list(range(total_size)) * num_repeats)[:target_size]
                return shuffled.select(repeated_indices)
        else:
            # Regular sampling
            return shuffled.select(range(target_size))
