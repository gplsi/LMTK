"""
Dataset sampling utilities for merge operations.
"""

import os
from pathlib import Path
from typing import Optional, Union
from datasets import Dataset, DatasetDict
import logging


class DatasetSampler:
    """Handles dataset sampling operations."""
    
    def __init__(
        self,
        logger=None,
        shuffle_full_sample: bool = True,
        indices_cache_dir: Optional[str] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.shuffle_full_sample = shuffle_full_sample
        self.indices_cache_dir = indices_cache_dir
        self._shuffle_counter = 0
    
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
        is_full_sample = self._is_full_percentage(percentage)

        if is_full_sample and not self.shuffle_full_sample:
            self.logger.debug(
                "Skipping shuffle for dataset (percentage=%.3f, seed=%s)",
                percentage,
                shuffle_seed,
            )
            return dataset

        shuffle_kwargs = {}
        if self.indices_cache_dir:
            os.makedirs(self.indices_cache_dir, exist_ok=True)
            cache_file = Path(self.indices_cache_dir) / f"shuffle_{self._shuffle_counter:04d}.arrow"
            shuffle_kwargs["indices_cache_file_name"] = str(cache_file)
        
        # Shuffle first (may be identity if percentage == 1 and shuffle_full_sample is True)
        shuffled = dataset.shuffle(seed=shuffle_seed, **shuffle_kwargs)
        self._shuffle_counter += 1
        
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

    @staticmethod
    def _is_full_percentage(percentage: Union[int, float]) -> bool:
        try:
            return abs(float(percentage) - 1.0) < 1e-9
        except (TypeError, ValueError):
            return False
