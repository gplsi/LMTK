"""
Dataset Merge Orchestrator Module

This module provides the DatasetMergeOrchestrator class, which handles the orchestration 
of merging multiple tokenized datasets into a single dataset with specified percentages.
The workflow includes validating dataset compatibility, sampling data according to 
specified percentages, and merging the results.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Union
import random
from box import Box
from datasets import DatasetDict, concatenate_datasets
from datasets import Dataset as HFDataset
from src.utils.logging import get_logger, VerboseLevel
from src.utils.dataset import DatasetStorage
from src.utils.orchestrator import BaseOrchestrator
from src.utils import inherit_init_params


@inherit_init_params
class DatasetMergeOrchestrator(BaseOrchestrator):
    """
    Orchestrator for merging multiple tokenized datasets with specified percentages.
    
    This orchestrator loads multiple tokenized datasets, validates their compatibility,
    samples data according to specified percentages, and creates a merged dataset.
    """
    
    def __init__(self, config: Box) -> None:
        """Initialize the DatasetMergeOrchestrator with the given configuration."""
        super().__init__(config)
        self.logger = get_logger(__name__, level=VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        ))
        
    def validate_config(self) -> None:
        """Validate the configuration specific to dataset merging."""
        super().validate_config()
        
        if not hasattr(self.config, 'datasets') or not self.config.datasets:
            raise ValueError("datasets configuration is required")
        
        if not hasattr(self.config, 'output_path') or not self.config.output_path:
            raise ValueError("output_path is required")
        
        # Validate dataset configurations
        for i, dataset_config in enumerate(self.config.datasets):
            if not hasattr(dataset_config, 'path') or not dataset_config.path:
                raise ValueError(f"Dataset {i}: path is required")
            if not hasattr(dataset_config, 'percentage') or dataset_config.percentage <= 0:
                raise ValueError(f"Dataset {i}: percentage must be > 0")
                
        self.logger.info("Dataset merge configuration validation passed")

    def load_dataset(self) -> HFDataset:
        """This method is not used in dataset merge, implemented for compatibility."""
        raise NotImplementedError("Use load_and_merge_datasets() instead")
    
    def load_and_validate_dataset(self, dataset_path: str) -> Tuple[Union[HFDataset, DatasetDict], Dict]:
        """
        Load a single dataset and extract its metadata for compatibility checking.
        
        Args:
            dataset_path: Path to the tokenized dataset
            
        Returns:
            Tuple of (dataset, metadata_dict)
        """
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(self.config.get("verbose_level", VerboseLevel.INFO))
        )
        
        dataset = dataset_handler.load_from_disk(dataset_path)
        
        # Extract metadata for compatibility checking
        if isinstance(dataset, DatasetDict):
            # Use the train split for metadata extraction
            if 'train' not in dataset:
                raise ValueError(f"Dataset {dataset_path} missing 'train' split")
            sample_dataset = dataset['train']
        else:
            sample_dataset = dataset
            
        metadata = {
            'column_names': sample_dataset.column_names,
            'num_rows': len(sample_dataset),
            'features': sample_dataset.features,
        }
        
        # Validate required columns for tokenized datasets
        required_columns = ['input_ids', 'attention_mask', 'labels']
        missing_columns = [col for col in required_columns if col not in metadata['column_names']]
        if missing_columns:
            raise ValueError(f"Dataset {dataset_path} missing required columns: {missing_columns}")
            
        self.logger.info(f"Dataset loaded: {metadata['num_rows']} rows, columns: {metadata['column_names']}")
        return dataset, metadata
        
    def validate_compatibility(self, datasets_metadata: List[Dict]) -> None:
        """
        Validate that all datasets are compatible for merging.
        
        Args:
            datasets_metadata: List of metadata dictionaries from all datasets
        """
        self.logger.info("Validating dataset compatibility...")
        
        if not datasets_metadata:
            raise ValueError("No datasets provided for validation")
        
        # Use first dataset as reference
        reference = datasets_metadata[0]
        reference_columns = set(reference['column_names'])
        reference_features = reference['features']
        
        for i, metadata in enumerate(datasets_metadata[1:], 1):
            current_columns = set(metadata['column_names'])
            current_features = metadata['features']
            
            # Check column compatibility
            if current_columns != reference_columns:
                raise ValueError(
                    f"Dataset {i} has incompatible columns. "
                    f"Expected: {sorted(reference_columns)}, "
                    f"Got: {sorted(current_columns)}"
                )
            
            # Check feature types compatibility
            for col_name in reference_columns:
                if col_name in current_features and col_name in reference_features:
                    if str(current_features[col_name]) != str(reference_features[col_name]):
                        self.logger.warning(
                            f"Dataset {i} has different feature type for column '{col_name}': "
                            f"{current_features[col_name]} vs {reference_features[col_name]}"
                        )
        
        self.logger.info("All datasets are compatible for merging")
    
    def sample_dataset(self, dataset: Union[HFDataset, DatasetDict], percentage: float, 
                      shuffle_seed: int = None) -> Union[HFDataset, DatasetDict]:
        """
        Sample a percentage of data from a dataset after shuffling.
        
        Args:
            dataset: The dataset to sample from
            percentage: Percentage of data to sample (can be > 1.0)
            shuffle_seed: Seed for shuffling
            
        Returns:
            Sampled dataset
        """
        if isinstance(dataset, DatasetDict):
            sampled_splits = {}
            for split_name, split_dataset in dataset.items():
                sampled_splits[split_name] = self._sample_split(split_dataset, percentage, shuffle_seed)
            return DatasetDict(sampled_splits)
        else:
            return self._sample_split(dataset, percentage, shuffle_seed)
    
    def _sample_split(self, dataset: HFDataset, percentage: float, shuffle_seed: int = None) -> HFDataset:
        """Sample a percentage from a single dataset split."""
        total_size = len(dataset)
        target_size = int(total_size * percentage)
        
        self.logger.info(f"Sampling {target_size} examples ({percentage*100:.1f}%) from {total_size} total")
        
        # Shuffle the dataset first
        if shuffle_seed is not None:
            shuffled_dataset = dataset.shuffle(seed=shuffle_seed)
        else:
            shuffled_dataset = dataset.shuffle()
        
        # Handle case where percentage > 1.0 (oversample)
        if target_size > total_size:
            # Create multiple copies and sample
            num_full_copies = target_size // total_size
            remainder = target_size % total_size
            
            self.logger.info(f"Oversampling: {num_full_copies} full copies + {remainder} additional samples")
            
            # Create full copies
            copies = [shuffled_dataset] * num_full_copies
            
            # Add remainder if needed
            if remainder > 0:
                copies.append(shuffled_dataset.select(range(remainder)))
            
            # Concatenate all copies
            sampled_dataset = concatenate_datasets(copies)
        else:
            # Regular sampling
            sampled_dataset = shuffled_dataset.select(range(target_size))
        
        return sampled_dataset
    
    def merge_datasets(self, datasets: List[Union[HFDataset, DatasetDict]], 
                      final_shuffle_seed: int = None) -> Union[HFDataset, DatasetDict]:
        """
        Merge multiple datasets and shuffle the result.
        
        Args:
            datasets: List of datasets to merge
            final_shuffle_seed: Seed for final shuffling
            
        Returns:
            Merged and shuffled dataset
        """
        self.logger.info(f"Merging {len(datasets)} datasets...")
        
        # Check if all datasets are DatasetDict or single Dataset
        all_dict = all(isinstance(d, DatasetDict) for d in datasets)
        all_single = all(isinstance(d, HFDataset) for d in datasets)
        
        if not (all_dict or all_single):
            raise ValueError("All datasets must be either DatasetDict or single Dataset")
        
        if all_dict:
            # Merge DatasetDicts split by split
            split_names = set()
            for dataset in datasets:
                split_names.update(dataset.keys())
            
            merged_splits = {}
            for split_name in split_names:
                split_datasets = []
                for dataset in datasets:
                    if split_name in dataset:
                        split_datasets.append(dataset[split_name])
                
                if split_datasets:
                    merged_split = concatenate_datasets(split_datasets)
                    # Shuffle each split
                    if final_shuffle_seed is not None:
                        merged_split = merged_split.shuffle(seed=final_shuffle_seed)
                    else:
                        merged_split = merged_split.shuffle()
                    merged_splits[split_name] = merged_split
            
            return DatasetDict(merged_splits)
        else:
            # Merge single datasets
            merged_dataset = concatenate_datasets(datasets)
            # Final shuffle
            if final_shuffle_seed is not None:
                merged_dataset = merged_dataset.shuffle(seed=final_shuffle_seed)
            else:
                merged_dataset = merged_dataset.shuffle()
            return merged_dataset
    
    def execute(self) -> None:
        """Execute the dataset merge workflow."""
        self.logger.info("Starting dataset merge workflow")
        
        # Validate configuration
        self.validate_config()
        
        # Load and validate all datasets
        datasets = []
        datasets_metadata = []
        
        for dataset_config in self.config.datasets:
            dataset, metadata = self.load_and_validate_dataset(dataset_config.path)
            datasets.append(dataset)
            datasets_metadata.append(metadata)
        
        # Validate compatibility
        self.validate_compatibility(datasets_metadata)
        
        # Get sampling configuration
        shuffle_seed = getattr(self.config, 'shuffle_seed', None)
        if shuffle_seed is None:
            shuffle_seed = random.randint(1, 2**32 - 1)
            self.logger.info(f"Using random shuffle seed: {shuffle_seed}")
        
        # Sample datasets according to specified percentages
        sampled_datasets = []
        for i, (dataset, dataset_config) in enumerate(zip(datasets, self.config.datasets)):
            self.logger.info(f"Processing dataset {i+1}: {dataset_config.path} ({dataset_config.percentage*100:.1f}%)")
            sampled_dataset = self.sample_dataset(dataset, dataset_config.percentage, shuffle_seed + i)
            sampled_datasets.append(sampled_dataset)
        
        # Merge all sampled datasets
        final_shuffle_seed = getattr(self.config, 'final_shuffle_seed', None)
        if final_shuffle_seed is None:
            final_shuffle_seed = shuffle_seed + len(datasets)
        
        merged_dataset = self.merge_datasets(sampled_datasets, final_shuffle_seed)
        
        # Save the merged dataset
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(self.config.get("verbose_level", VerboseLevel.INFO))
        )
        
        output_path = Path(self.config.output_path)
        dataset_handler.save_to_disk(merged_dataset, str(output_path))
        
        # Log summary
        if isinstance(merged_dataset, DatasetDict):
            total_examples = sum(len(split) for split in merged_dataset.values())
            split_info = {split: len(data) for split, data in merged_dataset.items()}
            self.logger.info(f"Merged dataset saved to {output_path}")
            self.logger.info(f"Total examples: {total_examples:,}")
            self.logger.info(f"Splits: {split_info}")
        else:
            self.logger.info(f"Merged dataset saved to {output_path}")
            self.logger.info(f"Total examples: {len(merged_dataset):,}")
        
        self.logger.info("Dataset merge workflow completed successfully")
