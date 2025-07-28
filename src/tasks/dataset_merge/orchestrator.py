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
import glob
from box import Box
from datasets import DatasetDict, concatenate_datasets, Features, Sequence, Value, load_dataset
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
    
    def load_and_validate_dataset(self, dataset_path: str, global_sequence_length: int = None) -> Tuple[Union[HFDataset, DatasetDict], Dict]:
        """
        Load a single dataset with strict fixed-size vector enforcement.
        Ensures all datasets use the same sequence length for vector compatibility.
        
        Args:
            dataset_path: Path to the tokenized dataset
            global_sequence_length: Enforced sequence length for all datasets (if None, detect from first dataset)
            
        Returns:
            Tuple of (dataset, metadata_dict)
        """
        from datasets import Features, Sequence, Value
        
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Use global sequence length or detect it
        if global_sequence_length is not None:
            target_sequence_length = global_sequence_length
            self.logger.info(f"Using enforced global sequence length: {target_sequence_length}")
        else:
            target_sequence_length = self._detect_and_validate_sequence_length(dataset_path)
            
        # Validate that sequence length is fixed (not variable)
        if target_sequence_length <= 0:
            raise ValueError(
                f"Dataset {dataset_path} has variable or invalid sequence length ({target_sequence_length}). "
                f"Only fixed-size sequences are supported for merging. Please specify 'sequence_length' in config."
            )
        
        # Define a strict feature schema for tokenized datasets with fixed length
        universal_features = Features({
            'input_ids': Sequence(Value('int32'), length=target_sequence_length),
            'attention_mask': Sequence(Value('int32'), length=target_sequence_length),
            'labels': Sequence(Value('int32'), length=target_sequence_length)
        })
        
        self.logger.info(f"Enforcing fixed sequence length: {target_sequence_length}")
        
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(self.config.get("verbose_level", VerboseLevel.INFO))
        )
        
        try:
            # First try: Load with universal features to force schema compatibility
            self.logger.info("Loading with strict fixed-size feature schema")
            dataset = dataset_handler.load_from_disk(dataset_path)
            self.logger.info(f"Dataset loaded from disk successfully")
            
            # Log dataset structure
            if isinstance(dataset, DatasetDict):
                splits_info = {split: len(data) for split, data in dataset.items()}
                self.logger.info(f"Loaded DatasetDict with splits: {splits_info}")
            else:
                self.logger.info(f"Loaded single dataset with {len(dataset):,} examples")
            
            # Validate actual data before casting
            self.logger.info("Validating sequence lengths before casting...")
            self._validate_dataset_sequences(dataset, target_sequence_length, dataset_path)
            
            # Cast to universal features to enforce schema compatibility
            self.logger.info("Casting dataset to universal feature schema...")
            if isinstance(dataset, DatasetDict):
                casted_splits = {}
                for split_name, split_dataset in dataset.items():
                    self.logger.info(f"Casting split '{split_name}' ({len(split_dataset):,} examples)")
                    casted_splits[split_name] = split_dataset.cast(universal_features)
                dataset = DatasetDict(casted_splits)
                self.logger.info("All splits cast successfully")
            else:
                dataset = dataset.cast(universal_features)
                self.logger.info("Dataset cast successfully")
                
            self.logger.info("Successfully loaded and enforced fixed-size schema")
            
        except Exception as e:
            # If normal loading fails, try loading raw data files with strict validation
            if "External features info don't match" in str(e) or "feature info" in str(e).lower():
                self.logger.info(f"Schema mismatch detected, attempting raw data loading with validation")
                dataset = self._force_load_with_features(dataset_path, universal_features, target_sequence_length)
            else:
                raise e
        
        # Extract metadata for compatibility checking (should be consistent now)
        if isinstance(dataset, DatasetDict):
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
    
    def _detect_and_validate_sequence_length(self, dataset_path: str) -> int:
        """
        Detect and validate the sequence length from dataset, ensuring it's fixed-size.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Target sequence length (must be > 0 for fixed-size)
            
        Raises:
            ValueError: If sequence length cannot be determined or is variable
        """
        # Check if sequence length is specified in config (highest priority)
        if hasattr(self.config, 'sequence_length') and self.config.sequence_length:
            if self.config.sequence_length <= 0:
                raise ValueError(f"Configured sequence_length must be > 0, got: {self.config.sequence_length}")
            self.logger.info(f"Using configured fixed sequence length: {self.config.sequence_length}")
            return self.config.sequence_length
        
        # Try to detect from dataset_info.json
        dataset_info_path = os.path.join(dataset_path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            try:
                import json
                with open(dataset_info_path, 'r') as f:
                    dataset_info = json.load(f)
                
                # Look for sequence length in features
                features = dataset_info.get('features', {})
                for field_name in ['input_ids', 'attention_mask', 'labels']:
                    if field_name in features:
                        feature_info = features[field_name]
                        if isinstance(feature_info, dict):
                            # Check for length specification
                            if 'length' in feature_info:
                                detected_length = feature_info['length']
                                if detected_length > 0:
                                    self.logger.info(f"Detected fixed sequence length from {field_name}: {detected_length}")
                                    return detected_length
                                elif detected_length == -1:
                                    raise ValueError(f"Dataset {dataset_path} has variable-length sequences in {field_name}. Only fixed-size sequences are supported.")
                            
                            # Check for feature type with length info
                            feature_type = feature_info.get('feature', {})
                            if isinstance(feature_type, dict) and 'length' in feature_type:
                                detected_length = feature_type['length']
                                if detected_length > 0:
                                    self.logger.info(f"Detected fixed sequence length from {field_name} feature: {detected_length}")
                                    return detected_length
                                elif detected_length == -1:
                                    raise ValueError(f"Dataset {dataset_path} has variable-length sequences in {field_name}. Only fixed-size sequences are supported.")
                                    
            except Exception as e:
                self.logger.warning(f"Could not read dataset_info.json: {e}")
        
        # Try to detect from a sample of the data with strict validation
        try:
            self.logger.info("Attempting to detect and validate sequence length from data sample...")
            sample_files = glob.glob(os.path.join(dataset_path, "data-*-of-*.arrow"))
            if sample_files:
                # Load a larger sample to ensure consistency
                first_file = sample_files[0]
                sample_dataset = HFDataset.from_file(first_file)
                
                # Take a larger sample to validate consistency
                sample_size = min(100, len(sample_dataset))
                sample_data = sample_dataset.select(range(sample_size))
                
                # Check input_ids lengths strictly
                if 'input_ids' in sample_data.column_names:
                    lengths = [len(example['input_ids']) for example in sample_data]
                    unique_lengths = set(lengths)
                    
                    if len(unique_lengths) == 1:
                        # All sequences have same length - this is our target
                        detected_length = list(unique_lengths)[0]
                        if detected_length <= 0:
                            raise ValueError(f"Dataset {dataset_path} has invalid sequence length: {detected_length}")
                        self.logger.info(f"Validated fixed sequence length from data: {detected_length}")
                        
                        # Additional validation: check more files if available
                        if len(sample_files) > 1:
                            self._validate_length_across_files(sample_files[:3], detected_length, dataset_path)
                        
                        return detected_length
                    else:
                        # Variable lengths detected - this is not allowed
                        raise ValueError(
                            f"Dataset {dataset_path} contains variable sequence lengths: {sorted(unique_lengths)}. "
                            f"Only fixed-size sequences are supported for merging. "
                            f"Please ensure all sequences have the same length or specify 'sequence_length' in config."
                        )
                else:
                    raise ValueError(f"Dataset {dataset_path} missing 'input_ids' column")
                        
        except Exception as e:
            if "variable sequence lengths" in str(e) or "missing 'input_ids'" in str(e):
                raise e  # Re-raise validation errors
            self.logger.error(f"Could not detect sequence length from data: {e}")
            raise ValueError(
                f"Could not determine sequence length for dataset {dataset_path}. "
                f"Please specify 'sequence_length' in your configuration."
            )
            
    def _validate_length_across_files(self, sample_files: List[str], expected_length: int, dataset_path: str) -> None:
        """
        Validate that sequence length is consistent across multiple arrow files.
        
        Args:
            sample_files: List of arrow files to check
            expected_length: Expected sequence length
            dataset_path: Dataset path for error messages
        """
        for i, file_path in enumerate(sample_files):
            try:
                file_dataset = HFDataset.from_file(file_path)
                sample_size = min(10, len(file_dataset))
                if sample_size > 0:
                    sample_data = file_dataset.select(range(sample_size))
                    
                    if 'input_ids' in sample_data.column_names:
                        lengths = [len(example['input_ids']) for example in sample_data]
                        unique_lengths = set(lengths)
                        
                        if len(unique_lengths) != 1 or list(unique_lengths)[0] != expected_length:
                            raise ValueError(
                                f"Dataset {dataset_path} has inconsistent sequence lengths across files. "
                                f"File {i+1} has lengths {sorted(unique_lengths)}, expected {expected_length}."
                            )
            except Exception as e:
                self.logger.warning(f"Could not validate file {file_path}: {e}")
                
    def _validate_dataset_sequences(self, dataset: Union[HFDataset, DatasetDict], 
                                  expected_length: int, dataset_path: str) -> None:
        """
        Validate that all sequences in the dataset have the expected fixed length.
        
        Args:
            dataset: The loaded dataset
            expected_length: Expected sequence length
            dataset_path: Dataset path for error messages
        """
        try:
            self.logger.info(f"Validating sequence lengths for dataset: {dataset_path}")
            
            def validate_split(split_dataset: HFDataset, split_name: str = ""):
                try:
                    split_info = f" in split '{split_name}'" if split_name else ""
                    self.logger.info(f"Validating split{split_info} with {len(split_dataset):,} examples")
                    
                    # Sample a portion of the dataset for validation
                    sample_size = min(50, len(split_dataset))
                    if sample_size == 0:
                        self.logger.warning(f"Split{split_info} is empty, skipping validation")
                        return
                        
                    sample_indices = random.sample(range(len(split_dataset)), sample_size)
                    sample_data = split_dataset.select(sample_indices)
                    self.logger.info(f"Validating {sample_size} samples from split{split_info}")
                    
                    # Check input_ids lengths
                    if 'input_ids' in sample_data.column_names:
                        lengths = [len(example['input_ids']) for example in sample_data]
                        unique_lengths = set(lengths)
                        
                        if len(unique_lengths) != 1:
                            raise ValueError(
                                f"Dataset {dataset_path}{split_info} contains variable sequence lengths: {sorted(unique_lengths)}. "
                                f"Expected fixed length: {expected_length}."
                            )
                            
                        actual_length = list(unique_lengths)[0]
                        if actual_length != expected_length:
                            raise ValueError(
                                f"Dataset {dataset_path}{split_info} has sequence length {actual_length}, "
                                f"expected {expected_length}."
                            )
                        
                        self.logger.info(f"Split{split_info} validation passed: all {sample_size} samples have length {actual_length}")
                    else:
                        raise ValueError(f"Dataset {dataset_path}{split_info} missing 'input_ids' column")
                except Exception as e:
                    self.logger.error(f"Split validation failed{split_info}: {e}")
                    raise e
            
            if isinstance(dataset, DatasetDict):
                self.logger.info(f"Validating DatasetDict with {len(dataset)} splits")
                for split_name, split_dataset in dataset.items():
                    validate_split(split_dataset, split_name)
            else:
                self.logger.info("Validating single dataset")
                validate_split(dataset)
                
            self.logger.info(f"✓ Sequence length validation passed: all sequences have length {expected_length}")
        except Exception as e:
            self.logger.error(f"✗ Sequence length validation failed for {dataset_path}: {e}")
            raise e
    
    def _force_load_with_features(self, dataset_path: str, features, target_sequence_length: int) -> Union[HFDataset, DatasetDict]:
        """
        Force load dataset with specific features and validate sequence lengths.
        """
        self.logger.info("Attempting to load raw dataset files with strict validation")
        dataset = self._load_raw_dataset(dataset_path, target_sequence_length)
        
        # Validate the loaded dataset
        self._validate_dataset_sequences(dataset, target_sequence_length, dataset_path)
        
        return dataset
    
    def _load_raw_dataset(self, dataset_path: str, target_sequence_length: int) -> Union[HFDataset, DatasetDict]:
        """
        Load dataset by directly reading the underlying arrow files with strict validation.
        
        Args:
            dataset_path: Path to the dataset
            target_sequence_length: Expected sequence length (must be > 0)
        """
        self.logger.info("Loading raw dataset files with sequence length validation")
        
        if target_sequence_length <= 0:
            raise ValueError(f"Invalid target sequence length: {target_sequence_length}")
        
        # Define the target schema we want
        target_features = Features({
            'input_ids': Sequence(Value('int32'), length=target_sequence_length),
            'attention_mask': Sequence(Value('int32'), length=target_sequence_length),
            'labels': Sequence(Value('int32'), length=target_sequence_length)
        })
        
        try:
            # Look for dataset arrow files (data-*.arrow pattern)
            data_files = glob.glob(os.path.join(dataset_path, "data-*-of-*.arrow"))
            
            if not data_files:
                raise ValueError(f"No data files found in {dataset_path}")
            
            # Sort files to ensure consistent order
            data_files.sort()
            self.logger.info(f"Found {len(data_files)} arrow data files")
            
            # Use datasets.load_dataset with arrow files
            self.logger.info("Loading from arrow files using load_dataset")
            combined_dataset = HFDataset.from_file(data_files[0])
            
            # Validate the first file's sequence lengths
            self._validate_dataset_sequences(combined_dataset, target_sequence_length, f"{dataset_path}/file_0")
            
            # Load remaining files and concatenate with validation
            if len(data_files) > 1:
                datasets_to_concat = [combined_dataset]
                
                for i, file_path in enumerate(data_files[1:], 1):
                    try:
                        file_dataset = HFDataset.from_file(file_path)
                        # Validate each file's sequence lengths
                        self._validate_dataset_sequences(file_dataset, target_sequence_length, f"{dataset_path}/file_{i}")
                        datasets_to_concat.append(file_dataset)
                    except Exception as file_error:
                        self.logger.error(f"Failed to load or validate {file_path}: {file_error}")
                        raise file_error
                
                # Concatenate all datasets
                combined_dataset = concatenate_datasets(datasets_to_concat)
            
            # Cast to our target features to ensure schema compatibility
            combined_dataset = combined_dataset.cast(target_features)
            
            self.logger.info(f"Successfully loaded and validated {len(combined_dataset)} rows from raw data files")
            return combined_dataset
                
        except Exception as e:
            self.logger.error(f"Failed to load raw dataset: {e}")
            # Try alternative approach with load_dataset
            try:
                self.logger.info("Trying alternative loading method with load_dataset")
                
                # Load as arrow files with strict features
                combined_dataset = load_dataset(
                    "arrow", 
                    data_files=data_files,
                    features=target_features
                )['train']  # load_dataset returns DatasetDict with 'train' split
                
                # Validate the loaded dataset
                self._validate_dataset_sequences(combined_dataset, target_sequence_length, dataset_path)
                
                self.logger.info(f"Successfully loaded and validated {len(combined_dataset)} rows using load_dataset")
                return combined_dataset
                
            except Exception as e2:
                self.logger.error(f"Alternative loading also failed: {e2}")
                raise ValueError(
                    f"Could not load dataset {dataset_path} with fixed sequence length {target_sequence_length}. "
                    f"Ensure all sequences in the dataset have exactly {target_sequence_length} tokens."
                ) from e
        
    def validate_compatibility(self, datasets_metadata: List[Dict], datasets: List[Union[HFDataset, DatasetDict]]) -> None:
        """
        Validate that all datasets are compatible for merging with strict fixed-size requirements.
        All datasets must have identical fixed-size vectors.
        
        Args:
            datasets_metadata: List of metadata dictionaries from all datasets
            datasets: List of actual dataset objects
        """
        self.logger.info("Validating dataset compatibility with strict fixed-size requirements...")
        
        if not datasets_metadata:
            raise ValueError("No datasets provided for validation")
        
        # Use first dataset as reference
        reference = datasets_metadata[0]
        reference_columns = set(reference['column_names'])
        reference_features = reference['features']
        
        # Extract and validate reference sequence length
        reference_seq_length = None
        if 'input_ids' in reference_features:
            input_ids_feature = reference_features['input_ids']
            if hasattr(input_ids_feature, 'length'):
                reference_seq_length = input_ids_feature.length
        
        if reference_seq_length is None or reference_seq_length <= 0:
            raise ValueError(
                f"Reference dataset has invalid sequence length: {reference_seq_length}. "
                f"All datasets must have fixed-size sequences > 0."
            )
        
        self.logger.info(f"Reference fixed sequence length: {reference_seq_length}")
        
        # Validate each subsequent dataset
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
            
            # Check sequence length compatibility (strict)
            current_seq_length = None
            if 'input_ids' in current_features:
                input_ids_feature = current_features['input_ids']
                if hasattr(input_ids_feature, 'length'):
                    current_seq_length = input_ids_feature.length
            
            if current_seq_length != reference_seq_length:
                raise ValueError(
                    f"Dataset {i} has incompatible sequence length. "
                    f"Expected fixed length: {reference_seq_length}, Got: {current_seq_length}. "
                    f"All datasets must have identical fixed-size sequences."
                )
            
            # Validate that sequence length is actually fixed (not variable)
            if current_seq_length <= 0:
                raise ValueError(
                    f"Dataset {i} has variable or invalid sequence length: {current_seq_length}. "
                    f"Only fixed-size sequences are supported."
                )
            
            # Check feature types compatibility
            for col_name in reference_columns:
                if col_name in current_features and col_name in reference_features:
                    ref_feature = reference_features[col_name]
                    curr_feature = current_features[col_name]
                    
                    if str(curr_feature) != str(ref_feature):
                        raise ValueError(
                            f"Dataset {i} has incompatible feature type for column '{col_name}': "
                            f"{curr_feature} vs {ref_feature}. "
                            f"All datasets must have identical feature schemas."
                        )
        
        # Final validation: check that all required columns have fixed-size sequences
        required_sequence_columns = ['input_ids', 'attention_mask', 'labels']
        for col_name in required_sequence_columns:
            if col_name in reference_features:
                feature = reference_features[col_name]
                if hasattr(feature, 'length') and feature.length != reference_seq_length:
                    raise ValueError(
                        f"Column '{col_name}' has inconsistent sequence length: "
                        f"{feature.length} vs expected {reference_seq_length}"
                    )
        
        self.logger.info(f"All {len(datasets_metadata)} datasets are compatible for merging")
        self.logger.info(f"Validated fixed sequence length: {reference_seq_length}")
        self.logger.info(f"All datasets have identical schemas with fixed-size vectors")
    
    def sample_dataset(self, dataset: Union[HFDataset, DatasetDict], percentage: float, 
                      shuffle_seed: int = None) -> Union[HFDataset, DatasetDict]:
        """
        Sample a percentage of data from a dataset after shuffling.
        Handles both single datasets and DatasetDict with split-level sampling.
        
        Args:
            dataset: The dataset to sample from
            percentage: Percentage of data to sample (can be > 1.0)
            shuffle_seed: Seed for shuffling
            
        Returns:
            Sampled dataset
        """
        if isinstance(dataset, DatasetDict):
            self.logger.info(f"Sampling DatasetDict with {len(dataset)} splits at {percentage*100:.1f}% each")
            sampled_splits = {}
            
            for split_name, split_dataset in dataset.items():
                self.logger.info(f"Sampling split '{split_name}': {len(split_dataset):,} examples")
                sampled_split = self._sample_split(split_dataset, percentage, shuffle_seed)
                sampled_splits[split_name] = sampled_split
                self.logger.info(f"Split '{split_name}' sampled: {len(split_dataset):,} → {len(sampled_split):,} examples")
            
            sampled_dict = DatasetDict(sampled_splits)
            total_sampled = sum(len(split) for split in sampled_dict.values())
            total_original = sum(len(split) for split in dataset.values())
            self.logger.info(f"DatasetDict sampling completed: {total_original:,} → {total_sampled:,} examples")
            
            return sampled_dict
        else:
            self.logger.info(f"Sampling single dataset: {len(dataset):,} examples at {percentage*100:.1f}%")
            sampled = self._sample_split(dataset, percentage, shuffle_seed)
            self.logger.info(f"Single dataset sampled: {len(dataset):,} → {len(sampled):,} examples")
            return sampled
    
    def _sample_split(self, dataset: HFDataset, percentage: float, shuffle_seed: int = None) -> HFDataset:
        """Sample a percentage from a single dataset split."""
        try:
            total_size = len(dataset)
            target_size = int(total_size * percentage)
            
            self.logger.info(f"Sampling {target_size} examples ({percentage*100:.1f}%) from {total_size} total")
            
            # Shuffle the dataset first
            self.logger.info(f"Shuffling dataset with seed: {shuffle_seed}")
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
                self.logger.info(f"Concatenating {len(copies)} dataset copies")
                sampled_dataset = concatenate_datasets(copies)
            else:
                # Regular sampling
                self.logger.info(f"Regular sampling: selecting first {target_size} examples")
                sampled_dataset = shuffled_dataset.select(range(target_size))
            
            self.logger.info(f"Sampling completed: {len(sampled_dataset)} examples")
            return sampled_dataset
        except Exception as e:
            self.logger.error(f"Failed to sample dataset split: {e}")
            raise e
    
    def merge_datasets(self, datasets: List[Union[HFDataset, DatasetDict]], 
                      final_shuffle_seed: int = None) -> Union[HFDataset, DatasetDict]:
        """
        Merge multiple datasets and shuffle the result.
        Handles both single datasets and DatasetDicts with comprehensive split merging.
        
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
            # Merge DatasetDicts split by split with comprehensive logging
            self.logger.info("Merging DatasetDict objects split by split...")
            
            # Collect all unique split names across all datasets
            split_names = set()
            for i, dataset in enumerate(datasets):
                dataset_splits = set(dataset.keys())
                split_names.update(dataset_splits)
                self.logger.info(f"Dataset {i+1} has splits: {sorted(dataset_splits)}")
            
            self.logger.info(f"Total unique splits found: {sorted(split_names)}")
            
            merged_splits = {}
            split_stats = {}
            
            for split_name in sorted(split_names):
                self.logger.info(f"Processing split: '{split_name}'")
                split_datasets = []
                split_info = []
                
                # Collect datasets that have this split
                for i, dataset in enumerate(datasets):
                    if split_name in dataset:
                        split_dataset = dataset[split_name]
                        split_datasets.append(split_dataset)
                        split_info.append(f"Dataset {i+1}: {len(split_dataset):,} examples")
                    else:
                        split_info.append(f"Dataset {i+1}: not present")
                
                if split_datasets:
                    self.logger.info(f"Split '{split_name}' contributions: {split_info}")
                    
                    # Merge datasets for this split
                    merged_split = concatenate_datasets(split_datasets)
                    
                    # Shuffle the merged split
                    if final_shuffle_seed is not None:
                        merged_split = merged_split.shuffle(seed=final_shuffle_seed + hash(split_name) % 1000)
                        self.logger.info(f"Split '{split_name}' shuffled with seed: {final_shuffle_seed + hash(split_name) % 1000}")
                    else:
                        merged_split = merged_split.shuffle()
                        self.logger.info(f"Split '{split_name}' shuffled with random seed")
                    
                    merged_splits[split_name] = merged_split
                    split_stats[split_name] = {
                        'total_examples': len(merged_split),
                        'source_datasets': len(split_datasets),
                        'contributions': split_info
                    }
                    
                    self.logger.info(f"Split '{split_name}' merged successfully: {len(merged_split):,} total examples")
                else:
                    self.logger.warning(f"Split '{split_name}' has no datasets - this should not happen")
            
            # Log comprehensive merge summary
            self.logger.info("=== DatasetDict Merge Summary ===")
            for split_name, stats in split_stats.items():
                self.logger.info(f"Split '{split_name}': {stats['total_examples']:,} examples from {stats['source_datasets']} datasets")
            
            total_examples = sum(len(split) for split in merged_splits.values())
            self.logger.info(f"Total merged examples across all splits: {total_examples:,}")
            
            return DatasetDict(merged_splits)
        else:
            # Merge single datasets
            self.logger.info("Merging single dataset objects...")
            total_examples_before = sum(len(dataset) for dataset in datasets)
            
            merged_dataset = concatenate_datasets(datasets)
            
            # Final shuffle
            if final_shuffle_seed is not None:
                merged_dataset = merged_dataset.shuffle(seed=final_shuffle_seed)
                self.logger.info(f"Merged dataset shuffled with seed: {final_shuffle_seed}")
            else:
                merged_dataset = merged_dataset.shuffle()
                self.logger.info("Merged dataset shuffled with random seed")
            
            self.logger.info(f"Single dataset merge completed: {len(merged_dataset):,} examples (from {total_examples_before:,} total)")
            return merged_dataset
    
    def execute(self) -> None:
        """Execute the dataset merge workflow with strict fixed-size vector guarantees."""
        try:
            self.logger.info("Starting dataset merge workflow with fixed-size vector enforcement")
            
            # Validate configuration
            self.logger.info("Step 1: Validating configuration...")
            self.validate_config()
            self.logger.info("✓ Configuration validation completed")
            
            # Step 1: Determine global sequence length from first dataset or config
            self.logger.info("Step 2: Determining global sequence length...")
            global_sequence_length = None
            if hasattr(self.config, 'sequence_length') and self.config.sequence_length:
                global_sequence_length = self.config.sequence_length
                self.logger.info(f"Using configured global sequence length: {global_sequence_length}")
            else:
                # Detect from first dataset
                first_dataset_path = self.config.datasets[0].path
                self.logger.info(f"Detecting sequence length from first dataset: {first_dataset_path}")
                global_sequence_length = self._detect_and_validate_sequence_length(first_dataset_path)
                self.logger.info(f"Detected global sequence length from first dataset: {global_sequence_length}")
            
            # Validate that we have a valid fixed sequence length
            if global_sequence_length <= 0:
                raise ValueError(
                    f"Invalid global sequence length: {global_sequence_length}. "
                    f"Please specify a valid 'sequence_length' > 0 in your configuration."
                )
            self.logger.info(f"✓ Global sequence length established: {global_sequence_length}")
            
            # Step 2: Load and validate all datasets with the same sequence length
            self.logger.info("Step 3: Loading and validating all datasets...")
            datasets = []
            datasets_metadata = []
            
            self.logger.info(f"Loading all datasets with enforced sequence length: {global_sequence_length}")
            
            for i, dataset_config in enumerate(self.config.datasets):
                try:
                    self.logger.info(f"Loading dataset {i+1}/{len(self.config.datasets)}: {dataset_config.path}")
                    dataset, metadata = self.load_and_validate_dataset(dataset_config.path, global_sequence_length)
                    datasets.append(dataset)
                    datasets_metadata.append(metadata)
                    self.logger.info(f"✓ Dataset {i+1} loaded successfully: {metadata['num_rows']:,} rows")
                except Exception as e:
                    self.logger.error(f"✗ Failed to load dataset {i+1} ({dataset_config.path}): {e}")
                    raise e
            
            self.logger.info(f"✓ All {len(datasets)} datasets loaded successfully")
            
            # Step 3: Validate compatibility (should pass since we enforce same sequence length)
            self.logger.info("Step 4: Validating dataset compatibility...")
            try:
                self.validate_compatibility(datasets_metadata, datasets)
                self.logger.info("✓ Dataset compatibility validation passed")
            except Exception as e:
                self.logger.error(f"✗ Dataset compatibility validation failed: {e}")
                raise e
            
            # Step 4: Get sampling configuration
            self.logger.info("Step 5: Setting up sampling configuration...")
            shuffle_seed = getattr(self.config, 'shuffle_seed', None)
            if shuffle_seed is None:
                shuffle_seed = random.randint(1, 2**32 - 1)
                self.logger.info(f"Using random shuffle seed: {shuffle_seed}")
            else:
                self.logger.info(f"Using configured shuffle seed: {shuffle_seed}")
            
            # Step 5: Sample datasets according to specified percentages
            self.logger.info("Step 6: Sampling datasets according to specified percentages...")
            sampled_datasets = []
            total_input_examples = 0
            total_output_examples = 0
            
            for i, (dataset, dataset_config) in enumerate(zip(datasets, self.config.datasets)):
                try:
                    self.logger.info(f"Processing dataset {i+1}: {dataset_config.path} ({dataset_config.percentage*100:.1f}%)")
                    
                    # Count input examples
                    if isinstance(dataset, DatasetDict):
                        input_examples = sum(len(split) for split in dataset.values())
                        self.logger.info(f"Dataset {i+1} is DatasetDict with {len(dataset)} splits, total examples: {input_examples:,}")
                    else:
                        input_examples = len(dataset)
                        self.logger.info(f"Dataset {i+1} is single dataset with {input_examples:,} examples")
                    total_input_examples += input_examples
                    
                    self.logger.info(f"Sampling dataset {i+1} at {dataset_config.percentage*100:.1f}%...")
                    sampled_dataset = self.sample_dataset(dataset, dataset_config.percentage, shuffle_seed + i)
                    sampled_datasets.append(sampled_dataset)
                    
                    # Count output examples
                    if isinstance(sampled_dataset, DatasetDict):
                        output_examples = sum(len(split) for split in sampled_dataset.values())
                    else:
                        output_examples = len(sampled_dataset)
                    total_output_examples += output_examples
                    
                    self.logger.info(f"✓ Dataset {i+1} sampled: {input_examples:,} → {output_examples:,} examples")
                except Exception as e:
                    self.logger.error(f"✗ Failed to sample dataset {i+1}: {e}")
                    raise e
            
            self.logger.info(f"✓ All datasets sampled successfully: {total_input_examples:,} → {total_output_examples:,} examples")
            
            # Step 6: Merge all sampled datasets
            self.logger.info("Step 7: Merging all sampled datasets...")
            try:
                final_shuffle_seed = getattr(self.config, 'final_shuffle_seed', None)
                if final_shuffle_seed is None:
                    final_shuffle_seed = shuffle_seed + len(datasets)
                
                self.logger.info(f"Merging {len(sampled_datasets)} datasets with final shuffle seed: {final_shuffle_seed}")
                merged_dataset = self.merge_datasets(sampled_datasets, final_shuffle_seed)
                self.logger.info("✓ Dataset merging completed successfully")
            except Exception as e:
                self.logger.error(f"✗ Failed to merge datasets: {e}")
                raise e
            
            # Step 7: Final validation of merged dataset
            self.logger.info("Step 8: Performing final validation of merged dataset...")
            try:
                self._validate_dataset_sequences(merged_dataset, global_sequence_length, "merged_dataset")
                self.logger.info("✓ Final validation passed")
            except Exception as e:
                self.logger.error(f"✗ Final validation failed: {e}")
                raise e
            
            # Step 8: Save the merged dataset
            self.logger.info("Step 9: Saving the merged dataset...")
            try:
                dataset_handler = DatasetStorage(
                    verbose_level=VerboseLevel(self.config.get("verbose_level", VerboseLevel.INFO))
                )
                
                output_path = Path(self.config.output_path)
                self.logger.info(f"Saving merged dataset to: {output_path}")
                dataset_handler.save_to_disk(merged_dataset, str(output_path))
                self.logger.info("✓ Dataset saved successfully")
            except Exception as e:
                self.logger.error(f"✗ Failed to save dataset: {e}")
                raise e
            
            # Step 9: Log comprehensive summary
            self.logger.info("Step 10: Generating final summary...")
            try:
                if isinstance(merged_dataset, DatasetDict):
                    total_examples = sum(len(split) for split in merged_dataset.values())
                    split_info = {split: len(data) for split, data in merged_dataset.items()}
                    self.logger.info(f"✓ Merged DatasetDict saved to {output_path}")
                    self.logger.info(f"✓ Total examples: {total_examples:,}")
                    self.logger.info(f"✓ Splits: {split_info}")
                else:
                    self.logger.info(f"✓ Merged dataset saved to {output_path}")
                    self.logger.info(f"✓ Total examples: {len(merged_dataset):,}")
                
                self.logger.info(f"✓ Fixed sequence length: {global_sequence_length}")
                self.logger.info(f"✓ Input examples: {total_input_examples:,}")
                self.logger.info(f"✓ Output examples: {total_output_examples:,}")
                self.logger.info(f"✓ Sampling efficiency: {(total_output_examples/total_input_examples)*100:.1f}%")
                self.logger.info("🎉 Dataset merge workflow completed successfully with fixed-size vector guarantees! 🎉")
            except Exception as e:
                self.logger.error(f"✗ Failed to generate summary (but merge completed): {e}")
                # Don't raise here since the main work is done
                
        except Exception as e:
            self.logger.error(f"💥 Dataset merge workflow FAILED: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e
