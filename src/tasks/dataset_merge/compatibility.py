"""
Dataset compatibility validation utilities.
"""

from typing import Dict, Iterable, List, Optional, Union
from datasets import Dataset, DatasetDict, Features, Sequence
import logging


class DatasetCompatibilityChecker:
    """Handles all dataset compatibility validation logic."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_compatibility(
        self,
        datasets: List[Union[Dataset, DatasetDict]],
        strict: bool = True,
        required_features: Optional[Iterable[str]] = None,
        feature_length_constraints: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """
        Check if datasets are compatible for merging.
        
        Args:
            datasets: List of datasets to check
            strict: Require an exact feature match across datasets
            required_features: Features that must exist in every dataset
            feature_length_constraints: Mapping feature name → expected sequence length
            
        Returns:
            Dict with compatibility info about the reference dataset
            
        Raises:
            ValueError: If datasets are incompatible or violate config expectations
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        required_features = list(required_features or [])
        feature_length_constraints = dict(feature_length_constraints or {})
        
        # Use first dataset as reference
        reference = datasets[0]
        reference_info = self._extract_dataset_info(reference)
        
        self.logger.info(f"Reference dataset: {reference_info['type']}")
        self.logger.info(f"Reference splits: {reference_info['splits']}")
        self.logger.info(f"Reference features: {list(reference_info['features'].keys())}")
        
        # Validate reference against config expectations
        self._validate_config_expectations(
            reference_info['features'],
            required_features,
            feature_length_constraints,
            dataset_label="reference dataset"
        )
        
        # Validate all other datasets against reference and config
        for i, dataset in enumerate(datasets[1:], 1):
            current_info = self._extract_dataset_info(dataset)
            self._validate_against_reference(current_info, reference_info, i, strict)
            self._validate_config_expectations(
                current_info['features'],
                required_features,
                feature_length_constraints,
                dataset_label=f"dataset {i}"
            )
        
        self.logger.info(f"✓ All {len(datasets)} datasets are compatible with config expectations")
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
    
    def _validate_against_reference(
        self,
        current_info: Dict,
        reference_info: Dict,
        dataset_idx: int,
        strict: bool,
    ) -> None:
        """Validate current dataset against reference dataset."""
        
        # Check dataset type compatibility
        if current_info['type'] != reference_info['type']:
            raise ValueError(
                f"Dataset {dataset_idx} type mismatch: "
                f"expected {reference_info['type']}, got {current_info['type']}"
            )
        
        # Check feature compatibility
        if strict:
            if current_info['features'] != reference_info['features']:
                self._raise_feature_mismatch(current_info, reference_info, dataset_idx)
        else:
            ref_keys = set(reference_info['features'].keys())
            curr_keys = set(current_info['features'].keys())
            missing_keys = ref_keys - curr_keys
            if missing_keys:
                raise ValueError(
                    f"Dataset {dataset_idx} missing required feature keys: {sorted(missing_keys)}"
                )
            for key in ref_keys:
                if reference_info['features'][key] != current_info['features'][key]:
                    raise ValueError(
                        f"Dataset {dataset_idx} feature '{key}' type mismatch under non-strict mode: "
                        f"expected {reference_info['features'][key]}, "
                        f"got {current_info['features'][key]}"
                    )
        
        # For DatasetDict, we allow different splits - they'll be merged appropriately
        if current_info['type'] == 'DatasetDict':
            self.logger.info(f"Dataset {dataset_idx} splits: {current_info['splits']}")
        
        self.logger.info(f"✓ Dataset {dataset_idx} compatible ({current_info['total_examples']:,} examples)")

    def _validate_config_expectations(
        self,
        features: Features,
        required_features: List[str],
        feature_length_constraints: Dict[str, int],
        dataset_label: str,
    ) -> None:
        """Validate dataset features against config-driven expectations."""
        if not required_features and not feature_length_constraints:
            return

        feature_keys = set(features.keys())

        if required_features:
            missing = [name for name in required_features if name not in feature_keys]
            if missing:
                raise ValueError(
                    f"{dataset_label} missing required features from config: {missing}"
                )

        for feature_name, expected_length in feature_length_constraints.items():
            if feature_name not in feature_keys:
                self.logger.warning(
                    f"{dataset_label} does not provide feature '{feature_name}' required for length check"
                )
                continue

            feature = features[feature_name]
            actual_length = self._get_sequence_length(feature)

            if actual_length is None:
                self.logger.warning(
                    f"{dataset_label} feature '{feature_name}' has no fixed length; "
                    f"skipping expected length check ({expected_length})"
                )
                continue

            if actual_length != expected_length:
                raise ValueError(
                    f"{dataset_label} feature '{feature_name}' length mismatch: "
                    f"expected {expected_length}, got {actual_length}"
                )

    def _raise_feature_mismatch(self, current_info: Dict, reference_info: Dict, dataset_idx: int) -> None:
        """Raise a detailed error for feature mismatches."""
        ref_keys = set(reference_info['features'].keys())
        curr_keys = set(current_info['features'].keys())
        
        if ref_keys != curr_keys:
            raise ValueError(
                f"Dataset {dataset_idx} feature keys mismatch: "
                f"expected {sorted(ref_keys)}, got {sorted(curr_keys)}"
            )
        
        for key in ref_keys:
            if reference_info['features'][key] != current_info['features'][key]:
                raise ValueError(
                    f"Dataset {dataset_idx} feature '{key}' type mismatch: "
                    f"expected {reference_info['features'][key]}, "
                    f"got {current_info['features'][key]}"
                )

    @staticmethod
    def _get_sequence_length(feature) -> Optional[int]:
        """Return the declared sequence length for a feature, if any."""
        if isinstance(feature, Sequence):
            return feature.length
        return None
