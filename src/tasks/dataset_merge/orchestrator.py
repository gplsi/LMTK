"""
Clean and Simple Dataset Merge Orchestrator.

Handles merging multiple datasets with specified percentages using a clean,
modular architecture with proper separation of concerns.
"""

from pathlib import Path
from typing import List, Union
from box import Box
from datasets import Dataset, DatasetDict

from src.utils.logging import get_logger, VerboseLevel
from src.utils.dataset import DatasetStorage
from src.utils.orchestrator import BaseOrchestrator
from src.utils import inherit_init_params

from src.tasks.dataset_merge.compatibility import DatasetCompatibilityChecker
from src.tasks.dataset_merge.sampling import DatasetSampler
from src.tasks.dataset_merge.merging import DatasetMerger


@inherit_init_params
class DatasetMergeOrchestrator(BaseOrchestrator):
    """
    Clean, simple orchestrator for dataset merging.
    
    Uses modular components for compatibility checking, sampling, and merging.
    Supports both single datasets and DatasetDicts with different splits.
    """
    
    def __init__(self, config: Box) -> None:
        """Initialize the orchestrator with configuration."""
        super().__init__(config)
        self.logger = get_logger(__name__, level=VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        ))
        
        # Initialize modular components
        self.compatibility_checker = DatasetCompatibilityChecker(self.logger)
        self.sampler = DatasetSampler(self.logger)
        self.merger = DatasetMerger(self.logger)
        
    def validate_config(self) -> None:
        """Validate the configuration for dataset merging."""
        required_fields = ['datasets', 'output_path']
        
        for field in required_fields:
            if not hasattr(self.config, field) or not getattr(self.config, field):
                raise ValueError(f"Configuration field '{field}' is required")
        
        # Validate dataset configurations
        for i, dataset_config in enumerate(self.config.datasets):
            if not hasattr(dataset_config, 'path') or not dataset_config.path:
                raise ValueError(f"Dataset {i}: 'path' is required")
            if not hasattr(dataset_config, 'percentage') or dataset_config.percentage <= 0:
                raise ValueError(f"Dataset {i}: 'percentage' must be > 0")
        
        self.logger.info("✓ Configuration validation passed")

    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        """This method is not used in dataset merge operations."""
        raise NotImplementedError("Use execute() method for dataset merging")
    
    def execute(self) -> None:
        """Execute the clean dataset merge workflow."""
        try:
            self.logger.info("🚀 Starting clean dataset merge workflow")
            
            # Step 1: Validate configuration
            self.validate_config()
            
            # Step 2: Load all datasets
            self.logger.info("📂 Loading datasets...")
            datasets = []
            dataset_handler = DatasetStorage(
                verbose_level=VerboseLevel(self.config.get("verbose_level", VerboseLevel.INFO))
            )
            
            for i, dataset_config in enumerate(self.config.datasets):
                path = dataset_config.path
                self.logger.info(f"Loading dataset {i+1}: {path}")
                dataset = dataset_handler.load_from_disk(path)
                datasets.append(dataset)
            
            # Step 3: Check compatibility
            self.logger.info("🔍 Checking dataset compatibility...")
            compatibility_info = self.compatibility_checker.check_compatibility(datasets)
            
            # Step 4: Sample datasets according to percentages
            self.logger.info("📊 Sampling datasets...")
            sampled_datasets = []
            shuffle_seed = getattr(self.config, 'shuffle_seed', 42)
            
            total_input = sum(compatibility_info.get('total_examples', 0) for _ in datasets)
            total_output = 0
            
            for i, (dataset, dataset_config) in enumerate(zip(datasets, self.config.datasets)):
                percentage = dataset_config.percentage
                path = dataset_config.path
                
                sampled = self.sampler.sample_dataset(dataset, percentage, shuffle_seed + i)
                sampled_datasets.append(sampled)
                
                sampled_size = sampled.num_rows if hasattr(sampled, 'num_rows') else sum(len(split) for split in sampled.values())
                total_output += sampled_size
                
                self.logger.info(f"✓ {Path(path).name}: {sampled_size:,} examples ({percentage*100:.1f}%)")
            
            # Step 5: Merge sampled datasets
            self.logger.info("🔗 Merging datasets...")
            final_shuffle_seed = getattr(self.config, 'final_shuffle_seed', shuffle_seed + 999)
            merged_dataset = self.merger.merge_datasets(sampled_datasets, final_shuffle_seed)
            
            # Step 6: Save merged dataset
            output_path = Path(self.config.output_path)
            self.logger.info(f"💾 Saving merged dataset to {output_path}")
            dataset_handler.save_to_disk(merged_dataset, str(output_path))
            
            # Step 7: Final summary
            final_size = merged_dataset.num_rows if hasattr(merged_dataset, 'num_rows') else sum(len(split) for split in merged_dataset.values())
            
            if isinstance(merged_dataset, DatasetDict):
                split_info = {split: len(data) for split, data in merged_dataset.items()}
                self.logger.info(f"✅ Merge complete! DatasetDict with {final_size:,} total examples")
                self.logger.info(f"Splits: {split_info}")
            else:
                self.logger.info(f"✅ Merge complete! Dataset with {final_size:,} examples")
            
            efficiency = (final_size / total_input * 100) if total_input > 0 else 0
            self.logger.info(f"Efficiency: {total_input:,} → {final_size:,} examples ({efficiency:.1f}%)")
            self.logger.info("🎉 Dataset merge completed successfully!")
            
        except Exception as e:
            self.logger.error(f"💥 Dataset merge FAILED: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
