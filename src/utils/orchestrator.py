"""
Module: orchestrator.py

This module defines the BaseOrchestrator class, which provides a framework for loading datasets
based on a given configuration. The orchestrator handles logging, dataset validation, loading,
and optional splitting. It supports local datasets (loaded from disk or files) and sets the stage
for future integration with HuggingFace datasets.
"""

from box import Box
from src.utils.logging import get_logger
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from abc import ABC
from datasets import Dataset as HFDataset
from typing import Optional


class BaseOrchestrator(ABC):
    """
    BaseOrchestrator is an abstract base class that defines a common interface for dataset orchestration.

    It provides methods for validating dataset configurations, loading datasets from various sources,
    and splitting datasets based on test size configurations. Logging is integrated for debugging and
    tracing purposes.
    """

    def __init__(self, config: Box) -> None:
        """
        Initialize the BaseOrchestrator instance with configuration details.

        The initializer sets up the configuration, logging mechanism, and a dataset storage utility
        based on the provided parameters. The configuration should specify at least a 'dataset' key,
        which further provides details like source, format, and file configurations.

        Parameters:
            config (Box): A configuration object containing parameters such as:
                          - verbose_level (optional): The verbosity level for logging.
                          - dataset: A dictionary containing dataset-specific configurations:
                                     * source: The dataset source ('local' or 'huggingface').
                                     * format: The format of the dataset (e.g., 'dataset' or 'files').
                                     * nameOrPath: For local datasets, a path or name to locate the dataset.
                                     * use_txt_as_samples (optional): A flag to enable text samples.
                                     * file_config: Additional file configurations when format is 'files'.
                          - test_size (optional): A float indicating the ratio to split the dataset for testing.
            fabric_rank (Optional[int]): The process rank in distributed training. Used to filter logs.
        """
        
        self.config = config
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )
        self.logger = get_logger(__name__, self.verbose_level)
        self.storage = DatasetStorage(self.verbose_level)

    def _validate__dataset_config(self) -> None:
        """
        Validate that the configuration contains the necessary dataset settings.

        This method checks whether the 'dataset' configuration is provided. If not, it raises a
        ValueError indicating that dataset configuration is mandatory.

        Raises:
            ValueError: If the dataset configuration is missing.
        """
        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")

    def load_dataset(self) -> HFDataset:
        """
        Load a dataset based on the provided configuration and handle optional dataset splitting.

        The process includes:
          1. Validating the presence of dataset configuration.
          2. Initializing a dataset handler with the appropriate verbose level and text sample settings.
          3. Loading the dataset based on the 'source' and 'format' specified in the configuration:
             - For a local source with 'dataset' format, the dataset is loaded from disk.
             - For a local source with 'files' format, the dataset is constructed by processing files in a directory.
          4. If a test size is defined, the dataset is split accordingly.
          5. Returning the loaded (and possibly split) dataset.

        Returns:
            HFDataset: The processed HuggingFace dataset.

        Raises:
            ValueError: If the dataset configuration is missing, or if an invalid dataset source or format is specified.
            NotImplementedError: If the dataset source is 'huggingface', as this functionality is not implemented.
        """
        
        self._validate__dataset_config()
        
        # Safely get use_txt_as_samples with a default value if not present
        use_txt_as_samples = False
        if hasattr(self.config.dataset, 'use_txt_as_samples'):
            use_txt_as_samples = self.config.dataset.use_txt_as_samples
        
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            ),
            enable_txt_samples=use_txt_as_samples
        )

        if self.config.dataset.source == "local":
            if self.config.dataset.format == "dataset":
                self.logger.info(f"Loading dataset from path '{self.config.dataset.nameOrPath}'")
                dataset = dataset_handler.load_from_disk(self.config.dataset.nameOrPath)
                
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            
            elif self.config.dataset.format == "files":
                self.logger.info(f"Loading dataset from files at dir '{self.config.dataset.nameOrPath}'")
                
                # Get file_config from config if available
                file_config = None
                if hasattr(self.config.dataset, 'file_config'):
                    file_config = self.config.dataset.file_config
                
                # Process files with complete file_config
                dataset = dataset_handler.process_files(
                    self.config.dataset.nameOrPath,
                    file_config=file_config,
                )
                
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            raise ValueError(f"Invalid dataset format: {self.config.dataset.format}")
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
