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
    Abstract base class for dataset orchestration.

    Provides a scalable interface for validating dataset configurations, loading datasets from various sources,
    and splitting datasets for robust machine learning workflows. Integrates logging for debugging and tracing.

    This class is designed for extensibility; subclasses should implement additional orchestration logic as needed.
    """

    def __init__(self, config: Box) -> None:
        """
        Initialize the BaseOrchestrator.

        :param config: Configuration object with keys such as 'verbose_level', 'dataset', and optional 'test_size'.
        :type config: Box
        """
        
        self.config = config
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )
        self.logger = get_logger(__name__, self.verbose_level)
        self.storage = DatasetStorage(self.verbose_level)

    def _validate_dataset_config(self) -> None:
        """
        Validate the presence of dataset configuration.

        :raises ValueError: If the dataset configuration is missing.

        Notes:
            This method is intended to be used internally by the load_dataset method.
        """

        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")

    def load_dataset(self) -> HFDataset:
        """
        Load and optionally split the dataset according to the configuration.

        Loads a dataset based on the provided configuration and handles optional dataset splitting.

        :returns: The processed HuggingFace dataset.
        :rtype: HFDataset

        :raises ValueError: If the dataset configuration is missing or invalid.
        :raises NotImplementedError: If the dataset source is 'huggingface'.

        Notes:
            This method is designed to be extensible; subclasses can override or extend it to support custom dataset sources or formats.
        """

        self.validate_config()
        
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
                # TODO: make it work for single files too
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
                # TODO: make it work for single files too
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            raise ValueError(f"Invalid dataset format: {self.config.dataset.format}")
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
