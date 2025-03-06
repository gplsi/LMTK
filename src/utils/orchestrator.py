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
        """
        self.config = config

        # Set the verbose level for logging using the configuration; default to INFO if not provided.
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )

        # Initialize and configure the module-level logger.
        self.logger = get_logger(__name__, self.verbose_level)

        # Instantiate the DatasetStorage utility for dataset operations.
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
        # Ensure that the dataset configuration exists.
        self._validate__dataset_config()
        
        # Initialize the dataset handler. Enable text sample processing if specified in the config.
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            ),
            enable_txt_samples=self.config.dataset.use_txt_as_samples or False
        )

        # Determine the dataset source.
        if self.config.dataset.source == "local":
            # Handle local datasets.
            if self.config.dataset.format == "dataset":
                # Log the start of loading a dataset from disk.
                self.logger.info(f"Loading dataset from path '{self.config.dataset.nameOrPath}'")
                # Load dataset from disk based on the provided path or name.
                dataset = dataset_handler.load_from_disk(self.config.dataset.nameOrPath)
                
                # If a test split size is provided, log the information and split the dataset.
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            
            elif self.config.dataset.format == "files":
                # Log the start of processing dataset files from the specified directory.
                self.logger.info(f"Loading dataset from files at dir '{self.config.dataset.nameOrPath}'")
                # Process files from the specified directory with the defined file extension.
                dataset = dataset_handler.process_files(
                    self.config.dataset.nameOrPath,
                    extension=self.config.dataset.file_config.format,
                )
                # If a test split size is provided, log the information and split the dataset.
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            
            # If the dataset format is not recognized for local sources, raise an error.
            raise ValueError(f"Invalid dataset format: {self.config.dataset.format}")
        
        elif self.config.dataset.source == "huggingface":
            # If attempting to load a HuggingFace dataset, indicate that this feature is not yet implemented.
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        
        # If the dataset source is neither local nor huggingface, raise a configuration error.
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
