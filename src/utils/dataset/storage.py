"""
This module implements the DatasetStorage class, which provides functionality for 
loading, processing, splitting, and saving datasets in Arrow format using the Hugging Face Datasets library.

Key features include:
  - Loading datasets from various file types (e.g., text, CSV, JSON)
  - Grouping files in a directory by file extension
  - Handling special processing for text files when enabled
  - Splitting datasets into training and validation subsets
  - Saving and loading datasets from disk as well as fetching from the Hugging Face Hub

The module makes extensive use of verbose logging to notify users of its operations,
and it is designed with extensibility and robust error handling in mind.
"""

from pathlib import Path
from typing import Dict
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    DatasetDict,
)
from datasets import Dataset as HFDataset
import os
from functools import partial
from enum import IntEnum
from src.utils.logging import VerboseLevel, get_logger
from src.utils.dataset.utils import SUPPORTED_EXTENSIONS, scan_directory


class DatasetStorage:
    """
    Class for handling dataset operations including loading, processing, splitting,
    and saving datasets in Arrow format.

    Attributes:
        verbose_level (VerboseLevel): Level of verbosity for logging.
        enable_txt_samples (bool): Flag to enable loading text files as individual samples.
        logger: Logger instance used to log messages at various verbosity levels.
        extension_to_method (dict): Mapping of file extensions to their corresponding dataset loading methods.

    Verbose Level Mapping:
        0: No messages
        1: Errors only
        2: Errors and warnings
        3: Errors, warnings, and informational messages
        4: Errors, warnings, informational messages, and debug messages
    """

    def __init__(self, verbose_level: VerboseLevel = VerboseLevel.DEBUG, enable_txt_samples: bool = False) -> None:
        """
        Initialize a DatasetStorage instance.

        Sets up the logger and creates a mapping from file extensions to their corresponding 
        dataset loading methods using partial functions for ease of use.

        Args:
            verbose_level (VerboseLevel): The verbosity level for logging. Defaults to VerboseLevel.DEBUG.
            enable_txt_samples (bool): Boolean flag to process text files as individual samples. Defaults to False.
        """
        self.logger = get_logger(__name__, level=verbose_level)
        self.verbose_level = verbose_level
        self.enable_txt_samples = enable_txt_samples
        # Map file extensions to specific dataset loading methods.
        self.extension_to_method = {
            "txt": partial(self.__load_dataset_from_extension, "text"),
            "csv": partial(self.__load_dataset_from_extension, "csv"),
            "json": partial(self.__load_dataset_from_extension, "json"),
            # Add more mappings as needed.
        }

    def __load_dataset_from_extension(self, data_type: str, files: list[str]) -> HFDataset:
        """
        Load a dataset from files with a specific file extension.

        This method checks if the data type is "text" and if special text sample processing is enabled.
        If so, it loads each text file as an individual sample. Otherwise, it delegates to 
        Hugging Face's built-in load_dataset function.

        Args:
            data_type (str): The type of data to load (e.g., 'text', 'csv', 'json').
            files (list[str]): A list of file paths to load.

        Returns:
            HFDataset: The loaded Hugging Face dataset.
        """
        if data_type == "text" and self.enable_txt_samples:
            return self.__load_text_files_as_samples(files)
        
        # Default behavior: use Hugging Face's load_dataset function.
        return load_dataset(data_type, data_files=files)

    def __load_text_files_as_samples(self, files: list[str]) -> DatasetDict:
        """
        Load text files such that each file represents a single sample.

        This method reads each text file, applies basic preprocessing (e.g., replacing double newlines with a single newline),
        and then aggregates the content into a Hugging Face Dataset wrapped in a DatasetDict.

        Args:
            files (list[str]): A list of text file paths.

        Returns:
            DatasetDict: A dataset dictionary with a single 'train' split containing the loaded samples.
        """
        data = []
        for i, file_path in enumerate(files):
            try:
                with open(file_path, 'r') as f:
                    # Read the file and do a simple preprocessing step.
                    content = f.read().replace('\n\n', '\n')
                data.append({
                    'text': content
                })
            except Exception as e:
                self.logger.error(f"Error loading file {file_path}: {e}")
        
        # Create a dataset from the list of dictionaries.
        dataset = HFDataset.from_list(data)
        return DatasetDict({"train": dataset})

    def _group_files_by_extension(self, files_path: str) -> dict:
        """
        Group files in the specified directory by their file extension.

        This helper method scans the directory for files, determines the extension for each file,
        and builds a dictionary that maps each extension to a list of file paths.

        Args:
            files_path (str): The path to the directory containing the files.

        Returns:
            dict: A dictionary with file extensions as keys and lists of corresponding file paths as values.
        """
        # Scan the directory to retrieve files grouped by their source structure.
        source_dict = scan_directory(files_path)
        extension_files = {}
        for source, files in source_dict.items():
            for file in files:
                # Get the extension of the file.
                extension = file.split(".")[-1]
                file_full_path = os.path.join(files_path, file)
                if extension not in extension_files:
                    extension_files[extension] = [file_full_path]
                else:
                    extension_files[extension].append(file_full_path)

        self.logger.debug(
            f"Grouped ({len(extension_files.keys())}) files by extensions: {list(extension_files.keys())}"
        )
        return extension_files

    def process_files(self, files_path: str, extension: str = None) -> HFDataset:
        """
        Process files within the given directory and build a consolidated dataset.

        The method scans the directory, groups the files by their extensions, 
        then uses the appropriate dataset loading method based on the extension.
        If multiple datasets are produced (one for each supported file extension), they are concatenated.

        Args:
            files_path (str): The directory path containing the files to be processed.
            extension (str, optional): Specific file extension to process (currently not used for filtering).

        Returns:
            HFDataset: The processed and (if necessary) combined Hugging Face dataset.

        Raises:
            ValueError: If the 'files_path' is not a valid directory or if no data is found.
        """
        if not os.path.isdir(files_path):
            raise ValueError(f"Invalid directory path: {files_path}.")

        self.logger.info(
            f"Processing files from '{files_path}' and grouping by file extension."
        )
        datasets = []
        extension_files = self._group_files_by_extension(files_path)
        for extension, files in extension_files.items():
            if extension not in SUPPORTED_EXTENSIONS:
                self.logger.warning(f"Unsupported file extension: {extension}")
                continue

            process_method = self.extension_to_method.get(extension)
            if process_method:
                datasets.append(process_method(files))
            else:
                self.logger.error(
                    f"Could not find Extension processing method for: '{extension}'"
                )

        if datasets and len(datasets) > 0:
            if len(datasets) == 1:
                self.logger.info(
                    f"Dataset successfully built from '{extension}' files."
                )
                return datasets[0]

            self.logger.info(
                f"Produced one dataset per file extension. Combining ({len(datasets)}) datasets into one."
            )
            combined_dataset = concatenate_datasets(
                [dataset["train"] for dataset in datasets]
            )
            return combined_dataset
        else:
            self.logger.error("No data found")
            raise ValueError("No data found")

    def split(self, dataset: DatasetDict, split_ratio: float):
        """
        Split the dataset into training and validation sets.

        This method extracts the 'train' split from the input dataset and, if a non-zero split_ratio is provided,
        uses Hugging Face's train_test_split to create a validation set. The resulting dataset dictionary contains 
        keys 'train' and 'valid', where 'valid' is derived from the split 'test' split.

        Args:
            dataset (DatasetDict): The dataset to be split; must contain a 'train' split.
            split_ratio (float): Fraction of the dataset to be used as the validation set (0 ≤ split_ratio ≤ 1).

        Returns:
            DatasetDict: A dictionary with keys 'train' and 'valid' representing the split datasets.
        """
        dataset_train = dataset["train"]
        if split_ratio == 0:
            dataset_dict = DatasetDict({"train": dataset_train})
            return dataset_dict
        
        # Split the dataset into training and test (to be relabeled as validation) sets.
        split_dataset = dataset_train.train_test_split(test_size=split_ratio)
        dataset_dict = DatasetDict(
            {
                "train": split_dataset["train"],
                "valid": split_dataset["test"],  # Renaming the 'test' split to 'valid'
            }
        )
        return dataset_dict

    def load_from_disk(self, path: str) -> HFDataset:
        """
        Load a dataset from a directory on disk.

        This method verifies that the given path is a valid directory before attempting to load the dataset
        using Hugging Face's load_from_disk function.

        Args:
            path (str): The directory path from which to load the dataset.

        Returns:
            HFDataset: The dataset loaded from disk.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(path):
            raise ValueError(f"Invalid directory path: {path}.")
        return load_from_disk(path)

    def load_from_hub(self, dataset_name: str, **kwargs) -> HFDataset:
        """
        Load a dataset directly from the Hugging Face Hub.

        Args:
            dataset_name (str): The name or identifier of the dataset on Hugging Face Hub.
            **kwargs: Additional keyword arguments to passthrough to the load_dataset function.

        Returns:
            HFDataset: The dataset loaded from the Hub.
        """
        return load_dataset(dataset_name, **kwargs)

    def save_to_disk(
        self,
        dataset: HFDataset,
        output_path: str,
        max_shard_size: str | int | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
    ) -> Path:
        """
        Save the given dataset to disk with support for sharding and multiprocessing.

        The method ensures that the parent directory of the output path exists, creates the directory if necessary,
        and then saves the dataset using Hugging Face's save_to_disk function.

        Args:
            dataset (HFDataset): The dataset to be saved.
            output_path (str): The destination file system path where the dataset will be stored.
            max_shard_size (str | int | None, optional): Maximum size for each shard of the dataset.
            num_shards (int | None, optional): Number of shards to divide the dataset into.
            num_proc (int | None, optional): Number of processes to use during saving.

        Returns:
            Path: A Path object representing the directory where the dataset was saved.
        """
        path = Path(output_path)
        # Ensure that the destination directory exists.
        path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving dataset to '{path}'")
        dataset.save_to_disk(
            str(path),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
        )
        return path
