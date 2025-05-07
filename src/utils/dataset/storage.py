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
from typing import Dict, Optional, List, Any, Union
import json
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

    .. note::
        Use :no-index: in duplicate object descriptions for Sphinx.

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
            "jsonl": partial(self.__load_dataset_from_extension, "json"),
            # Add more mappings as needed.
        }
        self.text_key = None  # Will be set when loading JSON/JSONL files if specified in config

    def __load_dataset_from_extension(self, data_type: str, files: list[str]) -> Union[HFDataset, DatasetDict]:
        """
        Load a dataset from files with a specific file extension.

        This method checks if the data type is "text" and if special text sample processing is enabled.
        If so, it loads each text file as an individual sample. Otherwise, it delegates to 
        Hugging Face's built-in load_dataset function.

        Args:
            data_type (str): The type of data to load (e.g., 'text', 'csv', 'json').
            files (list[str]): A list of file paths to load.

        Returns:
            Union[HFDataset, DatasetDict]: The loaded dataset.
        """
        
        if data_type == "text" and self.enable_txt_samples:
            return self.__load_text_files_as_samples(files)
        
        # Special handling for JSON/JSONL files when we're processing JSON format
        if data_type == "json" and (self.text_key is not None or any(f.endswith('.jsonl') for f in files)):
            return self.__load_json_with_text_key(files)
        
        # Default behavior: use Hugging Face's built-in loaders
        return load_dataset(data_type, data_files=files)
    
    
    def __load_json_with_text_key(self, files: list[str]) -> DatasetDict:
        """
        Load JSON/JSONL files and extract text.
        
        If text_key is specified, extract text from that key.
        If text_key is not specified, use the entire JSON object as text.
        
        Args:
            files (list[str]): List of JSON/JSONL file paths to load.
        
        Returns:
            DatasetDict: A dataset where each entry contains the extracted text.
        """
        
        data = []
        for file_path in files:
            try:
                file_extension = file_path.split('.')[-1].lower()
                is_jsonl = file_extension == 'jsonl'
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    if is_jsonl:
                        # Process line by line for JSONL
                        for i, line in enumerate(f):
                            try:
                                json_obj = json.loads(line.strip())
                                self.__process_json_entry(json_obj, data, file_path, i)
                            except json.JSONDecodeError:
                                self.logger.error(f"Error parsing JSON at line {i+1} in {file_path}: Invalid JSON format")
                    else:
                        # Process as single JSON object or array
                        json_content = json.load(f)
                        if isinstance(json_content, list):
                            for i, item in enumerate(json_content):
                                self.__process_json_entry(item, data, file_path, i)
                        else:
                            self.__process_json_entry(json_content, data, file_path, 0)
                            
            except Exception as e:
                self.logger.error(f"Error loading file {file_path}: {str(e)}")
        
        if not data:
            if self.text_key:
                self.logger.error(f"No valid data found in JSON/JSONL files with text_key: {self.text_key}")
            else:
                self.logger.error("No valid data found in JSON/JSONL files")
            raise ValueError("No valid data found in JSON/JSONL files")
            
        # Create a Dataset from the list of dictionaries and wrap in a DatasetDict
        dataset = HFDataset.from_list(data)
        return DatasetDict({"train": dataset})
    
    def __process_json_entry(self, json_obj: Any, data: List[Dict[str, str]], file_path: str, index: int) -> None:
        """
        Process a JSON entry and extract text. 
        
        If text_key is specified, extract text from that key.
        If text_key is not specified, convert the entire JSON object to string.
        
        Args:
            json_obj: The JSON object to process
            data: The list to append the extracted data to
            file_path: The source file path (for error reporting)
            index: The index of the entry (for error reporting)
        """
        # If text_key is specified, extract text from that key
        if self.text_key:
            if not isinstance(json_obj, dict):
                self.logger.warning(f"Skipping non-dictionary entry at index {index} in {file_path}")
                return
                
            if self.text_key not in json_obj:
                self.logger.warning(f"Key '{self.text_key}' not found in entry at index {index} in {file_path}")
                return
                
            text_value = json_obj[self.text_key]
            if not isinstance(text_value, str):
                self.logger.warning(f"Value for key '{self.text_key}' is not a string at index {index} in {file_path}")
                return
                
            data.append({"text": text_value})
        else:
            # If no text_key is specified, use the entire JSON object as text
            try:
                # Convert the JSON object to a pretty-printed string
                text_value = json.dumps(json_obj, ensure_ascii=False, indent=2)
                data.append({"text": text_value})
            except Exception as e:
                self.logger.warning(f"Failed to convert JSON to string at index {index} in {file_path}: {e}")
    
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
        source_dict = scan_directory(files_path, logger=self.logger)
        extension_files = {}
        for source, files in source_dict.items():
            for file in files:
                extension = file.split(".")[-1]
                if extension not in extension_files:
                    extension_files[extension] = [os.path.join(files_path, file)]
                else:
                    extension_files[extension].append(os.path.join(files_path, file))

        self.logger.debug(
            f"Grouped files by extensions: {[f'{k} ({len(v)})' for k, v in extension_files.items()]}"
        )
        return extension_files

    def process_files(self, files_path: str, file_config: Optional[Dict] = None) -> HFDataset:
        """
        Process files within the given directory and build a consolidated dataset.

        The method scans the directory, groups the files by their extensions, 
        then uses the appropriate dataset loading method based on the extension.
        If multiple datasets are produced (one for each supported file extension), they are concatenated.

        Args:
            files_path (str): Path to the directory containing files.
            file_config (Optional[Dict]): Configuration for file processing, including format and text_key.
        
        Returns:
            HFDataset: The processed dataset.
        """
         
        if not os.path.isdir(files_path):
            raise ValueError(f"Invalid directory path: {files_path}.")
            
        # Set text_key if specified in file_config
        self.text_key = None  # Reset text_key
        if file_config and "text_key" in file_config:
            self.text_key = file_config.get("text_key")
            self.logger.info(f"Using text_key: {self.text_key} for JSON/JSONL files")
        else:
            format_str = file_config.get("format") if file_config else None
            if format_str in ["json", "jsonl"]:
                self.logger.info("No text_key specified for JSON/JSONL files, using entire JSON as content")

        self.logger.info(
            f"Processing files from '{files_path}' and grouping by file extension."
        )
        datasets = []
        extension_files = self._group_files_by_extension(files_path)
        
        # If a specific format is requested in file_config, filter for that format
        specific_format = None
        if file_config and "format" in file_config and file_config["format"] != "any":
            specific_format = file_config["format"]
            
        for extension, files in extension_files.items():
            if specific_format and extension != specific_format:
                self.logger.debug(f"Skipping {extension} files, only processing {specific_format}")
                continue
                
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
                    f"Dataset successfully built from '{extension if 'extension' in locals() else specific_format}' files."
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

    def split(self, dataset: Union[HFDataset, DatasetDict], split_ratio: float) -> DatasetDict:
        """
        Split a dataset into training and validation sets.
        
        Args:
            dataset (Union[HFDataset, DatasetDict]): The dataset to split.
            split_ratio (float): The ratio of data to use for validation.
        
        Returns:
            DatasetDict: A dataset dictionary containing 'train' and optionally 'valid' splits.
        """
        
        if isinstance(dataset, DatasetDict):
            dataset_train = dataset["train"]
        else:
            dataset_train = dataset
            
        if split_ratio == 0:
            dataset_dict = DatasetDict({"train": dataset_train})
            return dataset_dict
            
        split_dataset = dataset_train.train_test_split(test_size=split_ratio)
        dataset_dict = DatasetDict(
            {
                "train": split_dataset["train"],
                "valid": split_dataset["test"],  # Renaming 'test' split to 'valid'
            }
        )
        return dataset_dict

    def load_from_disk(self, path: str) -> HFDataset:
        """
        Load a dataset from disk.
        
        Args:
            path (str): Path to the saved dataset.
            
        Returns:
            HFDataset: The loaded dataset.
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
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
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
        path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving dataset to '{path}'")
        dataset.save_to_disk(
            str(path),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
        )
        return path
