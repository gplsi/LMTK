"""
Module: orchestrator.py

This module defines the BaseOrchestrator class, which provides a framework for loading datasets
based on a given configuration. The orchestrator handles logging, dataset validation, loading,
and optional splitting. It supports local datasets (loaded from disk or files) and sets the stage
for future integration with HuggingFace datasets.
"""

import os
import json
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

    def _get_files_from_directory(self):
            """
            Get a list of all files in the dataset directory.
            """
            # use os.walk to get all files in the directory
            if not os.path.exists(self.config.dataset.nameOrPath):
                self.logger.error(f"Dataset path '{self.config.dataset.nameOrPath}' does not exist")
                raise ValueError(f"Dataset path '{self.config.dataset.nameOrPath}' does not exist")
            if not os.path.isdir(self.config.dataset.nameOrPath):
                self.logger.error(f"Dataset path '{self.config.dataset.nameOrPath}' is not a directory")
                raise ValueError(f"Dataset path '{self.config.dataset.nameOrPath}' is not a directory")
            files = []
            for root, dirs, filenames in os.walk(self.config.dataset.nameOrPath):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
            if not files:
                self.logger.warning(f"No files found in directory '{self.config.dataset.nameOrPath}'")    
            return files

    def _json_handler_with_keys(self, keys_found: dict[str, str]) -> HFDataset:
        # Create a dataset with the specific keys, the files are jsons
        all_splits = {}
        for key, file_path in keys_found.items():
            self.logger.info(f"Processing file for key '{key}': {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataJson = json.load(f)
                if isinstance(dataJson, list):
                    all_splits[key] = dataJson
                elif isinstance(dataJson, dict):
                    all_splits[key] = [dataJson]
                else:
                    self.logger.warning(f"Invalid data format in file '{file_path}', expected list or dict, got {type(dataJson)}")
                    continue
            except Exception as e:
                self.logger.error(f"Error reading file '{file_path}': {e}")
                continue

        return all_splits
   
    def _json_handler(self, files: list[str]) -> HFDataset:
        """
        Handle JSON files and extract data based on specific keys.

        :param files: List of file paths to process.
        :type files: list[str]
        :param specific_keys: Optional list of keys to look for in the files.
        :type specific_keys: Optional[list[str]]

        :returns: A HuggingFace dataset containing the processed data.
        :rtype: HFDataset
        """
        
        all_splits = {}
        for file in files:
            self.logger.info(f"Processing file: {file}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    dataJson = json.load(f)
                if isinstance(dataJson, list):
                    all_splits[file] = dataJson
                elif isinstance(dataJson, dict):
                    all_splits[file] = [dataJson]
                else:
                    self.logger.warning(f"Invalid data format in file '{file}', expected list or dict, got {type(dataJson)}")
                    continue
            except Exception as e:
                self.logger.error(f"Error reading file '{file}': {e}")
                continue
        return all_splits

    
    def _load_instruction_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Load and process an instruction dataset.

        This method processes the dataset by applying the instruction tokenizer to each example.
        It is designed to be overridden by subclasses for specific dataset types.

        :param dataset: The HuggingFace dataset to process.
        :type dataset: HFDataset

        :returns: The processed HuggingFace dataset.
        :rtype: HFDataset
        """
        if self.config.dataset.format == "files":
            self.logger.info(f"Loading Instruction dataset from files at dir '{self.config.dataset.nameOrPath}'")

            files = self._get_files_from_directory()

            specific_keys = self.config.dataset.get("specific_keys", None)
            if specific_keys:
                self.logger.info(f"Using specific keys for instruction dataset: {specific_keys}")
                keys_found = {}
                
                # look for the keys as exact matches in the files paths, make sure it is the full name between slashes
                for file in files:
                    for key in specific_keys:
                        if f"/{key}/" in file:
                            keys_found[key] = file
                if len(keys_found) == 0:
                    self.logger.error("No specific keys found in files")
                    return HFDataset.from_list([])  # Return empty dataset if no keys found

                self.logger.info(f"Found specific keys in files: {keys_found}")
                
                # Create a dataset with the specific keys, the files are jsons
                all_splits = self._json_handler_with_keys(keys_found)

                if all_splits:
                    # Combine all data into a single dataset
                    combined_data = []
                    for key, data in all_splits.items():
                        for item in data:
                            item['key'] = key
                            combined_data.append(item)
                    return HFDataset.from_list(combined_data)
                else:
                    self.logger.warning("No valid data collected from multilanguage files, returning empty dataset.")
                    return HFDataset.from_list([])
            else:
                self.logger.info("No specific keys provided, using all the files found in the specified directory")
                # Create a dataset with all the files found in the specified directory
                all_splits = self._json_handler(files)
                if all_splits:
                    # Combine all data into a single dataset
                    combined_data = []
                    for file, data in all_splits.items():
                        for item in data:
                            item['file'] = file
                            combined_data.append(item)
                    return HFDataset.from_list(combined_data)
                else:
                    self.logger.warning("No valid data collected from multilanguage files, returning empty dataset.")
                    return HFDataset.from_list([])
        elif self.config.dataset.format == "dataset":
            self.logger.info(f"Loading Instruction dataset from path '{self.config.dataset.nameOrPath}'")
            dataset = DatasetStorage(self.verbose_level).load_from_disk(self.config.dataset.nameOrPath)
            
            if self.config.test_size:
                self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")

        elif self.config.dataset.format == "multilanguage_files":
            self.logger.info(f"Loading multilanguage dataset from files at dir '{self.config.dataset.nameOrPath}'")

            languagesToLoad = self.config.dataset.languagesToLoad if hasattr(self.config.dataset, 'languagesToLoad') else None

            if languagesToLoad is not None:
                self.logger.info(f"Loading only languages: {languagesToLoad}")
            else:
                self.logger.info("Loading all languages from the dataset")
                languagesToLoad = os.listdir(self.config.dataset.nameOrPath)

            # Don't create empty dataset here - collect all data first
            all_splits = {}

            for lang in languagesToLoad:
                lang_path = os.path.join(self.config.dataset.nameOrPath, lang)
                
                if not os.path.exists(lang_path):
                    self.logger.warning(f"Language path '{lang_path}' does not exist, skipping.")
                    continue
                else:
                    self.logger.info(f"Processing language: {lang} at path '{lang_path}'")

                # Collect all data for this language
                language_data = []

                for file in os.listdir(lang_path):
                    file_path = os.path.join(lang_path, file)
                    if not os.path.isfile(file_path):
                        self.logger.warning(f"File '{file_path}' is not a valid file, skipping.")
                        continue
                    
                    self.logger.info(f"Processing file: {file_path}")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            dataJson = json.load(f)
                        if isinstance(dataJson, list):
                            language_data.extend(dataJson)
                        elif isinstance(dataJson, dict):
                            language_data.append(dataJson)
                        else:
                            self.logger.warning(f"Invalid data format in file '{file_path}', expected list or dict, got {type(dataJson)}")
                            continue
                    except Exception as e:
                        self.logger.error(f"Error reading file '{file_path}': {e}")
                        continue

                # Store the data for this language
                if language_data:
                    all_splits[lang] = language_data
                    self.logger.info(f"Collected {len(language_data)} examples for language '{lang}'")

            if all_splits:
                # Combine all data into a single dataset with language labels
                combined_data = []
                for lang, data in all_splits.items():
                    for item in data:
                        # Add language label to each example
                        item['language'] = lang
                        combined_data.append(item)
                
                dataset = HFDataset.from_list(combined_data)
            else:
                self.logger.warning("No valid data collected from multilanguage files, returning empty dataset.")
                dataset = HFDataset.from_list([])

            return dataset
    
    def _load_general_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Load and process a general dataset.

        This method processes the dataset by applying the general tokenizer to each example.
        It is designed to be overridden by subclasses for specific dataset types.

        :param dataset: The HuggingFace dataset to process.
        :type dataset: HFDataset

        :returns: The processed HuggingFace dataset.
        :rtype: HFDataset
        """

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

        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
    
                

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
        
        if self.config.tokenizer.task in ["clm_training", "mlm_training"]:
            return self._load_general_dataset(self.config.dataset)

        elif self.config.tokenizer.task == "instruction":
            return self._load_instruction_dataset(self.config.dataset)

            

       
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
