from abc import ABC, abstractmethod
from typing import Dict
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
import os
from functools import partial
from enum import IntEnum

SUPPORTED_EXTENSIONS = ["txt", "csv", "json"]

class VerboseLevel(IntEnum):
    NONE = 0
    ERRORS = 1
    WARNINGS = 2
    INFO = 3
    DEBUG = 4


class DatasetHelper:
    """
    Class for handling the conversion to Arrow format.

    verbose_level (int): Level of verbosity for printing messages (0 to 4).

    supported levels:
    0: No messages
    1: Errors only
    2: Errors and warnings
    3: Errors, warnings, and information
    4: Errors, warnings, information, and debug

    """

    def __init__(self, verbose_level: VerboseLevel = VerboseLevel.DEBUG):
        if (verbose_level < VerboseLevel.NONE) or (verbose_level > VerboseLevel.DEBUG):
            raise ValueError("Invalid verbose level. Must be between None and Debug.")

        self.verbose_level = verbose_level
        self.extension_to_method = {
            "txt": partial(self.__load_dataset_from_extension, "text"),
            "csv": partial(self.__load_dataset_from_extension, "csv"),
            "json": partial(self.__load_dataset_from_extension, "json"),
            # Add more mappings as needed
        }

    def __load_dataset_from_extension(self, data_type: str, files: list[str]):
        # Load text files into a dataset
        return load_dataset(data_type, data_files=files)
    
    def __print(self, message: str, level: VerboseLevel = VerboseLevel.NONE):
        if self.verbose_level >= level:
            print(message)

    def _group_files_by_extension(self, files_path: str) -> dict:
        source_dict = scan_directory(files_path)
        extension_files = {}
        for source, files in source_dict.items():
            for file in files:
                extension = file.split(".")[-1]
                if extension not in extension_files:
                    extension_files[extension] = [os.path.join(files_path, file)]
                else:
                    extension_files[extension].append(os.path.join(files_path, file))

        self.__print(
            f"Grouped ({len(extension_files.keys())}) files by extensions: {extension_files.keys()}",
            VerboseLevel.DEBUG,
        )
        return extension_files

    def process_files(self, files_path: str, extension: str = None) -> Dataset:
        if not os.path.isdir(files_path):
            raise ValueError(f"Invalid directory path: {files_path}.")
        
        datasets = []
        extension_files = self._group_files_by_extension(files_path)
        for extension, files in extension_files.items():
            if extension not in SUPPORTED_EXTENSIONS:
                self.__print(
                    f"Unsupported file extension: {extension}", VerboseLevel.WARNINGS
                )
                continue

            process_method = self.extension_to_method.get(extension)
            if process_method:
                datasets.append(process_method(files))
            else:
                self.__print(
                    f"Could not find Extension processing method for: '{extension}'",
                    VerboseLevel.ERRORS,
                )

        if datasets and len(datasets) > 0:
            if len(datasets) == 1:
                return datasets[0]
            
            combined_dataset = concatenate_datasets([dataset['train'] for dataset in datasets])
            return combined_dataset
        else:
            self.__print("No datasets to combine.", VerboseLevel.WARNINGS)
            return None

    def split(self, dataset: Dataset, split_ratio: float) -> None:
        split_dataset = dataset.train_test_split(test_size=split_ratio)
        dataset_dict = DatasetDict(
            {
                "train": split_dataset["train"],
                "valid": split_dataset["test"],  # Renaming 'test' split to 'valid'
            }
        )
        dataset_dict.save_to_disk(
            dataset_dict_path=f"data/arrow/{self.dataset_name}"
        )  # Save the split dataset to disk

    def load_from_disk(self, path: str) -> Dataset:
        if not os.path.isdir(path):
            raise ValueError(f"Invalid directory path: {path}.")
        
        return load_from_disk(path)
    
    def load_from_hub(self, dataset_name: str, **kwargs) -> Dataset:
        return load_dataset(dataset_name, **kwargs)

def scan_directory(path, extension: str = None) -> Dict:
    """
    Scans the given directory for text files and returns a dictionary of data sources and their files.

    Args:
        path (str): Path to the directory containing subfolders with text files.

    Returns:
        dict: A dictionary where keys are data sources (folder names) and values are lists of text file paths.
    """
    if (extension is not None) and (extension not in SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported file extension: {extension}.")

    data_sources = {}
    for root, dirs, files in os.walk(path):
        source = os.path.basename(root)
        if extension is not None:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.endswith(f".{extension}")
            ]  # Filter by extension
        else:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.split(".")[-1] in SUPPORTED_EXTENSIONS
            ]

        if data_files:
            data_sources[source] = data_files  # Exclude the root directory itself

    return data_sources


# Usage
# files_path = 'path_to_your_folder'
# arrow_dataset = ArrowDataset(files_path)
# combined_dataset = arrow_dataset.process_files()

# if combined_dataset:
#     print(combined_dataset)
