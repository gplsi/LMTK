from pathlib import Path
from typing import Dict
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)
import os
from functools import partial
from enum import IntEnum
from src.utils.logging import VerboseLevel, get_logger
from src.utils.dataset.utils import SUPPORTED_EXTENSIONS, scan_directory


class DatasetStorage:
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
        self.logger = get_logger(__name__, level=verbose_level)
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

        self.logger.debug(
            f"Grouped ({len(extension_files.keys())}) files by extensions: {extension_files.keys()}"
        )
        return extension_files

    def process_files(self, files_path: str, extension: str = None) -> Dataset:
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

    def split(self, dataset: DatasetDict, split_ratio: float) -> None:
        dataset = dataset["train"]
        split_dataset = dataset.train_test_split(test_size=split_ratio)
        dataset_dict = DatasetDict(
            {
                "train": split_dataset["train"],
                "valid": split_dataset["test"],  # Renaming 'test' split to 'valid'
            }
        )
        return dataset_dict

    def load_from_disk(self, path: str) -> Dataset:
        if not os.path.isdir(path):
            raise ValueError(f"Invalid directory path: {path}.")

        return load_from_disk(path)

    def load_from_hub(self, dataset_name: str, **kwargs) -> Dataset:
        return load_dataset(dataset_name, **kwargs)

    def save_to_disk(
        self,
        dataset: Dataset,
        output_path: str,
        max_shard_size: str | int | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
    ) -> Path:
        """Save dataset to disk with proper error handling."""
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


# Usage
# files_path = 'path_to_your_folder'
# arrow_dataset = ArrowDataset(files_path)
# combined_dataset = arrow_dataset.process_files()

# if combined_dataset:
#     print(combined_dataset)
