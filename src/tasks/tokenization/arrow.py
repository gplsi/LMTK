from abc import ABC, abstractmethod
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import os

class ArrowDataset:
    """
    Class for handling the conversion to Arrow format.
    """

    def __init__(self, files_path: str):
        self.files_path = files_path
        self.dataset_name = self.files_path.split('/')[-1]
        self.extension_files = self._group_files_by_extension()
        self.extension_to_method = {
            'txt': self.process_txt_files,
            'csv': self.process_csv_files,
            'json': self.process_json_files,
            # Add more mappings as needed
        }

    def _group_files_by_extension(self):
        files = os.listdir(self.files_path)
        extension_files = {}
        for file in files:
            extension = file.split('.')[-1]
            if extension not in extension_files:
                extension_files[extension] = [os.path.join(self.files_path, file)]
            else:
                extension_files[extension].append(os.path.join(self.files_path, file))
        return extension_files

    def process_files(self):
        datasets = []
        for extension, files in self.extension_files.items():
            process_method = self.extension_to_method.get(extension)
            if process_method:
                datasets.append(process_method(files))
            else:
                print(f"Unsupported file extension: {extension}")
        if datasets:
            combined_dataset = concatenate_datasets(datasets)
            return combined_dataset
        else:
            print("No datasets to combine.")
            return None

    def process_txt_files(self, files : list[str]):
        # Load text files into a dataset
        return load_dataset('text', data_files=files)

    def process_csv_files(self, files: list[str]) -> Dataset:
        # Load CSV files into a dataset
        return load_dataset('csv', data_files=files)

    def process_json_files(self, files: list[str]) -> Dataset:
        # Load JSON files into a dataset
        return load_dataset('json', data_files=files)
    
    def combine_datasets(self, datasets: list[Dataset]) -> Dataset:
        return concatenate_datasets(datasets)
    
    def split(self, dataset: Dataset, split_ratio: float) -> None:
        split_dataset = dataset.train_test_split(test_size=split_ratio)
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'valid': split_dataset['test']  # Renaming 'test' split to 'valid'
        })
        dataset_dict.save_to_disk(dataset_dict_path=f"data/arrow/{self.dataset_name}")  # Save the split dataset to disk
    

# Usage
# files_path = 'path_to_your_folder'
# arrow_dataset = ArrowDataset(files_path)
# combined_dataset = arrow_dataset.process_files()

# if combined_dataset:
#     print(combined_dataset)

SUPPORTED_EXTENSIONS = ['txt', 'csv', 'json']
