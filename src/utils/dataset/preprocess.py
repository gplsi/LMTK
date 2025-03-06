# File: preprocess.py
import os
import argparse
from datasets import load_dataset, DatasetDict, Dataset, Features, Value
from huggingface_hub import create_repo
from src.utils.dataset.utils import scan_directory
from typing import Dict, Generator


def data_generator(txt_files_dict: Dict) -> Generator[str]:
    """
    Generator function to yield each text file's complete content along with its metadata.
    
    Args:
        txt_files_dict (dict): Dictionary mapping data sources to their text file paths.
    
    Yields:
        dict: Contains the 'id', 'file_name', 'source', and 'content' (full file content).
    """
    count = 0
    for source, txt_files in txt_files_dict.items():
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().replace('\n\n', '\n')  # Preprocess content
            yield {
                'id': count,
                'file_name': os.path.basename(txt_file),
                'source': source,
                'content': content
            }
            count += 1

def create_dataset_from_generator(data_files_dict, dataset_name: str):
    """
    Creates a Hugging Face dataset from a generator that loads each file as a single sample.
    
    Args:
        data_files_dict (dict): Dictionary of data sources and their text file paths.
        dataset_name (str): Directory name to save the processed dataset.
    
    Returns:
        DatasetDict: Contains the train and validation splits.
    """
    dataset = Dataset.from_generator(
        data_generator,
        features=Features({
            'id': Value('int32'),
            'file_name': Value('string'),
            'source': Value('string'),
            'content': Value('string')
        }),
        gen_kwargs={'txt_files_dict': data_files_dict},
        keep_in_memory=False
    )
    
    split_dataset = dataset.train_test_split(test_size=0.05)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']  # Renaming 'test' split to 'valid'
    })
    dataset_dict.save_to_disk(dataset_name, max_shard_size="50MB")
    return dataset_dict

def create_dataset_from_files(data_files: str, dataset_name: str):
    """
    Creates a Hugging Face dataset using the default text loader which splits by line.
    
    Args:
        data_files (str): Glob pattern specifying text files.
        dataset_name (str): Directory name to save the dataset.
    
    Returns:
        DatasetDict: Contains training and validation splits.
    """
    dataset = load_dataset('text', data_files=data_files)
    if isinstance(dataset, DatasetDict) and 'train' in dataset:
        dataset = dataset['train']
    split_dataset = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']
    })
    dataset_dict.save_to_disk(dataset_name)
    return dataset_dict

def publish_to_hub(dataset_dict: DatasetDict, repo_name, organization):
    """
    Publishes the dataset to a private repository on the Hugging Face Hub.
    
    Args:
        dataset_dict (DatasetDict): The dataset.
        repo_name (str): Repository name.
        organization (str): Hugging Face organization.
    
    Returns:
        str: URL of the published dataset.
    """
    full_repo_name = f"{organization}/{repo_name}"
    create_repo(full_repo_name, private=True, repo_type="dataset", exist_ok=True,
                token="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")
    url = dataset_dict.push_to_hub(full_repo_name, private=True, token="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")
    return url

class DataHandler:
    def __init__(self, path, dataset_name, dataset_version, split_ratio, max_file_size, txt_as_file=False):
        """
        DataHandler loads and splits datasets from text files.
        
        Args:
            path (str): Directory containing text files.
            dataset_name (str): Name for the dataset.
            dataset_version (str): Version identifier.
            split_ratio (float): Ratio for splitting the dataset.
            max_file_size (int): Maximum file size allowed.
            txt_as_file (bool): If True, each .txt file is loaded as a single sample.
        """
        self.path = path
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.split_ratio = split_ratio
        self.max_file_size = max_file_size
        self.txt_as_file = txt_as_file

        self.data_sources = scan_directory(self.path)
        self.dataset_dict = self._create_dataset()

    def _create_dataset(self):
        """
        Creates the dataset using the appropriate function based on txt_as_file flag.
        """
        if not self.data_sources:
            raise ValueError("No text files found in the specified directory.")
        
        if self.txt_as_file:
            # Use generator: each text file is a single sample.
            print("Loading each text file as an independent sample using the generator approach.")
            return create_dataset_from_generator(self.data_sources, self.dataset_name)
        else:
            # Use Hugging Face default: loads each line as a sample.
            print("Loading text data using Hugging Face's default text loader (each line becomes a sample).")
            # Assume that the path contains only text files for simplicity.
            file_pattern = os.path.join(self.path, "**/*.txt")
            return create_dataset_from_files(file_pattern, self.dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and publish a Hugging Face dataset from text files.")
    parser.add_argument("--path", required=True, help="Path to the directory containing text files")
    parser.add_argument("--dataset_name", required=True, help="Name for the dataset")
    parser.add_argument("--organization", required=True, help="Hugging Face organization name")
    parser.add_argument("--txt_as_file", action="store_true",
                        help="If set, loads each .txt file as a complete sample instead of splitting by line")
    args = parser.parse_args()
    
    try:
        handler = DataHandler(args.path, args.dataset_name, dataset_version="v1",
                              split_ratio=0.1, max_file_size=50 * 1024 * 1024, txt_as_file=args.txt_as_file)
        print(f"Dataset '{args.dataset_name}' created and saved to disk.")
        
        # Publish the dataset to the Hugging Face Hub
        url = publish_to_hub(handler.dataset_dict, args.dataset_name, args.organization)
        print(f"Dataset published to: {url}")
    except ValueError as e:
        print(e)
