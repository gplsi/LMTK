import os
import glob
import argparse
from datasets import load_dataset, DatasetDict, Dataset, Features, Value, load_from_disk, IterableDataset
from sklearn.model_selection import train_test_split
import psutil
from huggingface_hub import HfApi, create_repo


class DataHandler():
    def __init__(self, path, dataset_name, dataset_version, split_ratio, max_file_size):
        
        self.path = path
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.split_ratio = split_ratio
        self.max_file_size = max_file_size
        
        self.data_sources = self.scan_directory()
        self.dataset_dict = self.create_dataset_from_generator()
        
        
    @staticmethod
    def scan_directory(path)


def scan_directory(path):
    """
    Scans the given directory for text files and returns a dictionary of data sources and their files.

    Args:
        path (str): Path to the directory containing subfolders with text files.

    Returns:
        dict: A dictionary where keys are data sources (folder names) and values are lists of text file paths.
    """
    data_sources = {}
    for root, dirs, files in os.walk(path):
        source = os.path.basename(root)
        if source != os.path.basename(path):  # Exclude the root directory itself
            txt_files = [os.path.join(root, file) for file in files if file.endswith('.txt')]
            if txt_files:
                data_sources[source] = txt_files
    return data_sources

def data_generator(txt_files_dict):
    """
    Generator function to yield each text file's content along with a unique ID, file name, and source.
    
    Args:
        txt_files_dict (dict): Dictionary of data sources and their text file paths.

    Yields:
        dict: A dictionary containing the 'id', 'file_name', 'source', and 'content' of each text file.
    """
    count = 0  # Initialize counter for unique ID
    for source, txt_files in txt_files_dict.items():
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().replace('\n\n', '\n')  # Read and preprocess content
                yield {
                    'id': count,
                    'file_name': os.path.basename(txt_file),
                    'source': source,
                    'content': content
                }
                count += 1

def create_dataset_from_generator(data_files_dict: dict, dataset_name: str):
    """
    Creates a Hugging Face dataset from a directory of text files using a generator function,
    then splits it into training and validation sets and saves to disk.
    
    Args:
        data_files_dict (dict): Dictionary of data sources and their text file paths.
        dataset_name (str): Name of the directory to save the processed dataset.
    """
    dataset = Dataset.from_generator(
        data_generator,
        features=Features({
            'id': Value('int32'),
            'file_name': Value('string'),
            'source': Value('string'),
            'content': Value('string')}),
        gen_kwargs={'txt_files_dict': data_files_dict},
        keep_in_memory=False
    )
    
    # Split the dataset into training and validation sets
    split_dataset = dataset.train_test_split(test_size=0.05)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']  # Renaming 'test' split to 'valid'
    })
    
    dataset_dict.save_to_disk(dataset_name, max_shard_size="50MB")  # Save the split dataset to disk
    return dataset_dict

def create_dataset_from_files(data_files: str, dataset_name: str):
    """
    Loads text files as a Hugging Face dataset, performs a train-test split,
    and saves the result to disk.
    
    Note: This function treats each line in the text files as a separate dataset item.

    Args:
        data_files (str): Glob pattern specifying the text files to be included.
        dataset_name (str): Name of the directory to save the processed dataset.
    """
    dataset = load_dataset('text', data_files=data_files)
    
    if isinstance(dataset, DatasetDict) and 'train' in dataset:
        dataset = dataset['train']
    
    split_dataset = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']  # Renaming 'test' split to 'valid'
    })
    
    dataset_dict.save_to_disk(dataset_name)  # Save the split dataset to disk
    
def publish_to_hub(dataset_dict: DatasetDict, repo_name, organization):
    """
    Publishes the dataset to a private repository in the specified Hugging Face organization.

    Args:
        dataset_dict (DatasetDict): The dataset to publish.
        repo_name (str): Name of the repository to create/update.
        organization (str): Name of the Hugging Face organization.

    Returns:
        str: URL of the published dataset.
    """
    full_repo_name = f"{organization}/{repo_name}"
    
    # Create or get the repository
    create_repo(full_repo_name, private=True, repo_type="dataset", exist_ok=True, token="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")
    
    # Push the dataset to the Hub
    url = dataset_dict.push_to_hub(full_repo_name, private=True, token="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")
    
    return url


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and publish a Hugging Face dataset from text files.")
    parser.add_argument("--path", help="Path to the directory containing text files")
    parser.add_argument("--dataset_name", help="Name for the dataset")
    parser.add_argument("--organization", help="Hugging Face organization name")
    args = parser.parse_args()

    data_sources = scan_directory(args.path)
    
    if not data_sources:
        print("No text files found in the specified directory.")
    else:
        dataset_dict = create_dataset_from_generator(data_sources, args.dataset_name)
        print(f"Dataset '{args.dataset_name}' created and saved to disk.")
        
        # Publish the dataset to the Hugging Face Hub
        url = publish_to_hub(dataset_dict, args.dataset_name, args.organization)
        print(f"Dataset published to: {url}")