import argparse
import shutil
import glob
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset, Features, Value, load_from_disk, IterableDataset, Sequence
import psutil
import os
from transformers import AutoTokenizer
from arguments import *


args = parse_args(sequence_length=2048, model_name="meta-llama/Meta-Llama-3-8B", input_dir="datos_txt_limpios")

def split_large_file(file_path, max_size=104857600, output_dir=None):
    """
    Splits a large text file into smaller ones, each not exceeding a specified size limit.
    Splitting is done at line breaks to ensure sentences are not broken across files.

    Args:
    - file_path (str): The path to the original large file.
    - max_size (int): Maximum file size in bytes. Defaults to 100MB (104857600 bytes).
    - output_dir (str): Directory where the split files will be saved. Defaults to the same directory as the original file.

    Returns:
    None
    """
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = []
        current_size = 0
        part = 1
        
        for line in file:
            line_size = len(line.encode('utf-8'))
            if current_size + line_size > max_size:
                # Save the current content to a new file and reset variables
                output_path = os.path.join(output_dir, f"{name}_{part}{ext}")
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(''.join(content))
                content = []
                current_size = 0
                part += 1
            
            content.append(line)
            current_size += line_size
        
        # Don't forget to save the last part if there's any
        if content:
            output_path = os.path.join(output_dir, f"{name}_{part}{ext}")
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(''.join(content))
            
            
def split_files_in_directory(directory, output_dir, max_size=104857600):
    """
    Iterates through all .txt files in the specified directory.
    If a file exceeds the specified size, it is split into smaller files.

    Args:
    - directory (str): The directory to search for text files.
    - max_size (int): Maximum file size in bytes for the split. Defaults to 100MB.

    Returns:
    None
    """
    
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            split_large_file(file_path, max_size, output_dir)
        else:
            # Copy the file to the output directory
            output_path = os.path.join(output_dir, file_name)
            shutil.copy(file_path, output_path)


def data_generator(txt_files):
    """
    Generator function to yield each text file's content along with a unique ID and file name.
    
    Args:
        txt_files (list): List of paths to text files.

    Yields:
        dict: A dictionary containing the 'id', 'file_name', and 'content' of each text file.
    """
    count = 0  # Initialize counter for unique ID
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            content = f.read().replace('\n\n', '\n')  # Read and preprocess content
            yield {
                'id': count,
                'file_name': os.path.basename(txt_file),
                'content': content
            }
            count += 1

def create_dataset_from_generator(data_files: str, dataset_name: str):
    """
    Creates a Hugging Face dataset from a directory of text files using a generator function,
    then splits it into training and validation sets and saves to disk.
    
    Args:
        data_files (str): Glob pattern specifying the text files to be included.
        dataset_name (str): Name of the directory to save the processed dataset.
    """
    txt_files = glob.glob(data_files)  # Collect text file paths
    dataset = Dataset.from_generator(
        data_generator,
        features=Features({
            'id': Value('int32'),
            'file_name': Value('string'),
            'content': Value('string')}),
        gen_kwargs={'txt_files': txt_files},
        keep_in_memory=False
    )
    
    # Split the dataset into training and validation sets
    split_dataset = dataset.train_test_split(test_size=args['test_split'])
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']  # Renaming 'test' split to 'valid'
    })
    
    dataset_dict.save_to_disk(dataset_name)  # Save the split dataset to disk


def load_dataset_test_memory(dataset_name: str):
    """
    Loads a dataset from disk and prints the memory usage before and after loading
    to demonstrate the memory efficiency of disk-based datasets.

    Args:
        dataset_name (str): The name of the directory where the dataset is saved.

    Returns:
        Dataset: The loaded Hugging Face dataset.
    """
    def print_memory_usage():
        """Prints the current process's memory usage."""
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

    print("Before loading dataset:")
    print_memory_usage() 

    dataset = load_from_disk(dataset_name, keep_in_memory=False)
    
    print(f"Loaded dataset '{dataset_name}' with {len(dataset['train'])} train examples and {len(dataset['valid'])} valid examples")
    print(f"After loading dataset: {print_memory_usage()}")
    
    return dataset


DATA_PATH = "/usrvol/small_valencian_dataset"

def get_used_memory():
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / (1024**2)} MB"

def load_dataset_from_disk(path = None):
    if path is None:
        path = "valencian_dataset"
    
    return load_from_disk(path, keep_in_memory=False)

def tokenize_dataset(tokenizer, dataset, tokenized_dataset_name, context_length=8000, overlap=2000):
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define the dataset features
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
    })
    
    def tokenize_and_filter(batch):
        outputs = tokenizer(
            batch["content"],
            truncation=True,
            max_length=context_length,  # Do not truncate here to handle splitting manually
            return_overflowing_tokens=True,
            return_length=True,
            stride=overlap,
            padding=True
        )
        
        input_batch = []
        att_mask_batch = []
        labels_batch=[]
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            input_batch.append(input_ids)
            att_mask_batch.append(attention_mask)
            """ labels = input_ids[1:]  # Skip the first token
            labels.append(-100) """
            labels = input_ids
            labels_batch.append(labels)  
            
        # Ensure output for every key
        assert len(input_batch) == len(att_mask_batch) == len(labels_batch), "Mismatch in output lengths."

        return {
            "input_ids": input_batch,
            "attention_mask": att_mask_batch,
            "labels": labels_batch,
        }

    os.makedirs(tokenized_dataset_name, exist_ok=True)
    tokenized_datasets = dataset.map(
        tokenize_and_filter, 
        batched=True,
        batch_size=1,
        remove_columns=dataset['train'].column_names,
        keep_in_memory=False,
        load_from_cache_file=False,
        features=features
    )
    
    tokenized_datasets.save_to_disk(tokenized_dataset_name)
    return tokenized_dataset_name


def get_last_folders(folder, folder_list):
    files = os.listdir(folder)
    for file in files:
        if not ".txt" in file:
            if not file.endswith(".json"):
                folder = folder + "/" + file
                get_last_folders(folder, folder_list)
        else:
            folder_list.append(folder)
    return folder_list


if __name__ == "__main__":
    # Define the directory to scan and the maximum file size    
    directory_to_scan = args['initial_dataset_dir']
    output_dir = args['dataset_dir']
    max_file_size = args['max_file_size']  # 100MB in bytes
    
    folders_to_scan = os.listdir(directory_to_scan)
    print('Split larges files for each dataset...')
    for folder in folders_to_scan:
        folder = directory_to_scan + "/" + folder
        split_files_in_directory(directory=folder, output_dir=output_dir, max_size=max_file_size)

    data_files = args['dataset_dir'] + "/*.txt"
    dataset_name = args['arrow_dataset_dir']

    # Option to create a dataset from full text files using a generator
    print('Creating HuggingFace Dataset')
    create_dataset_from_generator(data_files, dataset_name)
    
    # Demonstrate memory usage
    #load_dataset_from_disk_test_memory(dataset_name)
    
    dataset = load_dataset_from_disk(path= args['arrow_dataset_dir'])
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name'])
    print('Tokenizing the datasets...')
    tokenized_dataset_name = tokenize_dataset(tokenizer, dataset, tokenized_dataset_name=args['tokenized_dataset_dir'], 
                                              context_length=args['max_length'], overlap=args['overlap'])
    print(f"Tokenized dataset with context length {args['max_length']} and overlap {args['overlap']} saved as {tokenized_dataset_name}")
    print(f"Memory used: {get_used_memory()}")
