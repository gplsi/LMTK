from datasets import load_dataset, load_from_disk, Features, Sequence, Value, Dataset
import psutil
import os
from transformers import AutoTokenizer
from tokenization_functions import TOKENIZE_FUNCTION_HANDLER
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    
    """
    Abstract class for handling the tokenization, currently implemented for 
    loading datasets from Hugging Face Hub and from disk, and tokenizing them 
    into disk. This class is intended for continual pretraining and large 
    datasets in general that should not be tokenized on the fly.
    """
    
    def __init__(
        self,
        dataset_config: dict,
        dataset_name: str = None,
        path: str = None,
        tokenizer_name: str = None,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.path = path
        self.tokenizer_name = tokenizer_name
        self.kwargs = kwargs
    
        assert self.dataset_name is not None or self.path is not None, "Either dataset name or path must be provided."
        assert self.dataset_name is None or self.path is None, "Only one of dataset name or path must be provided."
    
        if self.dataset_name is not None:
            self.loaded_from_hub = True
        else:
            self.loaded_from_hub = False
            
        if self.path is not None:
            self.loaded_from_disk = True
        else:
            self.loaded_from_disk = False
    
    def load_dataset_from_hub(self):
        streaming_config = self.dataset_config
        streaming_config["streaming"] = True
        return load_dataset(self.dataset_name, **self.dataset_config)

    def load_dataset_from_disk(self):
        return load_from_disk(self.path)
    
    def tokenize_dataset(self, dataset: Dataset, tokenized_dataset_name: str, context_length: int, overlap: int):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define the dataset features
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
        })
        
        os.makedirs(tokenized_dataset_name, exist_ok=True)
        tokenized_datasets = dataset.map(
        TOKENIZE_FUNCTION_HANDLER['continual_pretraining'],  
        batched=True,
        batch_size=1,
        remove_columns=dataset['train'].column_names,
        keep_in_memory=False,
        load_from_cache_file=False,
        features=features
        )
    
        tokenized_datasets.save_to_disk(tokenized_dataset_name)
        return tokenized_dataset_name

        
        
    


# def load_dataset_from_hub(dataset_name: str, dataset_config: dict = None):
#     if dataset_config is None:
#         dataset_config = {"streaming": True}

#     return load_dataset(dataset_name, **dataset_config)


# def get_used_memory():
#     process = psutil.Process(os.getpid())
#     return f"{process.memory_info().rss / (1024**2)} MB"


# def load_dataset_from_disk(path: str = None):
#     if path is None:
#         # path = "/data" + "/dbpedia_rdf_analysis_results_dataset_2024_07_09_from_001_to_013_train_test"
#         # path = "/home/gplsi/NAS/GPLSI/llm-train-tokenizer-custom-dataset-main" + "/dbpedia_rdf_analysis_results_dataset_2024_07_09_from_001_to_013_train_test"
#         path = "/data/new_valencian_dataset"

#     return load_from_disk(path, keep_in_memory=False)


# def tokenize_dataset(
#     tokenizer, dataset, tokenized_dataset_name, context_length=8000, overlap=2000
# ):
#     tokenizer.pad_token = tokenizer.eos_token

#     # Define the dataset features
#     features = Features(
#         {
#             "input_ids": Sequence(Value("int32")),
#             "attention_mask": Sequence(Value("int32")),
#             "labels": Sequence(Value("int32")),
#         }
#     )

#     def tokenize_and_filter(batch):
#         outputs = tokenizer(
#             batch["content"],  #'Content'
#             truncation=True,
#             max_length=context_length,  # Do not truncate here to handle splitting manually
#             return_overflowing_tokens=True,
#             return_length=True,
#             stride=overlap,
#             padding=True,
#         )

#         input_batch = []
#         att_mask_batch = []
#         labels_batch = []
#         for length, input_ids, attention_mask in zip(
#             outputs["length"], outputs["input_ids"], outputs["attention_mask"]
#         ):
#             input_batch.append(input_ids)
#             att_mask_batch.append(attention_mask)
#             labels_batch.append(input_ids)

#         # Ensure output for every key
#         assert (
#             len(input_batch) == len(att_mask_batch) == len(labels_batch)
#         ), "Mismatch in output lengths."

#         return {
#             "input_ids": input_batch,
#             "attention_mask": att_mask_batch,
#             "labels": labels_batch,
#         }

#     os.makedirs(tokenized_dataset_name, exist_ok=True)
#     tokenized_datasets = dataset.map(
#         tokenize_and_filter,
#         batched=True,
#         batch_size=5000,
#         remove_columns=dataset["train"].column_names,
#         keep_in_memory=False,
#         load_from_cache_file=False,
#         features=features,
#     )

#     tokenized_datasets.save_to_disk(tokenized_dataset_name)
#     return tokenized_dataset_name


# if __name__ == "__main__":
#     for tokenizer_name in ["meta-llama/Meta-llama-3-8B"]:
#         for size in [2048]:
#             dataset = load_dataset_from_disk()
#             tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#             tokenized_dataset_name = tokenize_dataset(
#                 tokenizer,
#                 dataset,
#                 tokenized_dataset_name=f"{tokenizer_name}-prueba-{size}",
#                 context_length=size,
#                 overlap=size // 4,
#             )
#             print(
#                 f"Tokenized dataset with context length {size} and overlap {size//4} saved as {tokenized_dataset_name}"
#             )
#             print(f"Memory used: {get_used_memory()}")
