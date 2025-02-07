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