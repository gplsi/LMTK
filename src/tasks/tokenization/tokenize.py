from tokenized_format import *
from typing import Optional
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from datasets import load_dataset, load_from_disk, Features, Sequence, Value, Dataset
import os
from tokenization_utils import *

"""
This script contains functions for tokenizing text data for language modeling tasks.
"""


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


class Tokenize(BaseTokenizer):
    
    """
    This is the main class for tokenizing, it will perform specific tokenization based on the desired task.
    Currently supports values in ['continual_pretraining']. kwargs expects the specific parameters for the 
    tokenization task.
    """
    def __init__(
        self,
        task: str,
        **kwargs,
    ):
        self.task = task        # ['causal_pretraining']
        self.kwargs = kwargs
        
        if self.task == 'causal_pretraining':
            self.context_length = self.kwargs['context_length']
            self.overlap = kwargs['overlap']
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        super().__init__(**kwargs)
        
    def tokenize_dataset(self, dataset: Dataset):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define the dataset features
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
        })
        
        # Dealing with the tokenized file name if loaded from Hub or disk
        if self.loaded_from_hub:
            tokenized_dataset_name = self.dataset_name.replace('/', '-') + "_tokenized"
            os.makedirs(f"data/tokenized/{tokenized_dataset_name}", exist_ok=True) 
        elif self.loaded_from_disk:
            tokenized_dataset_name = self.path.split('/')[-1] + "_tokenized"
            os.makedirs(f"data/tokenized/{tokenized_dataset_name}", exist_ok=True)
        else:
            raise ValueError("Either dataset name or path must be provided.") 
        
        # Tokenize the dataset
        tokenized_datasets = dataset.map(
            lambda batch: TOKENIZE_FUNCTION_HANDLER[self.task](tokenizer, batch, context_length=self.context_length, overlap=self.overlap),
            batched=True, features=features)
        
        tokenized_datasets.save_to_disk(dataset_path=f"data/tokenized/{tokenized_dataset_name}")
        return tokenized_dataset_name