# src/tokenization/causal.py
from typing import Dict, List, Optional, Union
from datasets import  Features, Sequence, Value, DatasetDict
from datasets import Dataset as HFDataset
import os
import time
import multiprocessing
from tqdm.auto import tqdm
from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.utils import build_causal_lm_outputs
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
import numpy as np
from utils import get_optimal_thread_count
from utils.logging import get_logger

class CausalLMTokenizer(BaseTokenizer):
    """
    Tokenizer implementation for casual language modeling tasks.

    This class extends BaseTokenizer to provide tokenization methods specifically designed
    for causal language modeling. It supports both single HFDataset instances and multiple splits
    encapsulated in a DatasetDict. The tokenization process generates sequences for input IDs,
    attention masks, and labels, with the labels mirroring the input IDs for training purposes.
    """
    def __init__(self, config: TokenizerConfig) -> None:
        """
        Initialize the CausalLMTokenizer instance with the given configuration.

        This constructor sets up the tokenizer by invoking the base constructor and defines
        the expected output structure (features) of tokenization using Hugging Face Datasets.

        Args:
            config (TokenizerConfig): Configuration parameters for the tokenizer, including
                                      context length and overlap settings.
        """
        super().__init__(config)
        # Define features with fixed-length sequences for better performance
        # Using int32 for input_ids and labels to handle large vocabularies
        self._features = Features({
            "input_ids": Sequence(Value("int32"), length=config.context_length),
            "attention_mask": Sequence(Value("int32"), length=config.context_length),
            "labels": Sequence(Value("int32"), length=config.context_length)
        })
        
    def _get_optimal_num_proc(self) -> Optional[int]:
        """
        Get the optimal number of processes for multiprocessing based on tokenizer type and system capabilities.
        
        Fast tokenizers (Rust-based) use their own internal parallelism and should not use Python multiprocessing.
        Slow tokenizers (Python-based) benefit from Python multiprocessing.
        
        Returns:
            Optional[int]: Number of processes to use for slow tokenizers, None for fast tokenizers
        """
        # Check if we have a fast tokenizer
        is_fast_tokenizer = hasattr(self._tokenizer, 'is_fast') and self._tokenizer.is_fast
        
        if is_fast_tokenizer:
            # Fast tokenizers handle parallelism internally via Rust threads
            self.logger.debug("Fast tokenizer detected - using internal Rust parallelism (num_proc=None)")
            return None
        else:
            # Slow tokenizers benefit from Python multiprocessing
            if self.config.num_proc is not None and self.config.num_proc > 1:
                # User specified num_proc, respect their choice but cap at available cores
                max_cores = multiprocessing.cpu_count()
                optimal = min(self.config.num_proc, max_cores)
                if optimal != self.config.num_proc:
                    self.logger.warning(f"Requested num_proc={self.config.num_proc} exceeds available cores ({max_cores}). Using {optimal}")
                return optimal
            elif self.config.num_proc == 1 or self.config.num_proc is None:
                # Default case: use half of available cores (leave some for system)
                max_cores = multiprocessing.cpu_count()
                optimal = max(1, max_cores // 2)  # At least 1, at most half of cores
                self.logger.debug(f"Slow tokenizer detected - using Python multiprocessing with {optimal} processes")
                return optimal
            else:
                # Fallback to single process
                return 1
    
    def tokenize(self, dataset: Union[HFDataset, DatasetDict]) -> HFDataset:
        """
        Tokenize the input dataset for causal language modeling.

        This method applies tokenization to either a single dataset (HFDataset) or a dictionary 
        of datasets (DatasetDict) containing multiple splits. It utilizes the Hugging Face's map 
        function to efficiently process data in batches. Post tokenization, it removes unnecessary 
        columns from the original dataset.        Args:
            dataset (Union[HFDataset, DatasetDict]): The dataset or dataset dictionary to be tokenized.
            
        Returns:
            Union[HFDataset, DatasetDict]: A tokenized dataset if a single HFDataset is provided, or
                                           a DatasetDict if multiple splits were tokenized. In the case
                                           of a DatasetDict with only one split, the single tokenized dataset                                           is returned directly.        """
        self.logger.info("Initializing tokenizer")
        self._initialize_tokenizer()
          # Get optimal multiprocessing configuration
        num_proc = self._get_optimal_num_proc()
        batch_size = self.config.batch_size or 2000
        
        # Log performance configuration with parallelism strategy
        if num_proc is None:
            self.logger.info(f"Performance configuration: batch_size={batch_size}, parallelism=fast_tokenizer_internal")
        else:
            self.logger.info(f"Performance configuration: batch_size={batch_size}, num_proc={num_proc}, parallelism=python_multiprocessing")
        
        # Enable fast tokenizer detection for performance monitoring
        if isinstance(dataset, DatasetDict):
            self.logger.info("Detected 'DatasetDict' instance")
            # Handle DatasetDict case
            tokenized_datasets = DatasetDict()
            for split, split_dataset in dataset.items():
                self.logger.info(f"Tokenizing split '{split}' with {len(split_dataset)} examples")
                start_time = time.time()                # Configure mapping parameters based on config
                map_kwargs = {
                    "function": self._tokenize_function,
                    "batched": True,
                    "features": self._features,
                    "keep_in_memory": False,
                    "remove_columns": split_dataset.column_names,
                    "batch_size": batch_size,
                    "writer_batch_size": 40000,  # Write in larger chunks for better I/O performance
                }
                
                # Only add num_proc for slow tokenizers (fast tokenizers use internal Rust parallelism)
                if num_proc is not None:
                    map_kwargs["num_proc"] = num_proc
                
                # Add progress description if progress monitoring is enabled
                if hasattr(self.config, 'show_progress') and self.config.show_progress:
                    map_kwargs["desc"] = f"Tokenizing {split} split"

                tokenized_datasets[split] = split_dataset.map(**map_kwargs)                
                elapsed_time = time.time() - start_time
                throughput = len(split_dataset) / elapsed_time if elapsed_time > 0 else 0
                self.logger.info(f"Completed tokenizing split '{split}' in {elapsed_time:.2f} seconds")
                self.logger.info(f"Processed {len(split_dataset)} examples at {throughput:.1f} examples/sec")
            
            if len(tokenized_datasets) == 1:
                self.logger.debug("Only one dataset split found, returning single tokenized dataset")
                return tokenized_datasets[list(tokenized_datasets.keys())[0]]
            self.logger.debug("Multiple dataset splits found, returning tokenized 'DatasetDict'")
            return tokenized_datasets
        else:
            self.logger.info("Detected regular 'Dataset' instance")
            self.logger.info(f"Tokenizing dataset with {len(dataset)} examples")
            start_time = time.time()            # Configure mapping parameters based on config
            map_kwargs = {
                "function": self._tokenize_function,
                "batched": True,
                "features": self._features,
                "remove_columns": dataset.column_names,
                "batch_size": batch_size,
                "writer_batch_size": 40000,  # Write in larger chunks for better I/O performance
            }
            
            # Only add num_proc for slow tokenizers (fast tokenizers use internal Rust parallelism)
            if num_proc is not None:
                map_kwargs["num_proc"] = num_proc
            
            # Add progress description if progress monitoring is enabled
            if hasattr(self.config, 'show_progress') and self.config.show_progress:
                map_kwargs["desc"] = "Tokenizing dataset"
            
            # Handle single Dataset case
            tokenized_dataset = dataset.map(**map_kwargs)
            
            elapsed_time = time.time() - start_time
            throughput = len(dataset) / elapsed_time if elapsed_time > 0 else 0
            self.logger.info(f"Completed tokenization in {elapsed_time:.2f} seconds")
            self.logger.info(f"Processed {len(dataset)} examples at {throughput:.1f} examples/sec")
            self.logger.debug("Returning tokenized dataset")
            return tokenized_dataset

    def _tokenize_function(self, batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Batch-tokenize input texts with overflow/stride and build input IDs, attention masks, and labels.
        Optimized for performance with efficient label generation.
        """
        # Run tokenizer on the entire batch at once
        outputs = self._tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.context_length,
            stride=self.config.overlap,
            return_overflowing_tokens=True,
            padding="max_length",  # Pad to max_length for consistency
            return_tensors="np",  # Direct NumPy conversion
        )
        
        input_ids = outputs["input_ids"]       # Already NumPy arrays
        attention_mask = outputs["attention_mask"]
        
        # Efficient vectorized label creation
        labels = np.where(attention_mask == 1, input_ids, -100)
        
        # Single conversion at the end
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

