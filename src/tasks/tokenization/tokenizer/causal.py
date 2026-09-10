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
    
    def tokenize(self, dataset: Union[HFDataset, DatasetDict]) -> Union[HFDataset, DatasetDict]:
        """
        Tokenize the input dataset for causal language modeling.

        This method applies tokenization to either a single dataset (HFDataset) or a dictionary 
        of datasets (DatasetDict). For DatasetDict inputs, each split is tokenized separately. 
        The tokenization process includes text truncation, padding, and the generation of 
        attention masks and labels.

        Args:
            dataset (Union[HFDataset, DatasetDict]): The dataset(s) to tokenize.

        Returns:
            Union[HFDataset, DatasetDict]: The tokenized dataset(s). In the special case
                                           of a DatasetDict with only one split, the single tokenized dataset                                           is returned directly.        
        """
        # Initialize tokenizer if not already done
        if self._tokenizer is None:
            try:
                self._initialize_tokenizer()
            except Exception as e:
                self.logger.error(f"Failed to initialize tokenizer: {e}")
                raise RuntimeError(f"Tokenizer initialization failed: {e}") from e
        
        # Validate tokenizer is properly initialized
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is None after initialization. Check tokenizer_name in config.")
        
        start_time = time.time()
        self.logger.info("=== Starting Causal LM Tokenization ===")
        
        try:
            # Log configuration details
            self.logger.info(f"Tokenizer configuration:")
            self.logger.info(f"  - Context length: {self.config.context_length}")
            self.logger.info(f"  - Overlap: {self.config.overlap}")
            self.logger.info(f"  - Batch size: {getattr(self.config, 'batch_size', 1000)}")
            
            # Get optimal number of processes
            num_proc = self._get_optimal_num_proc()
            
            # Performance optimization parameters
            batch_size = getattr(self.config, 'batch_size', 1000)
            writer_batch_size = getattr(self.config, 'writer_batch_size', 40000)
            
            self.logger.info(f"Processing parameters:")
            self.logger.info(f"  - num_proc: {num_proc}")
            self.logger.info(f"  - batch_size: {batch_size}")
            self.logger.info(f"  - writer_batch_size: {writer_batch_size}")
            
            if isinstance(dataset, DatasetDict):
                self.logger.info(f"Processing DatasetDict with {len(dataset)} splits:")
                for split_name, split_data in dataset.items():
                    self.logger.info(f"  {split_name}: {len(split_data)} examples")
                
                result = DatasetDict()
                total_examples = 0
                
                for split_name, split_dataset in dataset.items():
                    split_start_time = time.time()
                    self.logger.info(f"Processing split: {split_name} ({len(split_dataset)} examples)")
                    
                    # Configure mapping parameters
                    map_kwargs = {
                        "function": self._tokenize_function,
                        "batched": True,
                        "features": self._features,
                        "remove_columns": split_dataset.column_names,
                        "batch_size": batch_size,
                        "writer_batch_size": writer_batch_size,
                    }
                    
                    # Only add num_proc for slow tokenizers
                    if num_proc is not None:
                        map_kwargs["num_proc"] = num_proc
                    
                    # Add progress description if enabled
                    if hasattr(self.config, 'show_progress') and self.config.show_progress:
                        map_kwargs["desc"] = f"Tokenizing {split_name} split"
                    
                    try:
                        result[split_name] = split_dataset.map(**map_kwargs)
                        
                        split_time = time.time() - split_start_time
                        split_throughput = len(split_dataset) / split_time if split_time > 0 else 0
                        total_examples += len(split_dataset)
                        
                        self.logger.info(f"Completed {split_name} in {split_time:.2f} seconds")
                        self.logger.info(f"Split throughput: {split_throughput:.1f} examples/sec")
                        self.logger.info(f"Final split size: {len(result[split_name])}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing split {split_name}: {str(e)}")
                        raise
                
                self.logger.info(f"Processed all splits: {total_examples} total examples")
                
                # Return single dataset if only one split
                if len(result) == 1:
                    self.logger.debug("Only one dataset split found, returning single tokenized dataset")
                    return result[list(result.keys())[0]]
                    
            elif isinstance(dataset, HFDataset):
                self.logger.info(f"Processing single dataset with {len(dataset)} examples")
                
                # Configure mapping parameters
                map_kwargs = {
                    "function": self._tokenize_function,
                    "batched": True,
                    "features": self._features,
                    "remove_columns": dataset.column_names,
                    "batch_size": batch_size,
                    "writer_batch_size": writer_batch_size,
                }
                
                # Only add num_proc for slow tokenizers
                if num_proc is not None:
                    map_kwargs["num_proc"] = num_proc
                
                # Add progress description if enabled
                if hasattr(self.config, 'show_progress') and self.config.show_progress:
                    map_kwargs["desc"] = "Tokenizing causal dataset"
                
                try:
                    result = dataset.map(**map_kwargs)
                    self.logger.info(f"Tokenization completed successfully")
                    self.logger.info(f"Final dataset size: {len(result)}")
                    
                except Exception as e:
                    self.logger.error(f"Error during tokenization: {str(e)}")
                    raise
                    
            else:
                raise ValueError(f"Unsupported dataset type: {type(dataset)}. Supported types: HFDataset, DatasetDict")
            
            # Final performance summary
            elapsed_time = time.time() - start_time
            
            # Calculate total examples processed
            if isinstance(result, HFDataset):
                total_processed = len(result)
            elif isinstance(result, DatasetDict):
                total_processed = sum(len(split) for split in result.values())
            else:
                total_processed = 0
            
            throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info("=== Causal LM Tokenization Completed Successfully ===")
            self.logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Total examples processed: {total_processed:,}")
            self.logger.info(f"Overall throughput: {throughput:.1f} examples/sec")
            self.logger.info(f"Average time per example: {(elapsed_time/total_processed)*1000:.2f} ms")
            
            # Memory usage info if available
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"Memory usage: {memory_mb:.1f} MB")
            except ImportError:
                pass
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"=== Causal LM Tokenization Failed ===")
            self.logger.error(f"Error after {elapsed_time:.2f} seconds: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise

    def _tokenize_function(self, batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Batch-tokenize input texts with overflow/stride and build input IDs, attention masks, and labels.
        Optimized for performance with efficient label generation.
        """
        # Ensure tokenizer is initialized (important for multiprocessing)
        if self._tokenizer is None:
            try:
                self._initialize_tokenizer()
            except Exception as e:
                self.logger.error(f"Failed to initialize tokenizer: {e}")
                raise RuntimeError(f"Tokenizer initialization failed: {e}") from e
        
        # Validate tokenizer is properly initialized
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is None after initialization. Check tokenizer_name in config.")
        
        # Run tokenizer on the entire batch at once
        try:
            outputs = self._tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.context_length,
                stride=self.config.overlap,
                return_overflowing_tokens=True,
                padding="max_length",  # Pad to max_length for consistency
                return_tensors="np",  # Direct NumPy conversion
            )
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            self.logger.error(f"Batch content preview: {str(batch)[:200]}...")
            raise RuntimeError(f"Tokenization failed: {e}") from e
        
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

