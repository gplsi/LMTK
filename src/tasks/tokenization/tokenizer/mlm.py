# src/tokenization/mlm.py
from typing import Dict, List, Optional, Union
from datasets import Features, Sequence, Value, DatasetDict
from datasets import Dataset as HFDataset
import os
import time
import multiprocessing
import random
from tqdm.auto import tqdm
from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.utils import build_masked_lm_outputs
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
import numpy as np
from utils import get_optimal_thread_count
from utils.logging import get_logger

class MaskedLMTokenizer(BaseTokenizer):
    """
    Tokenizer implementation for masked language modeling tasks.

    This class extends BaseTokenizer to provide tokenization methods specifically designed
    for masked language modeling (MLM). It supports both single HFDataset instances and multiple splits
    encapsulated in a DatasetDict. The tokenization process generates sequences for input IDs,
    attention masks, and labels, with random masking applied according to MLM strategies.
    
    The masking follows the BERT-style approach:
    - Select tokens for masking based on mlm_probability
    - Of selected tokens: 80% become [MASK], 10% become random tokens, 10% stay unchanged
    - Labels contain original tokens for masked positions, -100 for unmasked positions
    """
    
    def __init__(self, config: TokenizerConfig) -> None:
        """
        Initialize the MaskedLMTokenizer instance with the given configuration.

        This constructor sets up the tokenizer by invoking the base constructor and defines
        the expected output structure (features) of tokenization using Hugging Face Datasets.
        It also extracts MLM-specific parameters from the configuration.

        Args:
            config (TokenizerConfig): Configuration parameters for the tokenizer, including
                                      context length, MLM probability, masking strategy, etc.
        """
        super().__init__(config)
        
        # Define features with fixed-length sequences for better performance
        # Using int32 for input_ids and labels to handle large vocabularies
        self._features = Features({
            "input_ids": Sequence(Value("int32"), length=config.context_length),
            "attention_mask": Sequence(Value("int32"), length=config.context_length),
            "labels": Sequence(Value("int32"), length=config.context_length)
        })
        
        # MLM-specific configuration parameters
        self.mlm_probability = getattr(config, 'mlm_probability', 0.15)
        self.mask_token_id = getattr(config, 'mask_token_id', 103)  # Default BERT [MASK] token
        self.mask_special_tokens = getattr(config, 'mask_special_tokens', False)
        self.exclude_token_ids = set(getattr(config, 'exclude_token_ids', []))
        
        # Masking strategy probabilities
        masking_strategy = getattr(config, 'masking_strategy', {})
        self.mask_token_prob = masking_strategy.get('mask_token_prob', 0.8)
        self.random_token_prob = masking_strategy.get('random_token_prob', 0.1)
        self.unchanged_prob = masking_strategy.get('unchanged_prob', 0.1)
        
        # Validate probabilities sum to 1.0
        total_prob = self.mask_token_prob + self.random_token_prob + self.unchanged_prob
        if abs(total_prob - 1.0) > 1e-6:
            self.logger.warning(f"Masking strategy probabilities sum to {total_prob:.3f}, not 1.0. Normalizing.")
            self.mask_token_prob /= total_prob
            self.random_token_prob /= total_prob
            self.unchanged_prob /= total_prob
        
    def _get_optimal_num_proc(self) -> Optional[int]:
        """
        Get the optimal number of processes for multiprocessing based on tokenizer type and system capabilities.
        
        Fast tokenizers (Rust-based) use their own internal parallelism and should not use Python multiprocessing.
        Slow tokenizers (Python-based) benefit from Python multiprocessing.
        
        Returns:
            Optional[int]: Number of processes to use for slow tokenizers, None for fast tokenizers
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
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
    
    def _should_mask_token(self, token_id: int) -> bool:
        """
        Determine if a token should be considered for masking.
        
        Args:
            token_id (int): The token ID to check.
            
        Returns:
            bool: True if the token can be masked, False otherwise.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        # Don't mask tokens in the exclude list
        if token_id in self.exclude_token_ids:
            return False
            
        # Handle special tokens
        if not self.mask_special_tokens and self._tokenizer is not None:
            # Check if it's a special token
            special_tokens = set()
            if hasattr(self._tokenizer, 'all_special_ids') and self._tokenizer.all_special_ids:
                special_tokens.update(self._tokenizer.all_special_ids)
            if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                special_tokens.add(self._tokenizer.pad_token_id)
            if hasattr(self._tokenizer, 'cls_token_id') and self._tokenizer.cls_token_id is not None:
                special_tokens.add(self._tokenizer.cls_token_id)
            if hasattr(self._tokenizer, 'sep_token_id') and self._tokenizer.sep_token_id is not None:
                special_tokens.add(self._tokenizer.sep_token_id)
                
            if token_id in special_tokens:
                return False
                
        return True
    
    def _apply_mlm_masking(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> tuple:
        """
        Apply MLM masking to input sequences using vectorized operations for efficiency.
        
        Args:
            input_ids (np.ndarray): Input token IDs of shape (batch_size, seq_len).
            attention_mask (np.ndarray): Attention mask of shape (batch_size, seq_len).
            
        Returns:
            tuple: (masked_input_ids, labels) where labels contain original tokens for masked positions.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        batch_size, seq_len = input_ids.shape
        masked_input_ids = input_ids.copy()
        labels = np.full_like(input_ids, -100)  # -100 is ignored in loss computation
        
        # Get vocabulary size for random token replacement
        vocab_size = len(self._tokenizer.get_vocab()) if self._tokenizer and hasattr(self._tokenizer, 'get_vocab') else 30522
        
        # Create mask for tokens that can be masked (optimized)
        # Instead of checking every token ID in vocab, check only unique tokens in the batch
        unique_tokens = np.unique(input_ids[attention_mask == 1])  # Only check non-padded tokens
        maskable_tokens = np.zeros_like(input_ids, dtype=bool)
        
        # Vectorized check for maskable tokens (only unique tokens in batch)
        for token_id in unique_tokens:
            if self._should_mask_token(int(token_id)):
                maskable_tokens |= (input_ids == token_id)
        
        # Only consider non-padded tokens (attention_mask == 1)
        candidate_mask = maskable_tokens & (attention_mask == 1)
        
        # Generate random probabilities for all positions at once
        random_probs = np.random.random(input_ids.shape)
        
        # Select tokens to mask based on mlm_probability
        tokens_to_mask = candidate_mask & (random_probs < self.mlm_probability)
        
        # For tokens that will be masked, store original values in labels
        labels[tokens_to_mask] = input_ids[tokens_to_mask]
        
        # Generate masking strategy probabilities for masked tokens
        mask_strategy_probs = np.random.random(input_ids.shape)
        
        # Apply different masking strategies vectorized
        # Strategy 1: Replace with [MASK] token (80%)
        mask_with_token = tokens_to_mask & (mask_strategy_probs < self.mask_token_prob)
        masked_input_ids[mask_with_token] = self.mask_token_id
        
        # Strategy 2: Replace with random token (10%)
        replace_with_random = tokens_to_mask & (
            mask_strategy_probs >= self.mask_token_prob
        ) & (
            mask_strategy_probs < self.mask_token_prob + self.random_token_prob
        )
        
        if np.any(replace_with_random):
            # Generate random token IDs for positions that need random replacement
            random_tokens = np.random.randint(0, vocab_size, size=input_ids.shape)
            masked_input_ids[replace_with_random] = random_tokens[replace_with_random]
        
        # Strategy 3: Keep unchanged (10%) - no action needed
        
        return masked_input_ids, labels
    
    def tokenize(self, dataset: Union[HFDataset, DatasetDict]) -> Union[HFDataset, DatasetDict]:
        """
        Tokenize the input dataset for masked language modeling.

        This method applies tokenization to either a single dataset (HFDataset) or a dictionary 
        of datasets (DatasetDict) containing multiple splits. It utilizes the Hugging Face's map 
        function to efficiently process data in batches with MLM-specific masking applied.
        
        Args:
            dataset (Union[HFDataset, DatasetDict]): The dataset or dataset dictionary to be tokenized.
            
        Returns:
            Union[HFDataset, DatasetDict]: A tokenized dataset if a single HFDataset is provided, or
                                           a DatasetDict if multiple splits were tokenized. In the case
                                           of a DatasetDict with only one split, the single tokenized dataset
                                           is returned directly.
        """
        # Initialize tokenizer if not already done
        if self._tokenizer is None:
            self._initialize_tokenizer()
        
        start_time = time.time()
        self.logger.info("=== Starting MLM Tokenization ===")
        
        try:
            # Log configuration details
            self.logger.info(f"MLM Tokenizer configuration:")
            self.logger.info(f"  - Context length: {self.config.context_length}")
            self.logger.info(f"  - MLM probability: {self.mlm_probability}")
            self.logger.info(f"  - Mask token probability: {self.mask_token_prob}")
            self.logger.info(f"  - Random token probability: {self.random_token_prob}")
            self.logger.info(f"  - Unchanged probability: {self.unchanged_prob}")
            self.logger.info(f"  - Batch size: {getattr(self.config, 'batch_size', 1000)}")
            
            # Get optimal number of processes
            num_proc = self._get_optimal_num_proc()
            
            # Performance optimization parameters
            batch_size = getattr(self.config, 'batch_size', 1000)
            writer_batch_size = getattr(self.config, 'writer_batch_size', 10000)
            
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
                    "remove_columns": split_dataset.column_names,
                    "batch_size": batch_size,
                    "writer_batch_size": writer_batch_size,
                }
                
                # Only add num_proc for slow tokenizers
                if num_proc is not None:
                    map_kwargs["num_proc"] = num_proc
                
                # Add progress description if enabled
                if hasattr(self.config, 'show_progress') and self.config.show_progress:
                    map_kwargs["desc"] = f"Tokenizing {split_name} split with MLM"
                
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
                "remove_columns": dataset.column_names,
                "batch_size": batch_size,
                "writer_batch_size": writer_batch_size,
            }
            
            # Only add num_proc for slow tokenizers
            if num_proc is not None:
                map_kwargs["num_proc"] = num_proc
            
            # Add progress description if enabled
            if hasattr(self.config, 'show_progress') and self.config.show_progress:
                map_kwargs["desc"] = "Tokenizing dataset with MLM"
            
            try:
                result = dataset.map(**map_kwargs)
                self.logger.info(f"MLM tokenization completed successfully")
                self.logger.info(f"Final dataset size: {len(result)}")
                
            except Exception as e:
                self.logger.error(f"Error during MLM tokenization: {str(e)}")
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
        
        self.logger.info("=== MLM Tokenization Completed Successfully ===")
        self.logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Total examples processed: {total_processed:,}")
        self.logger.info(f"Overall throughput: {throughput:.1f} examples/sec")
        self.logger.info(f"Average time per example: {(elapsed_time/total_processed)*1000:.2f} ms")
        
        # MLM-specific statistics
        expected_masked_tokens = total_processed * self.config.context_length * self.mlm_probability
        self.logger.info(f"Expected masked tokens: ~{expected_masked_tokens:,.0f} ({self.mlm_probability*100:.1f}% of all tokens)")
        
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
            self.logger.error(f"=== MLM Tokenization Failed ===")
            self.logger.error(f"Error after {elapsed_time:.2f} seconds: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise



