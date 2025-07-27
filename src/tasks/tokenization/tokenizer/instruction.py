# src/tokenization/instruction.py
from typing import Dict, List, Optional, Union, Tuple
from datasets import Features, Sequence, Value, DatasetDict
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
import os
import json
import time
import multiprocessing
import pandas as pd
import torch
from tqdm.auto import tqdm
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.utils import build_causal_lm_outputs
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
import numpy as np
from utils import get_optimal_thread_count
from utils.logging import get_logger

class InstructionTokenizer(BaseTokenizer):
    """
    Tokenizer implementation for instruction tuning tasks.

    This class extends BaseTokenizer to provide tokenization methods specifically designed
    for instruction tuning with multi-turn conversations. It supports chat template formatting,
    conversation preprocessing, and proper label masking for instruction following tasks.
    The tokenization process generates sequences for input IDs, attention masks, and labels,
    with the labels properly masked to focus training on assistant responses.
    """
    
    def __init__(self, config: TokenizerConfig) -> None:
        """
        Initialize the InstructionTokenizer instance with the given configuration.

        This constructor sets up the tokenizer by invoking the base constructor and defines
        the expected output structure (features) of tokenization using Hugging Face Datasets.
        It also sets up instruction-specific configuration parameters.

        Args:
            config (TokenizerConfig): Configuration parameters for the tokenizer, including
                                      context length, masking settings, and other instruction-specific options.
        """
        super().__init__(config)
        
        # Define features with fixed-length sequences for better performance
        # Using int32 for input_ids and labels to handle large vocabularies
        self._features = Features({
            "input_ids": Sequence(Value("int32"), length=config.context_length),
            "attention_mask": Sequence(Value("int32"), length=config.context_length),
            "labels": Sequence(Value("int32"), length=config.context_length)
        })
        
        # Instruction-specific configuration
        self.mask_prompt = getattr(config, 'mask_prompt', True)
        self.ignore_index = getattr(config, 'ignore_index', -100)
        self.max_seq_length = getattr(config, 'max_seq_length', config.context_length)
        self.test_size = getattr(config, 'test_size', 0.3)
        self.seed = getattr(config, 'seed', 1234)
        
    def _get_optimal_num_proc(self) -> Optional[int]:
        """
        Get the optimal number of processes for multiprocessing based on tokenizer type and system capabilities.
        
        Fast tokenizers (Rust-based) use their own internal parallelism and should not use Python multiprocessing.
        Slow tokenizers (Python-based) benefit from Python multiprocessing.
        
        Returns:
            Optional[int]: Number of processes to use, or None to disable multiprocessing.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        # Fast tokenizers handle parallelism internally
        if self._tokenizer.is_fast:
            return None
        
        # For slow tokenizers, use multiprocessing with optimal thread count
        return get_optimal_thread_count()
    
    def _create_conversation(self, row: Dict, include_target: bool = True) -> tuple[List[Dict[str, str]], str]:
        """
        Create a conversation structure from a data row, handling system messages properly.
        
        Args:
            row (Dict): Data row containing 'system', 'input', 'assistance', and 'target' fields.
            include_target (bool): Whether to include the target response in the conversation.
                                 Set to False for prompt generation, True for complete conversation.
            
        Returns:
            tuple[List[Dict[str, str]], str]: Conversation with role-content pairs and system message.
        """
        conversation = []
        system_message = ""
        
        # Extract system message separately to handle tokenizer alternation requirements
        if row.get('system') and row['system'].strip():
            system_message = row['system'].strip()

        # Process conversation ensuring strict user/assistant alternation
        if row.get('assistance') and len(row['assistance']) > 0:
            # Multi-turn conversation - ensure proper alternation
            if 'input' not in row:
                raise ValueError(f"Missing 'input' field in row: {list(row.keys())}")
            inputs = row['input'] if isinstance(row['input'], list) else [row['input']]
            assistances = row['assistance']
            
            # Build conversation with strict alternation
            for i in range(max(len(inputs), len(assistances))):
                # Add user message if available
                if i < len(inputs):
                    conversation.append({'role': 'user', 'content': inputs[i]})
                
                # Add assistant message if available
                if i < len(assistances):
                    conversation.append({'role': 'assistant', 'content': assistances[i]})
        else:
            # Single-turn conversation (most common case)
            # Add the user input
            if 'input' not in row:
                raise ValueError(f"Missing 'input' field in row: {list(row.keys())}")
            
            user_input = row['input']
            if isinstance(user_input, list):
                if len(user_input) > 0:
                    user_input = user_input[0]
                else:
                    raise ValueError("'input' field is an empty list")
            elif user_input is None:
                raise ValueError("'input' field is None")
            
            conversation.append({'role': 'user', 'content': str(user_input)})
        
        # Add the target response as the final assistant message (only if include_target is True)
        if include_target and 'target' in row and row['target']:
            target_content = str(row['target']).strip() if row['target'] is not None else ""
            if target_content:  # Only add non-empty targets
                if conversation and conversation[-1]['role'] == 'assistant':
                    # Replace the last assistant message with the target
                    conversation[-1]['content'] = target_content
                else:
                    # Add target as new assistant message
                    conversation.append({'role': 'assistant', 'content': target_content})
    
        # Validate conversation alternation and fix any issues
        # Process from the end to avoid index issues when removing items
        i = len(conversation) - 1
        while i > 0:
            current_role = conversation[i]['role']
            prev_role = conversation[i - 1]['role']
            
            # Check for consecutive messages with same role
            if current_role == prev_role:
                if current_role == 'user':
                    # Two consecutive user messages - merge them
                    conversation[i - 1]['content'] += "\n\n" + conversation[i]['content']
                    conversation.pop(i)
                elif current_role == 'assistant':
                    # Two consecutive assistant messages - keep only the last one (target)
                    # This prevents duplication by removing the earlier assistant message
                    conversation.pop(i - 1)
            i -= 1
        
        # Ensure conversation starts with user and ends appropriately based on include_target
        if conversation:
            # Must start with user
            if conversation[0]['role'] != 'user':
                # This shouldn't happen with our logic, but just in case
                conversation.insert(0, {'role': 'user', 'content': 'Please help me with the following:'})
            
            # For complete conversations (include_target=True), must end with assistant
            # For prompt-only conversations (include_target=False), should end with user for generation
            if include_target and conversation[-1]['role'] != 'assistant':
                # Add a placeholder assistant response if missing for complete conversations
                conversation.append({'role': 'assistant', 'content': 'I understand. How can I help you?'})
        
        return conversation, system_message
    
    def _create_prompt(self, conversation: List[Dict[str, str]], system_message: str = "", add_generation_prompt: bool = True) -> str:
        """
        Create a formatted prompt from conversation using chat template.
        
        Args:
            conversation (List[Dict[str, str]]): Conversation with role-content pairs.
            system_message (str): Optional system message to include.
            add_generation_prompt (bool): Whether to add generation prompt (empty assistant token).
                                        Set to True for prompt generation, False for complete conversations.
            
        Returns:
            str: Formatted prompt string.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized. Cannot apply chat template.")
            
        date_string = datetime.today().strftime('%Y-%m-%d')
        
        # Try to use system_message parameter if supported, otherwise integrate into conversation
        try:
            prompt = self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                date_string=date_string,
                system_message=system_message if system_message else None
            )
        except TypeError:
            # Fallback: integrate system message into first user message if system_message param not supported
            if system_message and conversation and conversation[0]['role'] == 'user':
                original_content = conversation[0]['content']
                conversation[0]['content'] = f"{system_message}\n\n{original_content}"
            
            prompt = self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                date_string=date_string
            )
        
        return prompt
    
    def _tokenize_instruction_example(self, element: Dict) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single instruction example with proper label masking.
        
        Args:
            element (Dict): Dictionary containing structured conversation data with 'system', 'input', 'assistance' fields.
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized example with input_ids, attention_mask, and labels.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        # Create conversation and format with chat template
        conversation = element['prompt']
        
        # Encode instruction part
        encoded_instruction = self._tokenizer.encode(
            conversation,
            max_length=self.max_seq_length,
            add_special_tokens=False
        )

        # Create full conversation (with assistant response)
        full_conversation = element['prompt_and_response']
        
        # Encode full conversation (instruction + response)
        encoded_full = self._tokenizer.encode(
            full_conversation, 
            max_length=self.max_seq_length - 1,  # Reserve space for EOS
            return_tensors="pt", 
            add_special_tokens=False
        ).squeeze_()

        # Add EOS token
        encoded_prompt_and_response = torch.cat((
            encoded_full, 
            torch.tensor([self._tokenizer.eos_token_id])
        ))
        
        # Create labels and attention mask
        labels = encoded_prompt_and_response.clone()
        attention_mask = torch.ones_like(encoded_prompt_and_response)
        
        # Apply masking strategy based on configuration
        masking_strategy = getattr(self.config, 'masking_strategy', 'context_aware')
        
        if self.mask_prompt:
            # Calculate instruction length for proper masking
            instruction_length = len(encoded_instruction)
            
            if masking_strategy == 'context_aware':
                # Context-aware masking: Full attention, mask only instruction labels
                # Model sees full context but only learns from responses
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
                # attention_mask remains all 1s for real tokens
                
            elif masking_strategy == 'response_only':
                # Response-only masking: Mask both attention and labels
                # Model only attends to and learns from responses
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
                    attention_mask[:instruction_length] = 0
                
            else:
                # Default to context_aware for unknown strategies
                self.logger.warning(f"Unknown masking_strategy '{masking_strategy}', defaulting to 'context_aware'")
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
            
        # Apply padding strategy based on configuration
        padding_strategy = getattr(self.config, 'padding_strategy', 'fixed')
        
        if padding_strategy == 'dynamic':
            # Dynamic padding: return sequences as-is, padding will be handled at batch level
            # Store original length for batch-level padding
            return {
                "input_ids": encoded_prompt_and_response,
                "attention_mask": attention_mask,
                "labels": labels,
                "length": len(encoded_prompt_and_response)  # Store length for dynamic padding
            }
        else:
            # Fixed padding: pad to context length (current behavior)
            seq_len = len(encoded_prompt_and_response)
            if seq_len < self.config.context_length:
                pad_length = self.config.context_length - seq_len
                
                # Pad with pad_token_id (or 0 if not available)
                pad_token_id = getattr(self._tokenizer, 'pad_token_id', 0) or 0
                
                encoded_prompt_and_response = torch.cat([
                    encoded_prompt_and_response,
                    torch.full((pad_length,), pad_token_id, dtype=torch.long)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), self.ignore_index, dtype=torch.long)
                ])
                
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=torch.long)
                ])
            elif seq_len > self.config.context_length:
                # Truncate if too long
                encoded_prompt_and_response = encoded_prompt_and_response[:self.config.context_length]
                labels = labels[:self.config.context_length]
                attention_mask = attention_mask[:self.config.context_length]
        
        return {
            "input_ids": encoded_prompt_and_response,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _tokenize_instruction_example_original(self, element: Dict) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single instruction example with proper label masking.
        
        Args:
            element (Dict): Dictionary containing structured conversation data with 'system', 'input', 'assistance' fields.
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized example with input_ids, attention_mask, and labels.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        # Create conversation and format with chat template
        conversation, system_message = self._create_conversation(element)
        
        # Create full conversation (with assistant response)
        full_conversation = self._create_prompt(conversation, system_message)
        
        # For proper masking, we need to identify where the response starts
        # Instead of creating instruction-only conversation (which breaks alternation),
        # we'll create a conversation that ends just before the final assistant response
        if len(conversation) > 1 and conversation[-1]['role'] == 'assistant':
            # Create conversation without the final assistant response
            instruction_conversation = conversation[:-1]
            
            # Add generation prompt to get the instruction length
            instruction_prompt = self._create_prompt(instruction_conversation, system_message)
            
            # Encode instruction part
            encoded_instruction = self._tokenizer.encode(
                instruction_prompt,
                max_length=self.max_seq_length,
                add_special_tokens=False
            )
        else:
            # Fallback: if no proper conversation structure, use empty instruction
            encoded_instruction = []
        
        # Encode full conversation (instruction + response)
        encoded_full = self._tokenizer.encode(
            full_conversation, 
            max_length=self.max_seq_length - 1,  # Reserve space for EOS
            return_tensors="pt", 
            add_special_tokens=False
        ).squeeze_()
        
        # Add EOS token
        encoded_prompt_and_response = torch.cat((
            encoded_full, 
            torch.tensor([self._tokenizer.eos_token_id])
        ))
        
        # Create labels and attention mask
        labels = encoded_prompt_and_response.clone()
        attention_mask = torch.ones_like(encoded_prompt_and_response)
        
        # Apply masking strategy based on configuration
        masking_strategy = getattr(self.config, 'masking_strategy', 'context_aware')
        
        if self.mask_prompt:
            # Calculate instruction length for proper masking
            instruction_length = len(encoded_instruction)
            
            if masking_strategy == 'context_aware':
                # Context-aware masking: Full attention, mask only instruction labels
                # Model sees full context but only learns from responses
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
                # attention_mask remains all 1s for real tokens
                
            elif masking_strategy == 'response_only':
                # Response-only masking: Mask both attention and labels
                # Model only attends to and learns from responses
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
                    attention_mask[:instruction_length] = 0
                
            else:
                # Default to context_aware for unknown strategies
                self.logger.warning(f"Unknown masking_strategy '{masking_strategy}', defaulting to 'context_aware'")
                if instruction_length > 0:
                    labels[:instruction_length] = self.ignore_index
            
        # Apply padding strategy based on configuration
        padding_strategy = getattr(self.config, 'padding_strategy', 'fixed')
        
        if padding_strategy == 'dynamic':
            # Dynamic padding: return sequences as-is, padding will be handled at batch level
            # Store original length for batch-level padding
            return {
                "input_ids": encoded_prompt_and_response,
                "attention_mask": attention_mask,
                "labels": labels,
                "length": len(encoded_prompt_and_response)  # Store length for dynamic padding
            }
        else:
            # Fixed padding: pad to context length (current behavior)
            seq_len = len(encoded_prompt_and_response)
            if seq_len < self.config.context_length:
                pad_length = self.config.context_length - seq_len
                
                # Pad with pad_token_id (or 0 if not available)
                pad_token_id = getattr(self._tokenizer, 'pad_token_id', 0) or 0
                
                encoded_prompt_and_response = torch.cat([
                    encoded_prompt_and_response,
                    torch.full((pad_length,), pad_token_id, dtype=torch.long)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), self.ignore_index, dtype=torch.long)
                ])
                
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=torch.long)
                ])
            elif seq_len > self.config.context_length:
                # Truncate if too long
                encoded_prompt_and_response = encoded_prompt_and_response[:self.config.context_length]
                labels = labels[:self.config.context_length]
                attention_mask = attention_mask[:self.config.context_length]
        
        return {
            "input_ids": encoded_prompt_and_response,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _apply_dynamic_padding(self, dataset: HFDataset) -> HFDataset:
        """
        Apply dynamic padding to a dataset by padding all sequences to the maximum length found.
        
        Args:
            dataset (HFDataset): Dataset with variable-length sequences containing 'length' field.
            
        Returns:
            HFDataset: Dataset with sequences padded to maximum length.
        """
        if 'length' not in dataset.column_names:
            self.logger.warning("Dynamic padding requested but 'length' field not found. Skipping dynamic padding.")
            return dataset
            
        # Find maximum sequence length in the dataset
        max_length = max(dataset['length'])
        max_length = min(max_length, self.config.context_length)  # Respect context length limit
        
        self.logger.info(f"Applying dynamic padding to maximum length: {max_length}")
        
        # Get padding token
        pad_token_id = getattr(self._tokenizer, 'pad_token_id', 0) or 0
        
        def pad_sequence(example):
            current_length = len(example['input_ids'])
            
            if current_length < max_length:
                pad_length = max_length - current_length
                
                # Pad input_ids
                example['input_ids'] = example['input_ids'] + [pad_token_id] * pad_length
                
                # Pad attention_mask (0 for padding)
                example['attention_mask'] = example['attention_mask'] + [0] * pad_length
                
                # Pad labels (ignore_index for padding)
                example['labels'] = example['labels'] + [self.ignore_index] * pad_length
            
            return example
        
        # Apply padding and remove length column
        padded_dataset = dataset.map(pad_sequence, remove_columns=['length'])
        
        self.logger.info(f"Dynamic padding completed. All sequences padded to {max_length} tokens.")
        return padded_dataset

    def _process_multi_keys_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Process multiple keys datasets and combine them with enhanced performance and logging.

        Args:
            dataset (HFDataset): Dataset to process.
            
        Returns:
            HFDataset: Combined and tokenized dataset.
        """
        start_time = time.time()

        self.logger.info(f"Starting processing key instruction datasets")

        #Create a HFDataset with key, prompt, and prompt_and_response columns
        datasetProcessed = []

        keys = list(set(dataset['key']))
        key_stats = defaultdict(int)

        for key in keys:
            key_start_time = time.time()
            self.logger.info(f"Processing Key: {key}")
            key_data = dataset.filter(lambda x: x['key'] == key)

            if len(key_data) == 0:
                self.logger.warning(f"No data found for key '{key}', skipping.")
                continue

            # Process each example in the key dataset
            for example in key_data:
                try:
                    # Create conversation without target for prompt generation
                    conversation_without_target, system_message = self._create_conversation(example, include_target=False)
                    prompt = self._create_prompt(conversation_without_target, system_message, add_generation_prompt=True)
                    
                    # Create full conversation with target for complete response
                    conversation_with_target, _ = self._create_conversation(example, include_target=True)
                    prompt_and_response = self._create_prompt(conversation_with_target, system_message, add_generation_prompt=False)
                    
                    # Append to processed dataset
                    datasetProcessed.append({
                        'key': key,
                        'prompt': prompt,
                        'prompt_and_response': prompt_and_response
                    })

                except Exception as e:
                    self.logger.error(f"Error processing example for Key '{key}': {str(e)}")
                    continue

            key_time = time.time() - key_start_time
            key_stats[key] = len(key_data)

            self.logger.info(f"Completed {key}: {key_stats[key]} examples in {key_time:.2f}s")

        # Final processing statistics
        total_time = time.time() - start_time
        total_examples = len(datasetProcessed)
        
        if not datasetProcessed:
            raise ValueError("No valid data found in provided dataset paths")
        
        # Comprehensive logging
        self.logger.info("=== Multi-key Dataset Processing Summary ===")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Total examples loaded: {total_examples}")
        self.logger.info(f"Average processing speed: {total_examples/total_time:.1f} examples/sec")

        # Key distribution
        self.logger.info(f"Keys distribution:")
        for key, count in sorted(key_stats.items()):
            percentage = (count / total_examples) * 100 if total_examples > 0 else 0
            self.logger.info(f"  {key}: {count} examples ({percentage:.1f}%)")

        # Convert to HuggingFace Dataset efficiently
        self.logger.info("Converting to HuggingFace Dataset format...")
        dataset = HFDataset.from_list(datasetProcessed)
        
        self.logger.info(f"Dataset conversion completed. Final dataset size: {len(dataset)}")
        return dataset

    def _process_generic_dataset(self, dataset: HFDataset) -> HFDataset:
        """
        Process a generic dataset with enhanced performance and logging.

        Args:
            dataset (HFDataset): Dataset to process.
            
        Returns:
            HFDataset: Processed dataset with structured conversations.
        """
        start_time = time.time()
        self.logger.info(f"Starting processing of generic instruction dataset with {len(dataset)} examples")
        
        # Log dataset structure for debugging
        self.logger.info(f"Dataset column names: {dataset.column_names}")
        if len(dataset) > 0:
            sample = dataset[0]
            self.logger.info(f"Sample data structure: {list(sample.keys())}")
            self.logger.info(f"Sample data types: {[(k, type(v).__name__) for k, v in sample.items()]}")
        
        # Create a HFDataset with prompt and prompt_and_response columns
        datasetProcessed = []

        for example in dataset:
            try:
                # Create conversation without target for prompt generation
                conversation_without_target, system_message = self._create_conversation(example, include_target=False)
                prompt = self._create_prompt(conversation_without_target, system_message, add_generation_prompt=True)
                
                # Create full conversation with target for complete response
                conversation_with_target, _ = self._create_conversation(example, include_target=True)
                prompt_and_response = self._create_prompt(conversation_with_target, system_message, add_generation_prompt=False)
                
                # Append to processed dataset
                datasetProcessed.append({
                    'prompt': prompt,
                    'prompt_and_response': prompt_and_response
                })

            except Exception as e:
                self.logger.error(f"Error processing example: {str(e)}")
                continue

        # Final processing statistics
        total_time = time.time() - start_time
        total_examples = len(datasetProcessed)
        
        if not datasetProcessed:
            raise ValueError("No valid data found in provided dataset paths")
        
        # Comprehensive logging
        self.logger.info("=== Generic Instruction Dataset Processing Summary ===")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Total examples loaded: {total_examples}")
        self.logger.info(f"Average processing speed: {total_examples/total_time:.1f} examples/sec")

        # Convert to HuggingFace Dataset efficiently
        self.logger.info("Converting to HuggingFace Dataset format...")
        dataset = HFDataset.from_list(datasetProcessed)
        
        self.logger.info(f"Dataset conversion completed. Final dataset size: {len(dataset)}")
        return dataset

    def _tokenize_dataset_batch(self, dataset: HFDataset) -> HFDataset:
        """
        Tokenize the entire dataset using optimized batch processing.
        
        Args:
            dataset (HFDataset): Dataset to tokenize.
            
        Returns:
            HFDataset: Tokenized dataset.
        """
        start_time = time.time()
        self.logger.info(f"Starting dataset tokenization for {len(dataset)} examples")
        
        # Get optimal number of processes
        num_proc = self._get_optimal_num_proc()
        
        # Performance optimization parameters
        batch_size = getattr(self.config, 'batch_size', 1000)
        writer_batch_size = getattr(self.config, 'writer_batch_size', 10000)
        
        self.logger.info(f"Using num_proc={num_proc}, batch_size={batch_size}, writer_batch_size={writer_batch_size}")
        
        # Determine which tokenization function to use based on dataset structure
        if 'key' in dataset.column_names or set(dataset.column_names) == {'prompt', 'prompt_and_response'}:
            # Use the new tokenization function for processed datasets
            tokenization_function = self._tokenize_instruction_example
        else:
            # Use the original tokenization function for raw datasets
            tokenization_function = self._tokenize_instruction_example_original
        
        # Apply tokenization with enhanced performance settings
        map_kwargs = {
            "function": tokenization_function,
            "remove_columns": dataset.column_names,
            "batch_size": batch_size,
            "writer_batch_size": writer_batch_size,
            "desc": "Tokenizing instruction dataset"
        }
        
        # Only add num_proc for slow tokenizers
        if num_proc is not None:
            map_kwargs["num_proc"] = num_proc
            
        # Add progress tracking if enabled
        if hasattr(self.config, 'show_progress') and self.config.show_progress:
            map_kwargs["desc"] = "Tokenizing instruction dataset"
        
        try:
            tokenized_dataset = dataset.map(**map_kwargs)
            
            # Apply dynamic padding if configured
            padding_strategy = getattr(self.config, 'padding_strategy', 'fixed')
            if padding_strategy == 'dynamic':
                tokenized_dataset = self._apply_dynamic_padding(tokenized_dataset)
            
            # Set format for PyTorch with explicit column specification
            tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            
            # Performance metrics
            elapsed_time = time.time() - start_time
            throughput = len(dataset) / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info(f"Tokenization completed successfully")
            self.logger.info(f"Processed {len(dataset)} examples in {elapsed_time:.2f} seconds")
            self.logger.info(f"Throughput: {throughput:.1f} examples/sec")
            self.logger.info(f"Final dataset size: {len(tokenized_dataset)}")
            
        except Exception as e:
            self.logger.error(f"Error during batch tokenization: {str(e)}")
            raise

        try:
            # Split the dataset in train and validation sets per key if 'key' column exists else split directly
            if 'key' in tokenized_dataset.column_names:
                tokenized_dataset = self._split_dataset_per_key(tokenized_dataset)
            else:
                # If no 'key' column, split the dataset directly
                self.logger.info("No 'key' column found, splitting dataset directly")
                tokenized_dataset = tokenized_dataset.train_test_split(
                    test_size=self.test_size,
                    seed=self.seed
                )
                tokenized_dataset = DatasetDict({
                    'train': tokenized_dataset['train'],
                    'validation': tokenized_dataset['test']
                })
                
        except Exception as e:
            self.logger.error(f"Error during dataset splitting: {str(e)}")
            raise

        return tokenized_dataset

    def _split_dataset_per_key(self, dataset: HFDataset) -> DatasetDict:
        """
        Split the dataset into train and validation sets per key.
        
        Args:
            dataset (HFDataset): Dataset to split.
            
        Returns:
            DatasetDict: Dictionary with 'train' and 'validation' splits.
        """
        self.logger.info("Splitting dataset into train and validation sets per key")
        
        # Initialize empty splits
        train_split = None
        validation_split = None
        
        for key in list(set(dataset['key'])):
            key_rows = dataset.filter(lambda x: x['key'] == key)
            
            # Delete the key column
            key_rows = key_rows.remove_columns(['key'])
            
            split = key_rows.train_test_split(
                test_size=self.test_size,
                seed=self.seed
            )
            
            if train_split is None and validation_split is None:
                train_split = split['train']
                validation_split = split['test']
            else:
                train_split = concatenate_datasets([train_split, split['train']])
                validation_split = concatenate_datasets([validation_split, split['test']])
        
        return DatasetDict({
            'train': train_split,
            'validation': validation_split
        })

    def tokenize(self, dataset: HFDataset) -> Union[HFDataset, DatasetDict]:
        """
        Tokenize input dataset(s) for instruction tuning with comprehensive performance monitoring.

        This method handles different input types:
        - HFDataset: Single dataset to tokenize

        Args:
            dataset: Dataset(s) to tokenize - can be HFDataset, DatasetDict, or list of paths.

        Returns:
            Union[HFDataset, DatasetDict]: Tokenized dataset(s) with input_ids, attention_mask, and labels.
        """
        if self._tokenizer is None:
            self._initialize_tokenizer()
            
        start_time = time.time()
        
        self.logger.info("=== Starting Instruction Tokenization Process ===")
        self.logger.info(f"Input type: {type(dataset).__name__}")
        self.logger.info(f"Tokenizer: {self.config.tokenizer_name}")
        self.logger.info(f"Context length: {self.config.context_length}")
        self.logger.info(f"Max sequence length: {self.max_seq_length}")
        self.logger.info(f"Mask prompt: {self.mask_prompt}")
        self.logger.info(f"Ignore index: {self.ignore_index}")
        
        # Log masking strategy
        masking_strategy = getattr(self.config, 'masking_strategy', 'context_aware')
        self.logger.info(f"Masking strategy: {masking_strategy}")
        if masking_strategy == 'context_aware':
            self.logger.info("  → Full attention to context, labels masked for prompts")
        elif masking_strategy == 'response_only':
            self.logger.info("  → Attention and labels masked for prompts")
        else:
            self.logger.warning(f"  → Unknown strategy '{masking_strategy}', using 'context_aware'")
        
        # Log padding strategy
        padding_strategy = getattr(self.config, 'padding_strategy', 'fixed')
        self.logger.info(f"Padding strategy: {padding_strategy}")
        if padding_strategy == 'fixed':
            self.logger.info(f"  → Fixed padding to {self.config.context_length} tokens")
        elif padding_strategy == 'dynamic':
            self.logger.info("  → Dynamic padding to batch maximum length")
            self.logger.info("  → Sequences will be padded during batch processing")
        else:
            self.logger.warning(f"  → Unknown strategy '{padding_strategy}', using 'fixed'")
        
        try:
            if 'key' in dataset.column_names:
                # If dataset already has 'key' column, process as files dataset
                self.logger.info("Detected dataset with 'key' column")
                keys = dataset['key']
                keys = list(set(keys))
                self.logger.info(f"Processing HFDataset with {len(keys)} keys: {', '.join(keys)}")

                # Multi-key dataset joined into a single HFDataset
                result = self._process_multi_keys_dataset(dataset)
                self.logger.info(f"Multi-key dataset processing completed. Total examples: {len(result)}")
                self.logger.info(f"Dataset column names: {result.column_names}")
                result = self._tokenize_dataset_batch(result)
            
            else:
                # Processing Generic Dataset
                self.logger.info("Detected generic dataset")
                result = self._process_generic_dataset(dataset)
                self.logger.info(f"Generic dataset processing completed. Total examples: {len(result)}")
                self.logger.info(f"Dataset column names: {result.column_names}")
                result = self._tokenize_dataset_batch(result)

            # Final performance summary
            elapsed_time = time.time() - start_time
            
            # Calculate total examples processed
            if isinstance(result, DatasetDict):
                total_examples = sum(len(split) for split in result.values())
                self.logger.info(f"Dataset splits: {list(result.keys())}")
                for split_name, split_data in result.items():
                    self.logger.info(f"  {split_name}: {len(split_data)} examples")
            else:
                total_examples = len(result)
                self.logger.info(f"Single dataset: {total_examples} examples")
            
            throughput = total_examples / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info("=== Instruction Tokenization Complete ===")
            self.logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Total examples processed: {total_examples}")
            self.logger.info(f"Overall throughput: {throughput:.1f} examples/sec")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during instruction tokenization: {str(e)}")
            raise
