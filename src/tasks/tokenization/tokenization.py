# src/tasks/tokenization.py
from typing import Optional
from box import Box
from src.tokenization.factory import TokenizerFactory, TokenizerConfig
from src.tokenization.trainer import TokenizerTrainer
from src.utils.dataset_helper import DatasetHelper, VerboseLevel

def execute(config: Box):
    """Main tokenization orchestrator."""
    
    # Load dataset
    dataset = _load_dataset(config.dataset)
    
    # Create tokenizer
    tokenizer_config = TokenizerConfig(
        vocab_size=config.tokenizer.vocab_size,
        algorithm=config.tokenizer.algorithm,
        special_tokens=config.tokenizer.special_tokens,
        training_params=config.tokenizer.training_params
    )
    
    base_tokenizer = TokenizerFactory.create_tokenizer(tokenizer_config)
    
    # Train or load tokenizer
    if config.tokenizer.get('train', False):
        trainer = TokenizerTrainer(base_tokenizer, config.tokenizer)
        tokenizer = trainer.train(dataset)
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
    
    # Save tokenizer if configured
    if config.output.save_tokenizer:
        tokenizer.save_pretrained(config.output.save_tokenizer.path)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=config.processing.truncation,
            max_length=config.processing.max_length,
            padding=config.processing.padding
        ),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Save tokenized dataset if configured
    if config.output.save_tokenized.enabled:
        tokenized_dataset.save_to_disk(config.output.save_tokenized.path)
    
    return tokenized_dataset

def _load_dataset(dataset_config: Box) -> Dataset:
    """Helper function to load dataset based on config."""
    dataset_handler = DatasetHelper(
        verbose_level=VerboseLevel.INFO
    )
    
    if dataset_config.source == "local":
        if dataset_config.format == "dataset":
            return dataset_handler.load_from_disk(dataset_config.nameOrPath)
        elif dataset_config.format == "files":
            return dataset_handler.process_files(dataset_config.nameOrPath)
        else:
            raise ValueError("Invalid dataset format")
    elif dataset_config.source == "huggingface":
        return dataset_handler.load_from_hub(dataset_config.nameOrPath)
    else:
        raise ValueError("Invalid dataset source")
