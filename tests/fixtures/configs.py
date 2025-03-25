"""
Test fixtures for configuration objects.
Provides mock configurations for testing various components.
"""
import os
import tempfile
from pathlib import Path
from box import Box

def get_base_config():
    """Base configuration with common settings"""
    return Box({
        "task": "pretraining",
        "output_dir": tempfile.mkdtemp(prefix="test_output_"),
        "seed": 42
    }, box_dots=True)

def get_tokenizer_config():
    """Test tokenizer configuration"""
    config = get_base_config()
    config.task = "tokenization"
    config.tokenizer = {
        "name": "gpt2",
        "context_length": 1024,
        "overlap": 64,
        "task": "causal_pretraining"
    }
    config.dataset = {
        "source": "local",
        "nameOrPath": "tests/fixtures/data/sample_text",
        "format": "files",
        "file_config": {
            "format": "txt"
        }
    }
    config.test_size = 0.1
    return config

def get_pretraining_config(parallelization_strategy="none"):
    """Test pretraining configuration with specified parallelization strategy"""
    config = get_base_config()
    config.task = "pretraining"
    config.model_name = "gpt2"
    config.precision = "16-mixed"
    config.number_epochs = 1
    config.batch_size = 2
    config.gradient_accumulation_steps = 1
    config.grad_clip = 1.0
    config.lr = 1e-5
    config.lr_decay = True
    config.lr_scheduler = "cosine"
    config.warmup_proportion = 0.1
    config.weight_decay = 0.01
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.dataset = {
        "source": "local",
        "nameOrPath": "tests/fixtures/data/tokenized_dataset"
    }
    config.parallelization_strategy = parallelization_strategy
    config.logging_config = "none"
    
    # Add strategy specific configs
    if parallelization_strategy == "fsdp":
        config.auto_wrap_policy = "gpt2"
        config.sharding_strategy = "FULL_SHARD"
        config.state_dict_type = "sharded"
        config.limit_all_gathers = True
        config.cpu_offload = False
        config.num_workers = 1
        config.gradient_checkpointing = False
    elif parallelization_strategy == "ddp":
        config.num_workers = 1
        config.backend = "gloo"
    
    return config