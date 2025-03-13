"""
Test fixtures for data and models.
Provides mock datasets and model implementations for testing.
"""
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import torch
from datasets import Dataset

def create_mock_text_data():
    """
    Creates a temporary directory with sample text files for testing tokenization
    """
    data_dir = Path("tests/fixtures/data/sample_text")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample text files
    texts = [
        "This is a sample text for testing tokenization. It contains simple sentences.",
        "Another text sample with different words to ensure variety in the tokenization process.",
        "The quick brown fox jumps over the lazy dog. This sentence contains all letters in English."
    ]
    
    for i, text in enumerate(texts):
        with open(data_dir / f"sample_{i}.txt", 'w') as f:
            f.write(text)
    
    return data_dir

def create_mock_tokenized_dataset():
    """
    Creates a mock tokenized dataset for testing pretraining
    """
    data_dir = Path("tests/fixtures/data/tokenized_dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple tokenized dataset
    n_samples = 10
    context_length = 128
    
    # Random token IDs
    input_ids = np.random.randint(0, 50000, size=(n_samples, context_length))
    attention_mask = np.ones_like(input_ids)
    
    # Convert to PyTorch tensors and save
    dataset = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    
    # Save as Arrow dataset
    dataset.save_to_disk(str(data_dir))
    
    return data_dir

class MockGPT2:
    """Mock GPT2 model for testing"""
    def __init__(self):
        self.config = type('obj', (object,), {
            'vocab_size': 50257,
            'n_positions': 1024,
            'n_embd': 768,
            'n_layer': 2,
            'n_head': 4,
        })
        # Create a small mock model
        self.transformer = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Linear(768, 50257)
        )
    
    def to(self, device):
        for module in self.transformer:
            module.to(device)
        return self
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        # Mock embeddings
        hidden_states = torch.randn(batch_size, seq_len, 768, device=input_ids.device)
        # Get output from transformer
        logits = self.transformer(hidden_states)
        return type('obj', (object,), {'logits': logits})