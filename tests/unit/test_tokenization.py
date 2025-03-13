"""
Unit tests for the tokenization system.
"""
import pytest
import os
import tempfile
from pathlib import Path
import torch
from unittest.mock import patch, MagicMock

from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.causal import CausalTokenizer


@pytest.mark.unit
class TestTokenizer:
    
    @pytest.mark.parametrize("context_length", [512, 1024, 2048])
    def test_causal_tokenizer_initialization(self, context_length):
        """Test that the CausalTokenizer initializes with different context lengths"""
        tokenizer = CausalTokenizer("gpt2", context_length=context_length)
        assert tokenizer.context_length == context_length
        assert tokenizer.name == "gpt2"
    
    def test_tokenizer_encoding(self):
        """Test that the tokenizer correctly encodes text"""
        tokenizer = CausalTokenizer("gpt2", context_length=512)
        
        text = "This is a test sentence."
        encoding = tokenizer.encode(text)
        
        # Verify the encoding result is as expected
        assert isinstance(encoding, dict)
        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert len(encoding["input_ids"]) <= 512  # Should not exceed context length
        assert len(encoding["attention_mask"]) == len(encoding["input_ids"])
    
    def test_tokenizer_batch_encoding(self):
        """Test that the tokenizer correctly encodes a batch of texts"""
        tokenizer = CausalTokenizer("gpt2", context_length=512)
        
        texts = ["This is the first test.", "This is the second test."]
        encodings = tokenizer.encode_batch(texts)
        
        assert isinstance(encodings, list)
        assert len(encodings) == 2
        assert all("input_ids" in enc for enc in encodings)
        assert all("attention_mask" in enc for enc in encodings)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenizer_padding(self, mock_from_pretrained):
        """Test that tokenizer correctly pads sequences"""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__.return_value = {
            "input_ids": [[1, 2, 3], [4, 5]],
            "attention_mask": [[1, 1, 1], [1, 1]]
        }
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = CausalTokenizer("gpt2", context_length=5)
        
        texts = ["This is a short text", "Another text"]
        result = tokenizer.encode_batch(texts, padding=True)
        
        # Check that padding is applied
        mock_tokenizer.__call__.assert_called_with(
            texts, 
            return_tensors=None, 
            padding="max_length",
            max_length=5,
            truncation=True
        )
    
    def test_tokenizer_with_overlap(self):
        """Test tokenization with overlap"""
        tokenizer = CausalTokenizer("gpt2", context_length=10, overlap=2)
        
        # Generate a long text that will require overlap processing
        long_text = " ".join(["word"] * 20)  # Will be longer than context_length
        chunks = tokenizer.process_text_with_overlap(long_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # The last tokens of the current chunk should match the first tokens of the next chunk
            assert chunks[i]["input_ids"][-tokenizer.overlap:] == chunks[i+1]["input_ids"][:tokenizer.overlap]