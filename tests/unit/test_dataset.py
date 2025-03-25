"""
Unit tests for the dataset handling functionality.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict
import torch
import numpy as np

from src.utils.dataset.storage import DatasetHandler


@pytest.mark.unit
class TestDatasetHandler:
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_dataset_"))
    
    def teardown_method(self):
        """Clean up after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_handler_initialization(self):
        """Test that the DatasetHandler initializes correctly"""
        handler = DatasetHandler()
        assert handler is not None
    
    def test_process_files(self):
        """Test processing of text files into a dataset"""
        # Create sample text files
        sample_dir = self.temp_dir / "text_files"
        sample_dir.mkdir(exist_ok=True)
        
        sample_texts = [
            "This is sample text file one.",
            "This is sample text file two with more content."
        ]
        
        for i, text in enumerate(sample_texts):
            with open(sample_dir / f"sample_{i}.txt", "w") as f:
                f.write(text)
        
        # Process the files
        handler = DatasetHandler()
        dataset = handler.process_files(str(sample_dir), extension="txt")
        
        # Check that the dataset was created correctly
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert "text" in dataset.column_names
        
        # Verify contents
        texts = [sample["text"] for sample in dataset]
        assert sample_texts[0] in texts
        assert sample_texts[1] in texts
    
    @patch("datasets.load_dataset")
    def test_load_from_huggingface(self, mock_load_dataset):
        """Test loading dataset from HuggingFace"""
        # Mock the HuggingFace dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        # Load the dataset
        handler = DatasetHandler()
        result = handler.load_from_huggingface("test_dataset")
        
        # Verify the dataset was loaded
        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with("test_dataset")
    
    @patch("datasets.load_from_disk")
    def test_load_from_disk(self, mock_load_from_disk):
        """Test loading dataset from disk"""
        # Mock the dataset
        mock_dataset = MagicMock()
        mock_load_from_disk.return_value = mock_dataset
        
        # Load the dataset
        handler = DatasetHandler()
        result = handler.load_from_disk("test/path")
        
        # Verify the dataset was loaded
        assert result == mock_dataset
        mock_load_from_disk.assert_called_once_with("test/path")
    
    def test_split_dataset(self):
        """Test splitting dataset into train/test"""
        # Create a simple dataset
        data = {
            "input_ids": np.random.randint(0, 1000, size=(10, 128)),
            "attention_mask": np.ones((10, 128))
        }
        dataset = Dataset.from_dict(data)
        
        # Split the dataset
        handler = DatasetHandler()
        split_dataset = handler.split(dataset, split_ratio=0.2)
        
        # Verify the split
        assert isinstance(split_dataset, DatasetDict)
        assert "train" in split_dataset
        assert "test" in split_dataset
        assert len(split_dataset["train"]) == 8  # 80% of 10
        assert len(split_dataset["test"]) == 2   # 20% of 10
    
    def test_save_and_load_dataset(self):
        """Test saving and loading a dataset"""
        # Create a simple dataset
        data = {
            "input_ids": np.random.randint(0, 1000, size=(5, 128)),
            "attention_mask": np.ones((5, 128))
        }
        dataset = Dataset.from_dict(data)
        
        # Save the dataset
        save_path = self.temp_dir / "saved_dataset"
        dataset.save_to_disk(save_path)
        
        # Load the dataset
        handler = DatasetHandler()
        loaded_dataset = handler.load_from_disk(save_path)
        
        # Verify the dataset was loaded correctly
        assert isinstance(loaded_dataset, Dataset)
        assert "input_ids" in loaded_dataset.column_names
        assert "attention_mask" in loaded_dataset.column_names
        assert len(loaded_dataset) == 5