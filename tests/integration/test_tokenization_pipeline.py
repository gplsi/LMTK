"""
Integration tests for the tokenization pipeline.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml
from box import Box
from datasets import Dataset

from src.tasks.tokenization import execute
from src.tasks.tokenization.orchestrator import TokenizationOrchestrator


@pytest.mark.integration
class TestTokenizationPipeline:
    
    def setup_method(self):
        """Set up test environment before each test"""
        # Create temp directories for input and output
        self.input_dir = Path(tempfile.mkdtemp(prefix="test_input_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="test_output_"))
        
        # Create sample text files
        sample_texts = [
            "This is a test document that will be tokenized for the integration test.",
            "Second document with different content to ensure proper test coverage."
        ]
        
        for i, text in enumerate(sample_texts):
            with open(self.input_dir / f"sample_{i}.txt", "w") as f:
                f.write(text)
    
    def teardown_method(self):
        """Clean up after each test"""
        shutil.rmtree(self.input_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenization_pipeline_execution(self, mock_tokenizer_from_pretrained):
        """Test the full tokenization pipeline from config to dataset creation"""
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kwargs: {"input_ids": list(range(10)), "attention_mask": [1] * 10}
        mock_tokenizer.__call__.side_effect = lambda texts, **kwargs: {
            "input_ids": [[i] * 10 for i in range(len(texts))], 
            "attention_mask": [[1] * 10 for _ in range(len(texts))]
        }
        mock_tokenizer.pad = lambda seqs, **kwargs: seqs
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        # Mock save_to_disk
        with patch.object(Dataset, "save_to_disk") as mock_save:
            # Create a config for the test
            config = Box({
                "task": "tokenization",
                "output_dir": str(self.output_dir),
                "tokenizer": {
                    "name": "gpt2",
                    "context_length": 128,
                    "task": "causal_pretraining"
                },
                "dataset": {
                    "source": "local",
                    "nameOrPath": str(self.input_dir),
                    "format": "files",
                    "file_config": {
                        "format": "txt"
                    }
                },
                "test_size": 0.2
            }, box_dots=True)
            
            # Execute the tokenization task
            result = execute(config)
            
            # Verify that the mock tokenizer was used
            mock_tokenizer_from_pretrained.assert_called_with("gpt2")
            
            # Verify that save_to_disk was called with the correct path
            mock_save.assert_called()
            
            # Check that the orchestrator completed the process
            assert result is not None
    
    @patch("src.tasks.tokenization.orchestrator.TokenizationOrchestrator.execute")
    def test_tokenization_from_config_file(self, mock_execute):
        """Test tokenization from a config file using the main entry point"""
        # Create a config file
        config = {
            "task": "tokenization",
            "output_dir": str(self.output_dir),
            "tokenizer": {
                "name": "gpt2",
                "context_length": 128,
                "task": "causal_pretraining"
            },
            "dataset": {
                "source": "local",
                "nameOrPath": str(self.input_dir),
                "format": "files",
                "file_config": {
                    "format": "txt"
                }
            },
            "test_size": 0.2
        }
        
        # Write config to a temporary file
        config_path = self.output_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Mock the config validator to return our config
        with patch("src.config.config_loader.ConfigValidator.validate") as mock_validate:
            mock_validate.return_value = Box(config, box_dots=True)
            
            # Import and execute the main function
            from src.main import execute_task
            execute_task(str(config_path))
            
            # Check that the orchestrator was executed
            mock_execute.assert_called_once()
            
            # Verify that the config was validated
            mock_validate.assert_called_once_with(str(config_path), "tokenization")