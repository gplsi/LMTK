"""
Unit tests for FSDP to HuggingFace format conversion
"""
import os
import unittest
import tempfile
import torch
from unittest.mock import patch, MagicMock, mock_open
from box import Box

from src.tasks.publish.format.fsdp import ConvertFSDPCheckpoint
from src.tasks.publish.orchestrator import PublishOrchestrator


class TestFSDPFormatConversion(unittest.TestCase):
    """Test the FSDP to HuggingFace format conversion functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "model.pth")
        self.base_model = "test/model"
        self.host = "huggingface"
        
        # Create test config
        self.config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "repo_id": "test/repo",
                "checkpoint_path": self.checkpoint_path,
                "format": "fsdp"
            }
        })

    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()

    def _create_mock_checkpoint(self):
        """Create a mock FSDP checkpoint for testing"""
        # Create a mock state dict that mimics FSDP structure
        state_dict = {
            "model.model.layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
            "model.model.layers.0.self_attn.k_proj.weight": torch.randn(10, 10),
            "model.model.layers.0.self_attn.v_proj.weight": torch.randn(10, 10),
            "model.embed_tokens.weight": torch.randn(10, 10),
            "model.lm_head.weight": torch.randn(10, 10)
        }
        return state_dict

    @patch("torch.load")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_fix_fsdp_state_dict_keys(self, mock_model_class, mock_config_class, mock_torch_load):
        """Test the FSDP state dict key fixing functionality"""
        # Setup mocks
        mock_state_dict = self._create_mock_checkpoint()
        mock_torch_load.return_value = mock_state_dict
        
        # Create mock model with state dict
        mock_model = MagicMock()
        expected_model_state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(10, 10),
            "model.embed_tokens.weight": torch.randn(10, 10),
            "lm_head.weight": torch.randn(10, 10)
        }
        mock_model.state_dict.return_value = expected_model_state_dict
        mock_model_class.return_value = mock_model
        
        # Execute the conversion
        converter = ConvertFSDPCheckpoint(self.host, self.base_model, self.checkpoint_path)
        fixed_state_dict = converter._fix_fsdp_state_dict_keys(mock_state_dict, expected_model_state_dict)
        
        # Verify the key mapping
        self.assertIn("model.layers.0.self_attn.q_proj.weight", fixed_state_dict)
        self.assertIn("model.layers.0.self_attn.k_proj.weight", fixed_state_dict)
        self.assertIn("model.layers.0.self_attn.v_proj.weight", fixed_state_dict)
        self.assertIn("model.embed_tokens.weight", fixed_state_dict)
        self.assertIn("lm_head.weight", fixed_state_dict)
        
        # Verify the original keys are properly mapped
        self.assertEqual(
            fixed_state_dict["model.layers.0.self_attn.q_proj.weight"],
            mock_state_dict["model.model.layers.0.self_attn.q_proj.weight"]
        )

    @patch("torch.load")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_load_fsdp_checkpoint_safely_model_state_dict(self, mock_model_class, mock_config_class, mock_torch_load):
        """Test loading FSDP checkpoint with model_state_dict format"""
        # Setup mock with model_state_dict format
        mock_state_dict = self._create_mock_checkpoint()
        mock_torch_load.return_value = {"model_state_dict": mock_state_dict}
        
        # Execute the loading
        converter = ConvertFSDPCheckpoint(self.host, self.base_model, self.checkpoint_path)
        loaded_state_dict = converter._load_fsdp_checkpoint_safely(self.checkpoint_path)
        
        # Verify the correct dict was extracted
        self.assertEqual(loaded_state_dict, mock_state_dict)

    @patch("torch.load")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_load_fsdp_checkpoint_safely_state_dict(self, mock_model_class, mock_config_class, mock_torch_load):
        """Test loading FSDP checkpoint with state_dict format"""
        # Setup mock with state_dict format
        mock_state_dict = self._create_mock_checkpoint()
        mock_torch_load.return_value = {"state_dict": mock_state_dict}
        
        # Execute the loading
        converter = ConvertFSDPCheckpoint(self.host, self.base_model, self.checkpoint_path)
        loaded_state_dict = converter._load_fsdp_checkpoint_safely(self.checkpoint_path)
        
        # Verify the correct dict was extracted
        self.assertEqual(loaded_state_dict, mock_state_dict)

    @patch("torch.load")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_load_fsdp_checkpoint_safely_distributed(self, mock_model_class, mock_config_class, mock_torch_load):
        """Test loading FSDP checkpoint with distributed format"""
        # Setup mock with distributed format
        mock_state_dict = self._create_mock_checkpoint()
        mock_torch_load.return_value = {"app": {"model": mock_state_dict}}
        
        # Execute the loading
        converter = ConvertFSDPCheckpoint(self.host, self.base_model, self.checkpoint_path)
        loaded_state_dict = converter._load_fsdp_checkpoint_safely(self.checkpoint_path)
        
        # Verify the correct dict was extracted
        self.assertEqual(loaded_state_dict, mock_state_dict)

    @patch("torch.load")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_execute_conversion_flow(self, mock_model_class, mock_config_class, mock_torch_load):
        """Test the complete execution flow of the FSDP conversion"""
        # Setup mocks
        mock_state_dict = self._create_mock_checkpoint()
        mock_torch_load.return_value = mock_state_dict
        
        # Create mock model
        mock_model = MagicMock()
        expected_model_state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(10, 10),
            "model.embed_tokens.weight": torch.randn(10, 10),
            "lm_head.weight": torch.randn(10, 10)
        }
        mock_model.state_dict.return_value = expected_model_state_dict
        
        # Mock load_state_dict to return empty lists for missing/unexpected keys
        mock_model.load_state_dict.return_value = ([], [])
        mock_model_class.return_value = mock_model
        
        # Execute the conversion
        converter = ConvertFSDPCheckpoint(self.host, self.base_model, self.checkpoint_path)
        result_model = converter.execute()
        
        # Verify the model was returned
        self.assertEqual(result_model, mock_model)
        
        # Verify load_state_dict was called
        mock_model.load_state_dict.assert_called_once()
        
        # Verify tie_weights was called
        mock_model.tie_weights.assert_called_once()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.format.fsdp.ConvertFSDPCheckpoint")
    def test_orchestrator_format_model(self, mock_converter_class, mock_tokenizer_class):
        """Test the format_model method in the PublishOrchestrator"""
        # Setup mocks
        mock_converter = MagicMock()
        mock_model = MagicMock()
        mock_converter.execute.return_value = mock_model
        mock_converter_class.return_value = mock_converter
        
        # Execute the orchestrator method
        orchestrator = PublishOrchestrator(self.config)
        result_model = orchestrator.format_model()
        
        # Verify the converter was created with correct parameters
        mock_converter_class.assert_called_once_with(
            self.host, 
            self.base_model,
            self.checkpoint_path
        )
        
        # Verify execute was called
        mock_converter.execute.assert_called_once()
        
        # Verify the model was returned
        self.assertEqual(result_model, mock_model)


if __name__ == "__main__":
    unittest.main()
