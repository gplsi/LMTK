"""
Unit tests for the Publish Orchestrator
"""
import unittest
from unittest.mock import patch, MagicMock
from box import Box

from src.tasks.publish.orchestrator import PublishOrchestrator


class TestPublishOrchestrator(unittest.TestCase):
    """Test the PublishOrchestrator class"""

    def setUp(self):
        """Set up test environment"""
        self.base_model = "test/model"
        self.repo_id = "test/repo"
        self.checkpoint_path = "/path/to/checkpoint.pth"
        self.host = "huggingface"
        
        # Create test config
        self.config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "repo_id": self.repo_id,
                "checkpoint_path": self.checkpoint_path,
                "format": "fsdp"
            }
        })

    def test_validate_config_valid(self):
        """Test config validation with valid config"""
        orchestrator = PublishOrchestrator(self.config)
        # Should not raise an exception
        orchestrator.validate_config()

    def test_validate_config_missing_publish(self):
        """Test config validation with missing publish section"""
        config = Box({})
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish configuration must be provided", str(context.exception))

    def test_validate_config_missing_host(self):
        """Test config validation with missing host"""
        config = Box({
            "publish": {
                "base_model": self.base_model,
                "repo_id": self.repo_id,
                "checkpoint_path": self.checkpoint_path,
                "format": "fsdp"
            }
        })
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish host must be provided", str(context.exception))

    def test_validate_config_missing_base_model(self):
        """Test config validation with missing base_model"""
        config = Box({
            "publish": {
                "host": self.host,
                "repo_id": self.repo_id,
                "checkpoint_path": self.checkpoint_path,
                "format": "fsdp"
            }
        })
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish base_model must be provided", str(context.exception))

    def test_validate_config_missing_repo_id(self):
        """Test config validation with missing repo_id"""
        config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "checkpoint_path": self.checkpoint_path,
                "format": "fsdp"
            }
        })
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish repo_id must be provided", str(context.exception))

    def test_validate_config_missing_checkpoint_path(self):
        """Test config validation with missing checkpoint_path"""
        config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "repo_id": self.repo_id,
                "format": "fsdp"
            }
        })
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish checkpoint_path must be provided", str(context.exception))

    def test_validate_config_missing_format(self):
        """Test config validation with missing format"""
        config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "repo_id": self.repo_id,
                "checkpoint_path": self.checkpoint_path
            }
        })
        orchestrator = PublishOrchestrator(config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.validate_config()
        
        self.assertIn("Publish format must be provided", str(context.exception))

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.orchestrator.FORMAT_HANDLERS")
    def test_format_model(self, mock_format_handlers, mock_tokenizer_class):
        """Test the format_model method"""
        # Setup mocks
        mock_format_handler = MagicMock()
        mock_model = MagicMock()
        mock_format_handler.return_value.execute.return_value = mock_model
        mock_format_handlers.__getitem__.return_value = mock_format_handler
        
        # Execute the method
        orchestrator = PublishOrchestrator(self.config)
        result = orchestrator.format_model()
        
        # Verify the format handler was created with correct parameters
        mock_format_handlers.__getitem__.assert_called_once_with("fsdp")
        mock_format_handler.assert_called_once_with(
            self.host,
            self.base_model,
            self.checkpoint_path
        )
        
        # Verify execute was called
        mock_format_handler.return_value.execute.assert_called_once()
        
        # Verify the result is the model returned by the format handler
        self.assertEqual(result, mock_model)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.upload.huggingface.UploadHuggingface")
    def test_upload_model(self, mock_uploader_class, mock_tokenizer_class):
        """Test the upload_model method"""
        # Setup mocks
        mock_uploader = MagicMock()
        mock_uploader_class.return_value = mock_uploader
        mock_model = MagicMock()
        
        # Execute the method
        orchestrator = PublishOrchestrator(self.config)
        orchestrator.upload_model(mock_model)
        
        # Verify the uploader was created with correct parameters
        mock_uploader_class.assert_called_once()
        
        # Verify execute was called
        mock_uploader.execute.assert_called_once()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.orchestrator.PublishOrchestrator.validate_config")
    @patch("src.tasks.publish.orchestrator.PublishOrchestrator.format_model")
    @patch("src.tasks.publish.orchestrator.PublishOrchestrator.upload_model")
    def test_execute_success(self, mock_upload, mock_format, mock_validate, mock_tokenizer_class):
        """Test the execute method with successful execution"""
        # Setup mocks
        mock_model = MagicMock()
        mock_format.return_value = mock_model
        
        # Execute the method
        orchestrator = PublishOrchestrator(self.config)
        orchestrator.execute()
        
        # Verify methods were called in correct order
        mock_validate.assert_called_once()
        mock_format.assert_called_once()
        mock_upload.assert_called_once_with(mock_model)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.orchestrator.PublishOrchestrator.validate_config")
    def test_execute_error(self, mock_validate, mock_tokenizer_class):
        """Test the execute method with an error"""
        # Setup mock to raise an exception
        mock_validate.side_effect = ValueError("Test error")
        
        # Execute the method
        orchestrator = PublishOrchestrator(self.config)
        
        with self.assertRaises(ValueError) as context:
            orchestrator.execute()
        
        # Verify the error was raised
        self.assertIn("Test error", str(context.exception))
        
        # Verify validate_config was called
        mock_validate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
