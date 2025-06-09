"""
Unit tests for HuggingFace upload functionality
"""
import unittest
from unittest.mock import patch, MagicMock
from box import Box

from src.tasks.publish.upload.huggingface import UploadHuggingface
from src.tasks.publish.orchestrator import PublishOrchestrator


class TestHuggingFaceUpload(unittest.TestCase):
    """Test the HuggingFace upload functionality"""

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
        
        # Create mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

    def test_upload_tokenizer(self):
        """Test uploading tokenizer to HuggingFace Hub"""
        # Create uploader and call method
        uploader = UploadHuggingface(
            base_model=self.base_model,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            repo_id=self.repo_id
        )
        
        # Test message
        test_message = "Test commit message"
        uploader._upload_tokenizer(test_message)
        
        # Verify tokenizer was pushed to hub with correct parameters
        self.mock_tokenizer.push_to_hub.assert_called_once_with(
            self.repo_id,
            commit_message=test_message
        )

    def test_upload_model(self):
        """Test uploading model to HuggingFace Hub"""
        # Create uploader and call method
        uploader = UploadHuggingface(
            base_model=self.base_model,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            repo_id=self.repo_id
        )
        
        # Test parameters
        test_message = "Test commit message"
        test_shard_size = "10GB"
        test_safe_serialization = True
        test_create_pr = True
        
        uploader._upload_model(
            message=test_message,
            max_shard_size=test_shard_size,
            safe_serialization=test_safe_serialization,
            create_pr=test_create_pr
        )
        
        # Verify model was pushed to hub with correct parameters
        self.mock_model.push_to_hub.assert_called_once_with(
            self.repo_id,
            commit_message=test_message,
            max_shard_size=test_shard_size,
            safe_serialization=test_safe_serialization,
            create_pr=test_create_pr
        )

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("torch.no_grad")
    def test_validate_upload(self, mock_no_grad, mock_tokenizer_class, mock_model_class):
        """Test validating the uploaded model and tokenizer"""
        # Setup mocks
        mock_test_model = MagicMock()
        mock_test_tokenizer = MagicMock()
        mock_model_class.return_value = mock_test_model
        mock_tokenizer_class.return_value = mock_test_tokenizer
        
        # Mock tokenizer returning tensors
        mock_test_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        
        # Setup no_grad context manager
        mock_context = MagicMock()
        mock_no_grad.return_value = mock_context
        mock_context.__enter__.return_value = None
        
        # Create uploader and call method
        uploader = UploadHuggingface(
            base_model=self.base_model,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            repo_id=self.repo_id
        )
        
        result = uploader._validate_upload()
        
        # Verify model and tokenizer were loaded from Hub
        mock_model_class.assert_called_once_with(self.repo_id)
        mock_tokenizer_class.assert_called_once_with(self.repo_id)
        
        # Verify tokenizer was called with test input
        mock_test_tokenizer.assert_called_once()
        
        # Verify model was called with tokenizer output
        mock_test_model.assert_called_once()
        
        # Verify validation returned True
        self.assertTrue(result)

    @patch("src.tasks.publish.upload.huggingface.UploadHuggingface._upload_model")
    @patch("src.tasks.publish.upload.huggingface.UploadHuggingface._upload_tokenizer")
    @patch("src.tasks.publish.upload.huggingface.UploadHuggingface._validate_upload")
    def test_execute(self, mock_validate, mock_upload_tokenizer, mock_upload_model):
        """Test the complete execution flow of the HuggingFace upload"""
        # Setup mocks
        mock_validate.return_value = True
        
        # Create uploader and call method
        uploader = UploadHuggingface(
            base_model=self.base_model,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            repo_id=self.repo_id
        )
        
        # Test parameters
        test_message = "Test commit message"
        test_shard_size = "10GB"
        test_safe_serialization = True
        test_create_pr = True
        
        uploader.execute(
            message=test_message,
            max_shard_size=test_shard_size,
            safe_serialization=test_safe_serialization,
            create_pr=test_create_pr
        )
        
        # Verify methods were called in correct order with correct parameters
        mock_upload_model.assert_called_once_with(
            message=test_message,
            max_shard_size=test_shard_size,
            safe_serialization=test_safe_serialization,
            create_pr=test_create_pr
        )
        mock_upload_tokenizer.assert_called_once()
        mock_validate.assert_called_once()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.tasks.publish.orchestrator.UploadHuggingface")
    def test_orchestrator_upload_model(self, mock_uploader_class, mock_tokenizer_class):
        """Test the upload_model method in the PublishOrchestrator"""
        # Setup mocks
        mock_uploader = MagicMock()
        mock_uploader_class.return_value = mock_uploader
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # Create mock model
        mock_model = MagicMock()
        
        # Create config with default_box=True
        config = Box({
            "publish": {
                "host": self.host,
                "base_model": self.base_model,
                "repo_id": self.repo_id,
                "checkpoint_path": "/path/to/checkpoint.pth",
                "format": "fsdp"
            }
        }, default_box=True)
        
        # Execute the orchestrator method
        orchestrator = PublishOrchestrator(config)
        # Ensure tokenizer is set
        orchestrator.tokenizer = mock_tokenizer
        orchestrator.upload_model(mock_model)
        
        # Verify the uploader was created with correct parameters
        mock_uploader_class.assert_called_once_with(
            base_model=self.base_model,
            model=mock_model,
            tokenizer=mock_tokenizer,
            repo_id=self.repo_id
        )
        
        # Verify execute was called
        mock_uploader.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
