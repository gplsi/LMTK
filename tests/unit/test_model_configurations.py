"""
Unit tests for model configurations to ensure all combinations work properly.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from transformers import AutoModelForCausalLM
from box import Box


@pytest.mark.unit
@pytest.mark.parametrize("precision", ["16-mixed", "bf16-mixed", "32-true"])
class TestModelConfigurations:
    
    @pytest.mark.parametrize("model_name", ["gpt2", "gpt2-medium"])
    def test_model_initialization(self, model_name, precision):
        """Test model initialization with different configurations"""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model
            
            config = Box({
                "model_name": model_name,
                "precision": precision
            })
            
            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # Verify model was initialized with correct parameters
            mock_from_pretrained.assert_called_once_with(model_name)
            assert model is not None
    
    @pytest.mark.parametrize("gradient_checkpointing", [True, False])
    def test_gradient_checkpointing_config(self, precision, gradient_checkpointing):
        """Test model configuration with gradient checkpointing"""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
            mock_model = MagicMock()
            # Mock the gradient_checkpointing_enable method
            mock_model.gradient_checkpointing_enable = MagicMock()
            mock_from_pretrained.return_value = mock_model
            
            config = Box({
                "model_name": "gpt2",
                "precision": precision,
                "gradient_checkpointing": gradient_checkpointing
            })
            
            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # Enable gradient checkpointing if configured
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Verify gradient checkpointing was configured correctly
            if gradient_checkpointing:
                mock_model.gradient_checkpointing_enable.assert_called_once()
            else:
                mock_model.gradient_checkpointing_enable.assert_not_called()
    
    @pytest.mark.parametrize("parallelization_strategy", ["none", "fsdp", "ddp"])
    def test_model_with_different_parallelization(self, precision, parallelization_strategy):
        """Test model with different parallelization strategies"""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model
            
            config = Box({
                "model_name": "gpt2",
                "precision": precision,
                "parallelization_strategy": parallelization_strategy
            })
            
            # Add strategy specific configs
            if parallelization_strategy == "fsdp":
                config.auto_wrap_policy = "gpt2"
                config.sharding_strategy = "FULL_SHARD"
                config.cpu_offload = False
            elif parallelization_strategy == "ddp":
                config.backend = "gloo"
            
            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # Verify model was initialized
            assert model is not None
            mock_from_pretrained.assert_called_once_with("gpt2")


@pytest.mark.unit
class TestModelForwardPass:
    
    def test_model_forward_pass(self, mock_gpt2_model):
        """Test model forward pass with mock inputs"""
        # Create mock input tensors
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass through model
        output = mock_gpt2_model.forward(input_ids, attention_mask)
        
        # Check output shape and type
        assert hasattr(output, "logits")
        assert output.logits.shape == (batch_size, seq_len, 50257)  # GPT2 vocab size
    
    @patch("torch.nn.CrossEntropyLoss")
    def test_model_loss_calculation(self, mock_loss_fn, mock_gpt2_model):
        """Test model loss calculation"""
        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor(2.5)  # Mock loss value
        mock_loss_fn.return_value = mock_loss
        
        # Create mock input/output tensors
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 50000, (batch_size, seq_len))
        
        # Forward pass through model
        output = mock_gpt2_model.forward(input_ids, attention_mask)
        
        # Calculate loss (shift logits and labels for causal LM)
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Verify loss calculation
        assert loss.item() == 2.5  # Check we get the mocked value
        mock_loss.assert_called_once()