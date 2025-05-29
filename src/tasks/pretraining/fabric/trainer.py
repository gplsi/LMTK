"""
Pretraining trainer module.

This module provides the PretrainingTrainer class, which implements the specific
training logic for pretraining tasks using Lightning Fabric.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from datasets import Dataset as HFDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator,
)

from torch.utils.data import DataLoader
from src.training.fabric.base import FabricTrainerBase

logger = logging.getLogger(__name__)


class PretrainingTrainer(FabricTrainerBase):
    """
    Trainer for pretraining tasks.
    
    This class extends the FabricTrainerBase to implement pretraining-specific
    training logic, including model setup, data loading, and loss computation.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        cli_logger: Optional[logging.Logger] = None,
        dataset: Optional[HFDataset] = None,
    ) -> None:
        """
        Initialize the pretraining trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            cli_logger: Logger for CLI output
            dataset: Pre-loaded dataset (optional)
        """
        super().__init__(config, devices, output_dir, cli_logger)
        self.dataset = dataset
        self.tokenizer = None
    
    def _setup_model(self) -> nn.Module:
        """
        Set up and return the model for pretraining.
        
        Returns:
            A configured model
        """
        model_name = self.config.model.name
        self.cli_logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer_name = self.config.model.get("tokenizer", model_name)
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        device_map = self.config.model.get("device_map", "auto")
        
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map if self.devices > 1 else None,
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing if configured
        if self.config.training.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        
        return model
    
    def _setup_train_dataloader(self) -> DataLoader:
        """
        Set up and return the training dataloader.
        
        Returns:
            A configured training dataloader
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for pretraining")
        
        # Get training configuration
        batch_size = self.config.training.get("batch_size", 4)
        
        # Create dataloader
        train_dataset = self.dataset["train"] if isinstance(self.dataset, dict) else self.dataset
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            pin_memory=True,
        )
        
        return train_dataloader
    
    def _setup_val_dataloader(self) -> Optional[DataLoader]:
        """
        Set up and return the validation dataloader.
        
        Returns:
            A configured validation dataloader, or None if validation is not used
        """
        if self.dataset is None or not isinstance(self.dataset, dict) or "validation" not in self.dataset:
            self.cli_logger.warning("No validation dataset provided. Skipping validation.")
            return None
        
        # Get validation configuration
        batch_size = self.config.training.get("batch_size", 4)
        
        # Create dataloader
        val_dataset = self.dataset["validation"]
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            pin_memory=True,
        )
        
        return val_dataloader
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the loss for a batch.
        
        Args:
            batch: A batch of data
            
        Returns:
            A tuple containing the loss tensor and a dictionary of additional metrics
        """
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        loss = outputs.loss
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Return loss and metrics
        metrics = {
            "perplexity": perplexity.item(),
        }
        
        return loss, metrics
