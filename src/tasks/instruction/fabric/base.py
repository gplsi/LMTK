"""
This module defines a base trainer class using Lightning Fabric for instruction fine-tuning.

It encapsulates functionalities such as dataset loading, model instantiation, training loop, validation,
checkpointing, logging, and gradient accumulation. The FabricTrainerBase class is designed as an abstract base
class, providing a template for custom training strategies for instruction fine-tuning.
"""

import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from datasets import Dataset as HFDataset
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
from typing import Tuple, Union, Dict, Any
from box import Box
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

# Import Lightning and other necessary libraries
import lightning as L
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

# Import custom utilities
from src.tasks.pretraining.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.instruction.fabric.logger import step_csv_logger
from utils.logging import get_logger
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, DeepSpeedStrategy, DataParallelStrategy


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class InstructionFabricBase:
    """
    Base class for instruction fine-tuning with Fabric.
    
    This class provides common functionality for instruction fine-tuning
    across different distributed training strategies.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        dataset: Union[HFDataset, DatasetDict],
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the instruction fabric base.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            dataset: Dataset for training
            cli_logger: Logger for CLI output
        """
        self.config = config
        self.devices = devices
        self.output_dir = output_dir
        self.dataset = dataset
        self.logger = cli_logger or logger
    
    def setup_tokenizer(self) -> None:
        """
        Set up the tokenizer for instruction fine-tuning.
        """
        from transformers import AutoTokenizer
        
        tokenizer_name = self.config.model.tokenizer
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.model.get("trust_remote_code", True),
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_model(self) -> torch.nn.Module:
        """
        Set up the model for instruction fine-tuning.
        
        Returns:
            A configured model
        """
        from transformers import AutoModelForCausalLM
        
        model_name = self.config.model.name
        self.logger.info(f"Loading model: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=self.config.model.get("trust_remote_code", True),
            torch_dtype=self._get_torch_dtype(),
            device_map="auto" if self.config.model.get("device_map_auto", False) else None,
        )
        
        # Set padding token embedding if needed
        if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
            if model.get_input_embeddings().weight.shape[0] <= self.tokenizer.pad_token_id:
                model.resize_token_embeddings(len(self.tokenizer))
        
        return model
    
    def _get_torch_dtype(self) -> torch.dtype:
        """
        Get the torch dtype based on the configuration.
        
        Returns:
            A torch dtype
        """
        dtype_str = self.config.model.get("dtype", "float32")
        
        if dtype_str == "float16" or dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bfloat16" or dtype_str == "bf16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def setup_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Set up the optimizer for instruction fine-tuning.
        
        Args:
            model: The model to optimize
            
        Returns:
            A configured optimizer
        """
        # Get optimizer parameters from config
        optimizer_type = self.config.optimizer.get("type", "adamw")
        lr = self.config.optimizer.get("learning_rate", 5e-5)
        weight_decay = self.config.optimizer.get("weight_decay", 0.01)
        
        # Create parameter groups for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        if optimizer_type.lower() == "adamw":
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(self.config.optimizer.get("beta1", 0.9), self.config.optimizer.get("beta2", 0.999)),
                eps=self.config.optimizer.get("epsilon", 1e-8),
            )
        elif optimizer_type.lower() == "adam":
            from torch.optim import Adam
            optimizer = Adam(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(self.config.optimizer.get("beta1", 0.9), self.config.optimizer.get("beta2", 0.999)),
                eps=self.config.optimizer.get("epsilon", 1e-8),
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Set up the learning rate scheduler for instruction fine-tuning.
        
        Args:
            optimizer: The optimizer
            num_training_steps: The total number of training steps
            
        Returns:
            A configured scheduler, or None if no scheduler is used
        """
        scheduler_type = self.config.scheduler.get("type", None)
        
        if scheduler_type is None:
            return None
        
        # Get warmup steps
        warmup_steps = self.config.scheduler.get("warmup_steps", 0)
        if isinstance(warmup_steps, float) and 0.0 <= warmup_steps < 1.0:
            warmup_steps = int(num_training_steps * warmup_steps)
        
        # Create scheduler
        if scheduler_type.lower() == "linear":
            from transformers import get_linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_type.lower() == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def create_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create dataloaders for instruction fine-tuning.
        
        Returns:
            A dictionary containing the training and validation dataloaders
        """
        from torch.utils.data import DataLoader
        
        # Get batch size and other parameters
        batch_size = self.config.training.get("batch_size", 4)
        num_workers = self.config.training.get("num_workers", 4)
        
        # Create training dataloader
        train_dataset = self.dataset["train"] if isinstance(self.dataset, dict) else self.dataset
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Create validation dataloader if available
        val_dataloader = None
        if isinstance(self.dataset, dict) and "validation" in self.dataset:
            val_dataloader = DataLoader(
                self.dataset["validation"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        
        return {
            "train": train_dataloader,
            "val": val_dataloader,
        }
    
    def compute_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the loss for a batch.
        
        Args:
            model: The model
            batch: A batch of data
            
        Returns:
            The loss tensor
        """
        # Get input and target tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids.clone())
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs.loss