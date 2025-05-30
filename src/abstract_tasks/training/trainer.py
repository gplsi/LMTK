"""
Abstract trainer for training tasks.

This module provides the base trainer for all training tasks,
including pretraining and instruction fine-tuning. It handles common
functionality such as setup, training loop, and metrics logging.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.abstract_tasks.training.metrics_logger import MetricsLogger, create_metrics_logger

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """
    Base trainer for all training tasks.
    
    This abstract class provides the foundation for all trainers,
    handling common functionality such as setup, training loop,
    validation, and metrics logging. Task-specific trainers should
    inherit from this class and implement the abstract methods.
    """
    
    def __init__(
        self,
        config: Any,
        devices: Union[int, List[int]],
        output_dir: str,
        cli_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object or dictionary
            devices: Number of devices or list of device IDs
            output_dir: Directory to save outputs
            cli_logger: Logger for CLI output
        """
        self.config = config
        self.devices = devices
        self.output_dir = output_dir
        self.cli_logger = cli_logger or logger
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize attributes to be set later
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Initialize metrics logger
        self.metrics_logger = create_metrics_logger(
            config=self.config,
            output_dir=self.output_dir,
            experiment_name=getattr(self.config, "experiment_name", None),
        )
    
    def setup(self) -> None:
        """
        Set up the trainer.
        
        This method initializes the model, optimizer, scheduler,
        and dataloaders, and prepares them for training.
        """
        self.cli_logger.info("Setting up trainer")
        
        # Set up model
        self.model = self._setup_model()
        
        # Set up dataloaders
        self.train_dataloader = self._setup_train_dataloader()
        self.val_dataloader = self._setup_val_dataloader()
        
        # Set up optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Log model to metrics logger if configured
        self.metrics_logger.log_model(self.model, self.optimizer)
        
        self.cli_logger.info("Trainer setup complete")
    
    @abstractmethod
    def _setup_model(self) -> nn.Module:
        """
        Set up and return the model.
        
        Returns:
            A configured model
        """
        pass
    
    @abstractmethod
    def _setup_train_dataloader(self) -> DataLoader:
        """
        Set up and return the training dataloader.
        
        Returns:
            A configured training dataloader
        """
        pass
    
    @abstractmethod
    def _setup_val_dataloader(self) -> Optional[DataLoader]:
        """
        Set up and return the validation dataloader.
        
        Returns:
            A configured validation dataloader, or None if validation is not used
        """
        pass
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Set up and return the optimizer.
        
        Returns:
            A configured optimizer
        """
        # Get optimizer configuration
        optimizer_config = getattr(self.config, "optimizer", {})
        optimizer_name = optimizer_config.get("name", "adamw").lower()
        lr = optimizer_config.get("learning_rate", 1e-4)
        weight_decay = optimizer_config.get("weight_decay", 0.01)
        
        # Create optimizer
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    optimizer_config.get("beta1", 0.9),
                    optimizer_config.get("beta2", 0.999),
                ),
                eps=optimizer_config.get("eps", 1e-8),
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    optimizer_config.get("beta1", 0.9),
                    optimizer_config.get("beta2", 0.999),
                ),
                eps=optimizer_config.get("eps", 1e-8),
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Set up and return the learning rate scheduler.
        
        Returns:
            A configured scheduler, or None if no scheduler is used
        """
        # Get scheduler configuration
        scheduler_config = getattr(self.config, "scheduler", {})
        scheduler_name = scheduler_config.get("name", None)
        
        if scheduler_name is None:
            return None
        
        # Get total steps
        max_epochs = getattr(self.config, "max_epochs", 1)
        steps_per_epoch = len(self.train_dataloader)
        total_steps = max_epochs * steps_per_epoch
        
        # Create scheduler
        if scheduler_name.lower() == "cosine":
            # Get warmup steps
            warmup_steps = scheduler_config.get("warmup_steps", 0)
            if isinstance(warmup_steps, float):
                warmup_steps = int(warmup_steps * total_steps)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
            
            # Wrap with warmup if needed
            if warmup_steps > 0:
                scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1e-5,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    scheduler,
                ])
        elif scheduler_name.lower() == "linear":
            # Get warmup steps
            warmup_steps = scheduler_config.get("warmup_steps", 0)
            if isinstance(warmup_steps, float):
                warmup_steps = int(warmup_steps * total_steps)
            
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=scheduler_config.get("end_factor", 0.1),
                total_iters=total_steps - warmup_steps,
            )
            
            # Wrap with warmup if needed
            if warmup_steps > 0:
                scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1e-5,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    scheduler,
                ])
        elif scheduler_name.lower() == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return scheduler
    
    @abstractmethod
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the loss for a batch.
        
        Args:
            batch: A batch of data
            
        Returns:
            A tuple containing the loss tensor and a dictionary of additional metrics
        """
        pass
    
    
    def train(self) -> None:
        """
        Train the model.
        
        This method runs the training loop, including validation and metrics logging.
        """
        self.cli_logger.info("Starting training")
        
        # Get training parameters
        max_epochs = getattr(self.config, "max_epochs", 1)
        max_steps = getattr(self.config, "max_steps", -1)
        val_interval = getattr(self.config, "val_interval", -1)
        save_interval = getattr(self.config, "save_interval", -1)
        grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        
        # Initialize step counter
        global_step = 0
        
        # Training loop
        for epoch in range(max_epochs):
            self.cli_logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
            
            # Reset metrics
            epoch_loss = 0.0
            epoch_metrics = {}
            
            # Start epoch timer
            epoch_start_time = time.time()
            
            # Set model to training mode
            self.model.train()
            
            # Iterate over batches
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Compute loss
                loss, metrics = self._compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Update metrics
                epoch_loss += loss.item() * grad_accum_steps
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
                
                # Optimizer step
                if (batch_idx + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                        # Log learning rate
                        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                    
                    # Increment global step
                    global_step += 1
                    
                    # Log metrics
                    avg_loss = epoch_loss / (batch_idx + 1)
                    avg_metrics = {
                        key: value / (batch_idx + 1)
                        for key, value in epoch_metrics.items()
                    }
                    
                    # Add loss to metrics
                    avg_metrics["loss"] = avg_loss
                    
                    # Log to metrics logger
                    self.metrics_logger.log_metrics(avg_metrics, global_step)
                    
                    # Validation
                    if val_interval > 0 and global_step % val_interval == 0:
                        val_metrics = self._validate()
                        # Log validation metrics
                        self.metrics_logger.log_metrics(
                            {f"val_{key}": value for key, value in val_metrics.items()},
                            global_step
                        )
                    
                    # Save checkpoint
                    if save_interval > 0 and global_step % save_interval == 0:
                        self._save_checkpoint(global_step)
                
                # Check if max steps reached
                if max_steps > 0 and global_step >= max_steps:
                    self.cli_logger.info(f"Reached maximum steps ({max_steps})")
                    break
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            
            # Calculate average metrics
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_metrics = {
                key: value / len(self.train_dataloader)
                for key, value in epoch_metrics.items()
            }
            
            # Add loss and epoch time to metrics
            avg_metrics["loss"] = avg_loss
            avg_metrics["epoch"] = epoch + 1
            avg_metrics["epoch_time"] = epoch_time
            
            # Log epoch metrics
            self.metrics_logger.log_metrics(avg_metrics, global_step)
            
            # Validation at the end of each epoch
            if val_interval <= 0:
                val_metrics = self._validate()
                # Log validation metrics
                self.metrics_logger.log_metrics(
                    {f"val_{key}": value for key, value in val_metrics.items()},
                    global_step
                )
            
            # Save checkpoint at the end of each epoch
            if save_interval <= 0:
                self._save_checkpoint(global_step)
            
            # Check if max steps reached
            if max_steps > 0 and global_step >= max_steps:
                break
        
        # End of training
        self.cli_logger.info("Training completed")
        
        # Final save
        self._save_checkpoint(global_step, is_final=True)
        
        # Close metrics logger
        self.metrics_logger.close()
    
    def _validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            A dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.cli_logger.info("Running validation")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0.0
        val_metrics = {}
        
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over batches
            for batch in self.val_dataloader:
                # Compute loss
                loss, metrics = self._compute_loss(batch)
                
                # Update metrics
                val_loss += loss.item()
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
        
        # Calculate average metrics
        avg_loss = val_loss / len(self.val_dataloader)
        avg_metrics = {
            key: value / len(self.val_dataloader)
            for key, value in val_metrics.items()
        }
        
        # Add loss to metrics
        avg_metrics["loss"] = avg_loss
        
        # Log metrics
        self.cli_logger.info(
            f"Validation: Loss: {avg_loss:.4f}, "
            + ", ".join([
                f"{key.capitalize()}: {value:.4f}"
                for key, value in avg_metrics.items()
            ])
        )
        
        # Set model back to training mode
        self.model.train()
        
        # Return metrics
        return avg_metrics
    
    def _save_checkpoint(self, step: int, is_final: bool = False) -> None:
        """
        Save a checkpoint.
        
        Args:
            step: Current step
            is_final: Whether this is the final checkpoint
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine checkpoint path
        if is_final:
            checkpoint_path = os.path.join(checkpoint_dir, "final")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}")
        
        # Save checkpoint
        self.cli_logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Create checkpoint dictionary
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        
        # Add scheduler if available
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, f"{checkpoint_path}.pt")
