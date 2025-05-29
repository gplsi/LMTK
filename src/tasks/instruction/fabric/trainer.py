"""
Instruction fine-tuning trainer module.

This module provides the InstructionTrainer class, which implements the specific
training logic for instruction fine-tuning tasks using Lightning Fabric.
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
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

from torch.utils.data import DataLoader
from src.abstract_tasks.training.trainer import TrainerBase
from lightning.fabric import Fabric
from lightning.fabric.strategies import (
    FSDPStrategy,
    DeepSpeedStrategy,
    DDPStrategy,
    SingleDeviceStrategy,
)

logger = logging.getLogger(__name__)


class InstructionTrainer(TrainerBase):
    """
    Trainer for instruction fine-tuning tasks.
    
    This class extends the FabricTrainerBase to implement instruction-specific
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
        Initialize the instruction trainer.
        
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
        self.fabric = None
    
    def setup(self) -> None:
        """
        Set up the Fabric trainer.
        
        This method initializes the Fabric instance, sets up the model, optimizer,
        scheduler, and dataloaders, and prepares them for training with Fabric.
        """
        self.cli_logger.info("Setting up Fabric trainer")
        
        # Set up Fabric
        self._setup_fabric()
        
        # Set up model
        self.model = self._setup_model()
        
        # Set up dataloaders
        self.train_dataloader = self._setup_train_dataloader()
        self.val_dataloader = self._setup_val_dataloader()
        
        # Set up optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Prepare model, optimizer, and dataloaders with Fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)
        if self.val_dataloader is not None:
            self.val_dataloader = self.fabric.setup_dataloaders(self.val_dataloader)
        
        # Log model to metrics logger if configured
        self.metrics_logger.log_model(self.model, self.optimizer)
        
        self.cli_logger.info("Fabric trainer setup complete")
    
    def _setup_fabric(self) -> None:
        """
        Set up the Fabric instance based on the configuration.
        """
        # Get precision from config
        precision = getattr(self.config, "precision", "32-true")
        
        # Get strategy from config
        strategy_name = getattr(self.config, "strategy", "fsdp").lower()
        
        if strategy_name == "fsdp":
            # FSDP strategy
            fsdp_config = getattr(self.config, "fsdp", {})
            sharding_strategy = fsdp_config.get("sharding_strategy", "full_shard")
            
            strategy = FSDPStrategy(
                auto_wrap_policy=fsdp_config.get("auto_wrap_policy", None),
                activation_checkpointing=fsdp_config.get("activation_checkpointing", False),
                cpu_offload=fsdp_config.get("cpu_offload", False),
                sharding_strategy=sharding_strategy,
            )
        elif strategy_name == "deepspeed":
            # DeepSpeed strategy
            deepspeed_config = getattr(self.config, "deepspeed", {})
            
            strategy = DeepSpeedStrategy(
                config=deepspeed_config
            )
        elif strategy_name == "ddp":
            # DDP strategy
            strategy = DDPStrategy()
        elif strategy_name == "dp":
            # Single device strategy (DataParallel will be handled manually)
            strategy = SingleDeviceStrategy(device="cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Default to single device
            strategy = SingleDeviceStrategy(device="cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create Fabric instance
        self.fabric = Fabric(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=self.devices,
            precision=precision,
            strategy=strategy,
        )
    
    def _setup_model(self) -> nn.Module:
        """
        Set up and return the model for instruction fine-tuning.
        
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
        load_in_8bit = self.config.model.get("load_in_8bit", False)
        load_in_4bit = self.config.model.get("load_in_4bit", False)
        device_map = self.config.model.get("device_map", "auto")
        
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map if self.devices > 1 else None,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Apply LoRA if enabled
        if self.config.training.get("lora", {}).get("enabled", False):
            lora_config = self.config.training.lora
            
            # Prepare model for k-bit training if using quantization
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("alpha", 16),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get("target_modules", None),
                bias="none",
            )
            
            self.cli_logger.info(f"Applying LoRA with config: {peft_config}")
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
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
            raise ValueError("Dataset must be provided for instruction fine-tuning")
        
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
        loss = outputs.loss
        
        # Compute metrics
        metrics = {
            "perplexity": torch.exp(loss).item(),
        }
        
        return loss, metrics
        
    def train(self) -> None:
        """
        Train the model using Fabric.
        
        This method overrides the base train method to use Fabric for
        distributed training.
        """
        self.cli_logger.info("Starting Fabric training")
        
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
            
            # Set model to training mode
            self.model.train()
            
            # Iterate over batches
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Compute loss
                loss, metrics = self._compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                
                # Backward pass with Fabric
                self.fabric.backward(loss)
                
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
    
    def _save_checkpoint(self, step: int, is_final: bool = False) -> None:
        """
        Save a checkpoint using Fabric.
        
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
        
        # Save checkpoint with Fabric
        self.cli_logger.info(f"Saving checkpoint to {checkpoint_path}")
        self.fabric.save(checkpoint_path, self.model, self.optimizer, self.scheduler)
