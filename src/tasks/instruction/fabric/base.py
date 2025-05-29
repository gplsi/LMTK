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

class FabricTrainerBase(ABC):
    """
    Abstract base trainer class for managing instruction fine-tuning using Lightning Fabric.
    
    This class provides the basic structure required for training by handling dataset preparation,
    strategy setup, logging, checkpointing, gradient accumulation, and validation.
    It is intended to be subclassed with a concrete implementation of the _setup_strategy method.
    """
    def __init__(self, devices: int, config: Box, dataset: HFDataset, checkpoint_path: str = None) -> None:
        """
        Initialize the FabricTrainerBase instance.

        Parameters:
        - devices (int): The number of devices to use for training.
        - config (Box): Configuration object containing training parameters.
        - dataset (HFDataset): The dataset (or DatasetDict) used for training.
        - checkpoint_path (str, optional): Path to a checkpoint to resume training, if applicable.

        Raises:
        - ValueError: If dataset is None.
        """
        self.cli_logger = get_logger(__name__, config.verbose_level)
        
        if dataset is None:
            raise ValueError("Dataset must be provided for training.")
        
        self.devices = devices
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.state = {}
        self.dataset = dataset
        
        # Load datasets and create dataloaders
        result = self._load_fabric_datasets_dataloaders(self.config, self.dataset)
        self.datasets = result["datasets"]
        self.dataloaders = result["dataloaders"]
    
    
    @abstractmethod
    def _setup_strategy(self) -> Union[FSDPStrategy, DDPStrategy, DeepSpeedStrategy, DataParallelStrategy]:
        """
        Abstract method to set up the training strategy.
        
        This method should be implemented in subclasses to return the desired training strategy instance.
        """
        pass
        
    def setup(self) -> None:
        """
        Set up and launch the training pipeline.

        This method configures the training strategy, sets up loggers, and then launches the training pipeline using Lightning Fabric.
        """
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.get("precision", "bf16-true"),
            loggers=loggers,
        )
        self.hparams = {
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        self.cli_logger.debug(self.hparams)
        fabric.launch(self._pipeline)

    def _set_loggers(self) -> list:
        """
        Set up training loggers based on configuration.

        Returns:
        - list: A list of logger objects to be used during training.
        """
        logger = step_csv_logger(
            self.config.output_dir, 
            self.config.model.name, 
            flush_logs_every_n_steps=self.config.get("log_iter_interval", 10)
        )
        
        if self.config.get("logging_config") == "wandb":
            wandb_logger = WandbLogger(
                entity=self.config.wandb_entity, 
                project=self.config.wandb_project, 
                log_model=self.config.get("log_model", False)
            )
            return [logger, wandb_logger]
        return [logger]
    
    def _save(self, fabric: L.Fabric) -> None:
        """
        Save the training checkpoint.

        This method saves the current training state to the output directory if provided.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the distributed training.
        """
        if self.config.output_dir is None:
            self.cli_logger.warning("Output directory not provided. Skipping checkpoint saving.")
            return
        
        output_checkpoint_path = Path(self.config.output_dir,  f"iter-{self.state['iter_num']:06d}-ckpt.pth")
        self.cli_logger.debug(f"Saving checkpoint to {str(output_checkpoint_path)!r}")
        fabric.save(output_checkpoint_path, self.state)
    
    def _get_resume_iterator(self, iterator: int, resume_iter: int) -> Tuple[int, int]:
        """
        Get the resumed iterator state for training.

        Parameters:
        - iterator (int): The current iterator over the dataset.
        - resume_iter (int): The iteration number from which to resume training.

        Returns:
        - tuple: A tuple containing the possibly sliced iterator and the updated resume_iter.
        """
        epoch_batch_count = len(iterator)
        if resume_iter >= epoch_batch_count:
            return None, resume_iter - epoch_batch_count
        elif resume_iter > 0:
            return itertools.islice(iterator, resume_iter, None), 0
        else:
            return iterator, resume_iter
    
    def _load_from_checkpoint(self, fabric: L.Fabric) -> None:
        """
        Load model and optimizer state from a checkpoint if a checkpoint path is provided.

        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the training.
        """
        if self.checkpoint_path is not None:
            self.cli_logger.debug(f"Resuming training from '{self.checkpoint_path}'")
            fabric.load(self.checkpoint_path, self.state)
    
    def _train_logs(self, fabric: L.Fabric, loss: torch.Tensor) -> None:
        """
        Log training metrics for monitoring.
        """
        
        self.cli_logger.debug(
            f"iter {self.state['iter_num']} step {self.state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(self.train_t1 - self.train_iter_t0) * 1000:.2f}ms"
        )
        self.monitor.on_train_batch_end(
            self.state["iter_num"] * self.config.batch_size,
            self.train_t1 - self.train_total_t0,
            fabric.world_size,
            self.state["step_count"],
            lengths=self.total_lengths,
            train_loss=loss.item()
        )
    
    def _gradient_clipping(self, fabric: L.Fabric, model: L.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """
        Clip model gradients to avoid exploding gradients.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - optimizer (torch.optim.Optimizer): The optimizer used for training.
        """
        if self.config.get("grad_clip", 0.0) > 0:
            fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    def _load_fabric_datasets_dataloaders(self, config: Box, dataset: HFDataset) -> Dict[str, Any]:
        """
        Load datasets and create dataloaders for training.

        Parameters:
        - config (Box): Configuration object containing dataset parameters.
        - dataset (HFDataset): The dataset to load.

        Returns:
        - dict: A dictionary containing the loaded datasets and dataloaders.
        """
        # Prepare datasets
        if isinstance(dataset, DatasetDict):
            datasets = dataset
        else:
            datasets = DatasetDict({"train": dataset})
        
        # Create dataloaders
        batch_size = config.get("batch_size", 4)
        
        dataloaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )
        }
        
        if "validation" in datasets:
            dataloaders["validation"] = DataLoader(
                datasets["validation"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )
        
        return {"datasets": datasets, "dataloaders": dataloaders}
    
    def _setup_model_and_optimizer(self, fabric: L.Fabric) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Set up the model and optimizer for training.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.

        Returns:
        - tuple: A tuple containing the model and optimizer.
        """
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if "bf16" in self.config.get("precision", "bf16-true") else torch.float32,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("lr", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.95)),
        )
        
        # Setup learning rate scheduler
        num_training_steps = len(self.dataloaders["train"]) * self.config.get("num_epochs", 3)
        lr_scheduler = get_scheduler(
            name=self.config.get("lr_scheduler", "linear"),
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * self.config.get("warmup_proportion", 0.06)),
            num_training_steps=num_training_steps,
        )
        
        # Setup model, optimizer, and scheduler with fabric
        model, optimizer = fabric.setup(model, optimizer)
        lr_scheduler = fabric.setup_scheduler(lr_scheduler)
        
        return model, optimizer, lr_scheduler
    
    def _pipeline(self, fabric: L.Fabric) -> None:
        """
        Main training pipeline.

        This method orchestrates the complete training process, including model setup,
        optimizer configuration, training loop, validation, and checkpointing.

        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the distributed training.
        """
        self.cli_logger.info("Setting up model and optimizer")
        model, optimizer, lr_scheduler = self._setup_model_and_optimizer(fabric)
        
        # Initialize state
        self.state = {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "iter_num": 0,
            "step_count": 0,
        }
        
        # Load from checkpoint if provided
        self._load_from_checkpoint(fabric)
        
        # Setup speed monitor
        self.monitor = Monitor(fabric, self.config.get("log_iter_interval", 10))
        
        # Training loop
        self.cli_logger.info("Starting training")
        self.train_total_t0 = time.time()
        
        for epoch in range(self.config.get("num_epochs", 3)):
            self.cli_logger.info(f"Starting epoch {epoch}")
            
            # Training
            model.train()
            for batch in tqdm(self.dataloaders["train"], desc=f"Epoch {epoch}"):
                self.train_iter_t0 = time.time()
                
                # Forward pass
                outputs = model(**{k: v.to(fabric.device) for k, v in batch.items() if k != "text"})
                loss = outputs.loss
                
                # Backward pass
                fabric.backward(loss)
                
                # Gradient clipping
                self._gradient_clipping(fabric, model, optimizer)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update state
                self.state["iter_num"] += 1
                self.state["step_count"] += 1
                
                # Logging
                self.train_t1 = time.time()
                self.total_lengths = batch["input_ids"].numel()
                self._train_logs(fabric, loss)
                
                # Save checkpoint
                if self.state["iter_num"] % self.config.get("save_iter_interval", 100) == 0:
                    self._save(fabric)
            
            # Validation
            if "validation" in self.dataloaders and self.config.get("validate_after_epoch", True):
                self.cli_logger.info(f"Validating after epoch {epoch}")
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.dataloaders["validation"], desc="Validation"):
                        outputs = model(**{k: v.to(fabric.device) for k, v in batch.items() if k != "text"})
                        val_loss += outputs.loss.item()
                
                val_loss /= len(self.dataloaders["validation"])
                self.cli_logger.info(f"Validation loss: {val_loss:.4f}")
                fabric.log_dict({"val_loss": val_loss})
            
            # Save checkpoint after each epoch
            self._save(fabric)
        
        # Final save
        self._save(fabric)
        self.cli_logger.info("Training completed")
