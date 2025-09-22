"""
This module defines a base trainer class using Lightning Fabric for training deep learning models.
It encapsulates functionalities such as dataset loading, model instantiation, training loop, validation,
checkpointing, logging, and gradient accumulation. The FabricTrainerBase class is designed as an abstract base
class, providing a template for custom training strategies.
"""

import math
from pathlib import Path
import time
from src.tasks.clm_training.utils import select_optimizer, select_scheduler, deterministic 
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from datasets import Dataset as HFDataset
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
from typing import Tuple, Union
from box import Box
from transformers import AutoTokenizer

# Import Lightning and other necessary libraries
import lightning as L
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

# Import custom utilities
from src.tasks.clm_training.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.clm_training.fabric.logger import step_csv_logger
from src.tasks.clm_training.utils import *
from src.tasks.clm_training.fabric.generation import FabricGeneration
from utils.logging import get_logger
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, DeepSpeedStrategy, DataParallelStrategy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FabricTrainerBase(ABC):
    """
    Abstract base trainer class for managing training using Lightning Fabric.
    
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
        self.cli_logger.info("Setting up FSDP strategy.")
        
        # Debug logging for configuration values that might cause type issues
        config_keys_to_check = ['gradient_accumulation_steps', 'validate_after_k_steps', 'max_epochs', 'max_steps', 'batch_size', 'eval_batch_size']
        for key in config_keys_to_check:
            if hasattr(self.config, key):
                value = getattr(self.config, key)
                self.cli_logger.debug(f"Config {key}: value={value}, type={type(value)}")
        
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.precision,
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
            self.config.model_name, 
            flush_logs_every_n_steps=self.config.log_iter_interval
        )

        if self.config.get("logging_config", None) == "wandb" \
            and self.config.get("wandb_entity", None) is not None \
            and self.config.get("wandb_project", None) is not None:
            wandb_logger = WandbLogger(
                entity=self.config.wandb_entity, 
                project=self.config.wandb_project,                
                log_model=self.config.get("log_model", False)
            )
            return [logger, wandb_logger]
        return [logger]
    
    def _save(self, fabric: L.Fabric, epochFinished: bool = False, trainingFinished: bool = False) -> None:
        """
        Save the training checkpoint.

        This method saves the current training state to the output directory if provided.
        The checkpoint name will reflect the epoch number when saving after epoch completion
        or training completion, otherwise it will use the iteration number.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the distributed training.
        - epochFinished (bool): Flag indicating if the current epoch has finished.
        - trainingFinished (bool): Flag indicating if training has completed.
        """
        if self.config.output_dir is None:
            self.cli_logger.warning("Output directory not provided. Skipping checkpoint saving.")
            return
        
        # Generate checkpoint name based on context
        if epochFinished or trainingFinished:
            current_epoch = self.state.get('current_epoch', 0)
            if trainingFinished:
                checkpoint_name = f"epoch-{current_epoch:03d}-final-ckpt.pth"
            else:
                checkpoint_name = f"epoch-{current_epoch:03d}-ckpt.pth"
        else:
            checkpoint_name = f"iter-{self.state['iter_num']:06d}-ckpt.pth"
        
        output_checkpoint_path = Path(self.config.output_dir, checkpoint_name)
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
            self.cli_logger.info(f"Resuming training from '{self.checkpoint_path}'")
            # Use strict=False to allow loading checkpoints that may not have all current state keys
            fabric.load(self.checkpoint_path, self.state, strict=False)
            
            # Ensure that essential keys have default values if they weren't in the checkpoint
            if 'current_epoch' not in self.state:
                self.state['current_epoch'] = 0
                self.cli_logger.info("'current_epoch' not found in checkpoint, defaulting to 0")
            
            if 'iter_num' not in self.state:
                self.state['iter_num'] = 0
                self.cli_logger.info("'iter_num' not found in checkpoint, defaulting to 0")
                
            if 'step_count' not in self.state:
                self.state['step_count'] = 0
                self.cli_logger.info("'step_count' not found in checkpoint, defaulting to 0")
            
            # Log the loaded state for debugging
            self.cli_logger.info(f"Loaded state keys: {list(self.state.keys())}")
            self.cli_logger.info(f"Resuming from iteration {self.state.get('iter_num', 0)}, step {self.state.get('step_count', 0)}, epoch {self.state.get('current_epoch', 0)}")
    
    def _train_logs(self, fabric: L.Fabric, loss: torch.Tensor) -> None:
        """
        Log training metrics for monitoring.
        """
        
        self.cli_logger.debug(
            f"iter {self.state['iter_num']} step {self.state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(self.train_t1 - self.train_iter_t0) * 1000:.2f}ms remaining time: "
            # f"{(self.train_t1 - self.train_total_t0) / (self.state['iter_num'] - self.initial_iter) * (self.config.max_iters - self.state['iter_num']) / 3600:.2f} hours. "
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.cli_logger.debug(f"Gradient norm before clipping: {grad_norm:.4f}")
        fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    def _accumulate_training(self, fabric: L.Fabric, model: L.LightningModule, batch: Tuple[torch.Tensor, ...], step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform training step with gradient accumulation.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - batch (tuple): A batch of training data.
        - step (int): The current training step.

        Returns:
        - tuple: Contains the outputs from the training step and the loss tensor.
        """
        # Debug logging for gradient accumulation steps
        gradient_accumulation_steps_raw = self.config.gradient_accumulation_steps
        self.cli_logger.debug(f"gradient_accumulation_steps raw value: {gradient_accumulation_steps_raw}, type: {type(gradient_accumulation_steps_raw)}")
        
        gradient_accumulation_steps = int(self.config.gradient_accumulation_steps)
        self.cli_logger.debug(f"gradient_accumulation_steps converted: {gradient_accumulation_steps}, type: {type(gradient_accumulation_steps)}")
        self.cli_logger.debug(f"iter_num value: {self.state['iter_num']}, type: {type(self.state['iter_num'])}")
        self.cli_logger.debug(f"About to check: (iter_num + 1) % gradient_accumulation_steps != 0 -> ({self.state['iter_num']} + 1) % {gradient_accumulation_steps} != 0")
        
        is_accumulating = (self.state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            real_loss = (loss / gradient_accumulation_steps) if is_accumulating else loss
            fabric.backward(real_loss)
        if not is_accumulating:
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            self._gradient_clipping(fabric, model, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.state["step_count"] += 1
            self._try_validate(fabric)
        self.state["iter_num"] += 1
        return outputs, loss
    
    def _try_validate(self, fabric: L.Fabric, epochFinished: bool = False, trainingFinished: bool = False) -> None:
        """
        Determine whether to run validation based on configured conditions and perform validation if necessary.

        This method checks various conditions including:
          - Running every 'k' training steps (if specified)
          - At the end of an epoch
          - At the end of training

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - epochFinished (bool): Flag indicating if the current epoch has finished.
        - trainingFinished (bool): Flag indicating if training has completed.
        """
        validate_after_k_steps = self.config.get("validate_after_k_steps", None)
        validate_on_end = self.config.get("validate_on_end", True)
        validate_after_epoch = self.config.get("validate_after_epoch", True)
        save_on_validate = self.config.get("save_on_validate", True)
        

        if validate_after_k_steps is not None:
            # Debug logging for type issues
            self.cli_logger.debug(f"validate_after_k_steps value: {validate_after_k_steps}, type: {type(validate_after_k_steps)}")
            self.cli_logger.debug(f"step_count value: {self.state['step_count']}, type: {type(self.state['step_count'])}")
            validate_after_k_steps = int(validate_after_k_steps)
        
        # Debug logging before the condition check
        self.cli_logger.debug(f"Validation condition check - epochFinished: {epochFinished}, trainingFinished: {trainingFinished}")
        if validate_after_k_steps is not None:
            self.cli_logger.debug(f"About to check: step_count % validate_after_k_steps == 0 -> {self.state['step_count']} % {validate_after_k_steps} == 0")
        
        steps_condition = epochFinished == False and trainingFinished == False and (validate_after_k_steps is not None and self.state["step_count"] % validate_after_k_steps == 0)
        epoch_condition = validate_after_epoch and epochFinished
        end_condition = validate_on_end and trainingFinished
        
        if steps_condition or epoch_condition or end_condition:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            fabric.barrier()
            if 'valid' in self.dataloaders.keys():
                self._validate(fabric)
                
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if save_on_validate:
                self._save(fabric, epochFinished, trainingFinished)
    
    def _normal_training(self, fabric: L.Fabric, model: L.LightningModule, batch: Tuple[torch.Tensor, ...], step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a standard training step without gradient accumulation.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - batch (tuple): A batch of training data.
        - step (int): The current training step.

        Returns:
        - tuple: Contains the outputs from the training step and the loss tensor.
        """      
        
        with self.autocast_context():
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            gradient_accumulation_steps = int(self.config.gradient_accumulation_steps)
            fabric.backward(loss / gradient_accumulation_steps)
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            self._gradient_clipping(fabric, model, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.state["step_count"] += 1
            
            self._try_validate(fabric)
            self.state["iter_num"] += 1
            return outputs, loss
            
    def _train(self, fabric: L.Fabric) -> None:
        """
        Execute the main training loop over the specified number of epochs.

        This method iterates over the training dataset, handling both gradient accumulation and normal training,
        resuming from checkpoints when applicable, and logging training progress as well as performing validation.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance driving the training.
        """
        model = self.state["model"]
        self.total_lengths = 0
        self.train_total_t0 = time.perf_counter()
        self.initial_iter = self.state["iter_num"]
        epochs = self.config.number_epochs
        self.model.train()
        resume_iter = self.state["iter_num"]
        save_on_end = self.config.get("save_on_end", False)
        save_on_validate = self.config.get("save_on_validate", False)
        
        # Ensure current_epoch is initialized (for backwards compatibility with old checkpoints)
        if "current_epoch" not in self.state:
            self.state["current_epoch"] = 0
        
        for epoch in range(epochs):
            # Update current epoch in state for checkpoint naming
            self.state["current_epoch"] = epoch + 1
            
            if fabric.global_rank == 0:
                self.cli_logger.debug(f"Running Epoch {epoch + 1} of {epochs}")
            batch_iterator = tqdm(self.dataloaders['train'], mininterval=0, colour="blue") \
                if fabric.global_rank == 0 else self.dataloaders['train']
            batch_iterator, resume_iter = self._get_resume_iterator(batch_iterator, resume_iter)
            if batch_iterator is None:
                continue            
            for step, batch in enumerate(batch_iterator):
                self.train_iter_t0 = time.perf_counter()
                if self.config.gradient_accumulation_steps:
                    _, loss = self._accumulate_training(fabric, model, batch, step)
                else:
                    _, loss = self._normal_training(fabric, model, batch, step)
                self.total_lengths += batch["input_ids"].size(1)
                self.train_t1 = time.perf_counter()
                self._train_logs(fabric, loss)
                
            self._try_validate(fabric, epochFinished=True)
        self._try_validate(fabric, trainingFinished=True)

        # Only save at end if not already saved by validation at end
        if save_on_end and not (self.config.get("validate_on_end", True) and save_on_validate):
            self.cli_logger.info("Saving final checkpoint")
            self._save(fabric, trainingFinished=True)
    
    @torch.no_grad()
    def _validate(self, fabric: L.Fabric) -> None:
        """
        Validate the model on the validation dataset.

        This method switches the model to evaluation mode, processes the validation data, computes
        the mean loss, logs the validation metrics, and synchronizes across processes.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        """       
        
        t0 = time.perf_counter()
        self.model.eval()
        losses = []
        
        batch_iterator = tqdm(
            self.dataloaders['valid'],
            desc="Validating...",
            mininterval=0,
            colour="green"
        )
        for k, val_data in enumerate(batch_iterator):
            validation_output = self.model.validation_step(val_data, k)
            loss = validation_output["loss"]
            losses.append(loss.detach())
            
        out = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=fabric.device)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        self.monitor.eval_end(t1)
        def fabric_eval_log(loss):
            self.cli_logger.info(f"step {self.state['iter_num']}: val loss {loss:.4f}, val time: {elapsed_time * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": loss.item()}, self.state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(loss.item())}, self.state["step_count"])
        fabric_eval_log(out)
        fabric.barrier()
      
    def _load_fabric_datasets_dataloaders(self, config: Box, dataset: Union[HFDataset, DatasetDict]) -> dict[DatasetDict, Union[dict[str, DataLoader], DataLoader]]:
        """
        Load datasets and create dataloaders from the given dataset and configuration.

        This method validates the dataset, sets the required format, and creates DataLoader objects for each split.

        Parameters:
        - config (Box): Configuration parameters including batch_size and num_workers.
        - dataset (Union[HFDataset, DatasetDict]): The dataset or dictionary of datasets to use.

        Returns:
        - dict: A dictionary containing the processed datasets and corresponding dataloaders.

        Raises:
        - TypeError: If the dataset is not a DatasetDict or HFDataset.
        - ValueError: If required config parameters or dataset splits/columns are missing.
        - RuntimeError: If setting the format or creating a DataLoader fails.
        """        
        
        if not isinstance(dataset, Union[DatasetDict, HFDataset]):
            raise TypeError("Expected dataset to be a DatasetDict or Dataset")
        if not hasattr(config, 'batch_size') or not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise ValueError("config.batch_size must be a positive integer")
        if not hasattr(config, 'num_workers') or not isinstance(config.num_workers, int) or config.num_workers < 0:
            raise ValueError("config.num_workers must be a non-negative integer")
        
        # TODO: reformat this logic to be more maintainable
        if isinstance(dataset, HFDataset):
            dataset = DatasetDict({"train": dataset})
        if not dataset.keys():
            raise ValueError("Dataset is empty, no splits found")
        required_columns = ["input_ids", "attention_mask", "labels"]
        for split in dataset.keys():
            missing_columns = [col for col in required_columns if col not in dataset[split].column_names]
            if missing_columns:
                raise ValueError(f"Missing required columns {missing_columns} in {split} split")
            try:
                dataset[split].set_format(type="torch", columns=required_columns)
            except Exception as e:
                raise RuntimeError(f"Failed to set format for {split} split: {str(e)}")
        dataloaders = {}
        for split in dataset.keys():
            try:
                dataloaders[split] = DataLoader(
                    dataset[split], 
                    batch_size=config.batch_size, 
                    shuffle=(split == "train"), 
                    num_workers=config.num_workers,
                    pin_memory=True,
                    drop_last=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader for {split} split: {str(e)}")
        return {
            "datasets": dataset,
            "dataloaders": dataloaders
        }
        
    def _pipeline(self, fabric: L.Fabric) -> None:
        """
        Orchestrate the complete training pipeline.

        This method sets deterministic seeds if provided, sets up monitoring and log directories,
        prepares dataloaders for Fabric, instantiates and configures the model (including gradient checkpointing),
        sets up the optimizer and scheduler, loads from a checkpoint if available, and finally starts training.

        Parameters:
        - fabric (L.Fabric): The Fabric instance coordinating distributed training.
        """
        # DETERMINISTIC RESULTS
        if self.config.get("seed", None) is not None:
            deterministic(self.config.seed)
            fabric.seed_everything(self.config.seed)

        # MONITORING
        self.monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=self.config.log_iter_interval)

        # OUTPUT DIR AND SYNC
        if fabric.global_rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
        fabric.barrier()

        # FABRIC DATALOADERS SETUP
        self.dataloaders = {k: fabric.setup_dataloaders(v) for k, v in self.dataloaders.items()}

        # MODEL: instantiate within the fabric.init_module() context
        t0 = time.perf_counter()
        with fabric.init_module():
            self.model = FabricGeneration(**self.config)
            # Properly set up the model with fabric for FSDP
            self.model = fabric.setup(self.model)

        # GRADIENT CHECKPOINTING
        if self.config.gradient_checkpointing:
            self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                "use_reentrant": False
            })
        else:
            self.model.model.gradient_checkpointing_disable()

        self.cli_logger.info(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        # OPTIMIZER
        optimizer = select_optimizer(
            self.config.get("optimizer", "adamw"), 
            self.model, 
            self.config.lr, 
            self.config.weight_decay, 
            self.config.beta1, 
            self.config.beta2
        )
        optimizer = fabric.setup_optimizers(optimizer)
        # SCHEDULER
        scheduler = select_scheduler(
            optimizer, 
            self.config.lr_scheduler, 
            self.config.number_epochs, 
            fabric.world_size, 
            self.config.batch_size, 
            self.dataset['train'], 
            self.config.warmup_proportion, 
            self.config.gradient_accumulation_steps
        )        # STATE
        self.state = {
            "model": self.model, 
            "optimizer": optimizer, 
            "hparams": self.hparams, 
            "iter_num": 0, 
            "step_count": 0, 
            "current_epoch": 0,
            "scheduler": scheduler
        }
        # RESUME
        self._load_from_checkpoint(fabric)
        # TRAINING
        train_time = time.perf_counter()
        self._train(fabric)
        self.cli_logger.info(f"Training time: {(time.perf_counter() - train_time):.2f}s")
        if fabric.device.type == "cuda":
            self.cli_logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")