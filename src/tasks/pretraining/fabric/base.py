

import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools

# Import Lightning and other necessary libraries
import lightning as L
from pytorch_lightning.loggers import WandbLogger
import os

# Import custom utilities
from src.tasks.pretraining.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.pretraining.fabric.logger import step_csv_logger
from src.tasks.pretraining.utils import *
from src.tasks.pretraining.fabric.generation import FabricGeneration
from utils.logging import get_logger

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FabricTrainerBase(ABC):
    def __init__(self, devices, config, dataset: Dataset, checkpoint_path: str = None):
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
    def _setup_strategy(self):
        pass
        
    def setup(self) -> None:
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

    def _set_loggers(self):
        logger = step_csv_logger(
            self.config.output_dir, 
            self.config.model_name, 
            flush_logs_every_n_steps=self.config.log_iter_interval
        )
        
        if self.config.logging_config == "wandb":
            wandb_logger = WandbLogger(
                entity=self.config.wandb_entity, 
                project=self.config.wandb_project, 
                log_model=self.config.log_model
            )
            return [logger, wandb_logger]
        return [logger]
    
    def _save(self, fabric) -> None:
        if self.config.output_dir is None:
            self.cli_logger.warning("Output directory not provided. Skipping checkpoint saving.")
            return
        
        output_checkpoint_path = Path(self.config.output_dir,  f"iter-{self.state['iter_num']:06d}-ckpt.pth")
        self.cli_logger.debug(f"Saving checkpoint to {str(output_checkpoint_path)!r}")
        fabric.save(output_checkpoint_path, self.state)
    
    def _get_resume_iterator(self, iterator, resume_iter):
        epoch_batch_count = len(iterator)
        if resume_iter >= epoch_batch_count:
            return None, resume_iter - epoch_batch_count
        elif resume_iter > 0:
            return itertools.islice(iterator, resume_iter, None), 0
        else:
            return iterator, resume_iter
    
    def _load_from_checkpoint(self, fabric):
        if self.checkpoint_path is not None:
            self.cli_logger.debug(f"Resuming training from '{self.checkpoint_path}'")
            fabric.load(self.checkpoint_path, self.state)
    
    def _train_logs(self, fabric: L.Fabric, loss):
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
    
    def _gradient_clipping(self, fabric: L.Fabric, model, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.cli_logger.debug(f"Gradient norm before clipping: {grad_norm:.4f}")
        fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    def _accumulate_training(self, fabric: L.Fabric, model, batch, step):
        is_accumulating = (self.state["iter_num"] + 1) % self.config.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            real_loss = (loss / self.config.gradient_accumulation_steps) if is_accumulating else loss
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
    
    def _try_validate(self, fabric: L.Fabric, epochFinished: bool = False, trainingFinished: bool = False):
        validate_after_k_steps = self.config.get("validate_after_k_steps", None)
        validate_on_end = self.config.get("validate_on_end", False)
        validate_after_epoch = self.config.get("validate_after_epoch", False)
        
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
                
            self._save(fabric)
    
    def _normal_training(self, fabric: L.Fabric, model, batch, step):
        with self.autocast_context():
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            fabric.backward(loss / self.config.gradient_accumulation_steps)
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
            
    def _train(self, fabric):
        model = self.state["model"]
        self.total_lengths = 0
        self.train_total_t0 = time.perf_counter()
        self.initial_iter = self.state["iter_num"]
        epochs = self.config.number_epochs
        self.model.train()
        resume_iter = self.state["iter_num"]
        for epoch in range(epochs):
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
    
    @torch.no_grad()
    def _validate(self, fabric: L.Fabric) -> None:
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
    
    def _load_fabric_datasets_dataloaders(self, config, dataset: Dataset | DatasetDict):
        if not isinstance(dataset, DatasetDict|Dataset):
            raise TypeError("Expected dataset to be a DatasetDict or Dataset")
        if not hasattr(config, 'batch_size') or not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise ValueError("config.batch_size must be a positive integer")
        if not hasattr(config, 'num_workers') or not isinstance(config.num_workers, int) or config.num_workers < 0:
            raise ValueError("config.num_workers must be a non-negative integer")
        
        # TODO: reformat this logic to be more maintainable
        if isinstance(dataset, Dataset):
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
    
    def _pipeline(self, fabric):
        # DETERMINISTIC RESULTS
        if self.config.get("seed", None) is not None:
            setup_environment(self.config.seed)
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
        )
        # STATE
        self.state = {
            "model": self.model, 
            "optimizer": optimizer, 
            "hparams": self.hparams, 
            "iter_num": 0, 
            "step_count": 0, 
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