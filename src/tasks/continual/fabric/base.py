import math
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
# Importando Lightning y otras librerías necesarias
import lightning as L
from pytorch_lightning.loggers import WandbLogger

# Importando utilidades personalizadas
from src.tasks.continual.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.continual.fabric.logger import step_csv_logger
from src.tasks.continual.utils import *
from src.tasks.continual.fabric.generation import FabricGeneration
from utils.logging import get_logger
from datasets import Dataset


class FabricTrainerBase(ABC):
    def __init__(self, devices, config, dataset: Dataset, checkpoint_path: str = None):
        self.cli_logger = get_logger(__name__, config.verbose_level)
        
        if (dataset is None):
            raise ValueError("Dataset must be provided for training.")
        
        self.devices = devices
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.state = {}
        self.dataset, self.dataloaders = self._load_fabric_datasets_dataloaders(self.config)
    
    @abstractmethod
    def _setup_strategy(self):
        pass
    
    @abstractmethod
    def setup(self) -> None:
        pass
    
    
    def _set_loggers(self):
        logger = step_csv_logger(self.config.output_dir, 
                                self.config.model_name, 
                                flush_logs_every_n_steps=self.config.log_iter_interval)
        
        if self.config.logging_config == "wandb":
        
            wandb_logger = WandbLogger(entity=self.config.wandb_entity, 
                                    project=self.config.wandb_project, 
                                    log_model=self.config.log_model)
            
            return logger, wandb_logger

        return [logger]
    
    def _save(self, fabric) -> None:
        output_checkpoint_path = self.config.output_dir / f"iter-{self.state['iter_num']:06d}-ckpt.pth"
        self.cli_logger.debug(f"Saving checkpoint to {str(output_checkpoint_path)!r}")
        fabric.save(output_checkpoint_path, self.state)
    
    
    def _get_resume_iterator(self, iterator, resume_iter):
        """
        Given an iterator (for one epoch) and the number of batches already processed (resume_iter),
        return a tuple (new_iterator, updated_resume_iter).
        
        If resume_iter is greater than the number of batches in the iterator,
        returns (None, resume_iter - number_of_batches) indicating the entire epoch should be skipped.
        Otherwise, returns an iterator starting from resume_iter and resets resume_iter to 0.
        """
        epoch_batch_count = len(iterator)
        if resume_iter >= epoch_batch_count:
            return None, resume_iter - epoch_batch_count
        elif resume_iter > 0:
            return itertools.islice(iterator, resume_iter, None), 0
        else:
            return iterator, resume_iter
    
    def _load_from_checkpoint(self, fabric):
        if not self.checkpoint_path is None:
            self.cli_logger.debug(f"Resuming training from '{self.checkpoint_path}'")
            fabric.load(self.checkpoint_path, self.state)
            
            
    def _train_logs(self, fabric: L.Fabric, loss):
        self.cli_logger.info(
                    f"iter {self.state['iter_num']} step {self.state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(self.train_t1 - self.train_iter_t0) * 1000:.2f}ms remaining time: "
                    f"{(self.train_t1 - self.train_total_t0) / (self.state['iter_num'] - self.initial_iter) * (self.config.max_iters - self.state['iter_num']) / 3600:.2f} hours. "
                )
        
        self.monitor.on_train_batch_end(
                    self.state["iter_num"] * self.config.micro_batch_size,
                    self.train_t1 - self.train_total_t0,
                    fabric.world_size,
                    self.state["step_count"],
                    lengths=self.total_lengths,
                    train_loss=loss.item()
                )
    
    def _gradient_clipping(self, fabric: L.Fabric, model, optimizer):
        """Clip gradients during training to avoid exploding gradients."""
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.cli_logger.info(f"Gradient norm before clipping: {grad_norm:.4f}")
        fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    
    def _accumulate_training(self, fabric: L.Fabric, model, batch, step):
        """
            Accumulate gradients over multiple steps before backpropagating.
            Automatically called by train method if gradient_accumulation_steps > 1.
        """
        is_accumulating = (self.state["iter_num"] + 1) % self.config.gradient_accumulation_steps != 0
        
        # FORWARD PASS
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            outputs, loss = model.training_step(batch, step)
            real_loss = (loss / self.config.gradient_accumulation_steps) if is_accumulating else loss
            fabric.backward(real_loss)

        # BACKPROPAGATION
        if not is_accumulating:
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            
            # Log the gradient norms before clipping for monitoring
            self._gradient_clipping(fabric, model, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.state["step_count"] += 1
            
            if self.state["step_count"] % self.config.eval_steps == 0:
                fabric.barrier()
                if 'valid' in self.dataloaders.keys():
                    self._validate(fabric)
                
                if 'test' in self.dataloaders.keys():
                    #TODO: Implement test method
                    raise NotImplementedError("Test method not implemented.")
                self._save(fabric)
                
        self.state["iter_num"] += 1
        
        return outputs, loss
    
    
    def _normal_training(self, fabric: L.Fabric, model, batch, step):
        
        """
        Performs the usual forward pass, backward pass and optimization step.
        Automatically called by train method if gradient_accumulation_steps == 1.
        """
        
        outputs, loss = model.training_step(batch, step)
        fabric.backward(loss / self.config.gradient_accumulation_steps)
        
        optimizer = self.state["optimizer"]
        scheduler = self.state["scheduler"]
        
        self._gradient_clipping(fabric, model, optimizer)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        self.state["step_count"] += 1
        
        if self.state["step_count"] % self.config.eval_steps == 0:
            fabric.barrier()
            if 'valid' in self.dataloaders.keys():
                    self._validate(fabric)
                
            if 'test' in self.dataloaders.keys():
                #TODO: Implement test method
                raise NotImplementedError("Test method not implemented.")            
            self._save(fabric)
        
        self.state["iter_num"] += 1
        
        return outputs, loss
            
    
    def _train(self, fabric):
        model = self.state["model"]
        self.total_lengths = 0
        self.train_total_t0 = time.perf_counter()
        self.initial_iter = self.state["iter_num"]
        epochs = self.config.number_epochs
        self.model.train()
        
        
        # resume_iter holds the number of batches already processed in total.
        resume_iter = self.state["iter_num"]
        
        # TRAINING LOOP
        for epoch in range(epochs):
            if fabric.global_rank == 0:
                print(f"Running Epoch {epoch + 1} of {epochs}")
            
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


                # LOGS
                self._train_logs(fabric, loss)
                
        
            # EPOCH END - VALIDATION
            fabric.barrier()
            if 'valid' in self.dataloaders.keys():
                    self._validate(fabric)
                
            if 'test' in self.dataloaders.keys():
                #TODO: Implement test method
                raise NotImplementedError("Test method not implemented.")

            # EPOCH END - SAVE CHECKPOINT
            self._save(fabric)


    @torch.no_grad()
    def _validate(self, fabric: L.Fabric) -> None:
        t0 = time.perf_counter()
        self.model.eval()

        losses = []  # Lista para acumular pérdidas de forma dinámica
        batch_iterator = tqdm(
            self.dataloaders['valid'],
            desc="Validating...",
            mininterval=0,
            colour="green"
        )

        for k, val_data in enumerate(batch_iterator):
            outputs = self.model.validation_step(val_data, k)
            losses.append(outputs.loss.detach())

        if losses:
            out = torch.mean(torch.stack(losses))
        else:
            out = torch.tensor(0.0, device=fabric.device)
            
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        self.monitor.eval_end(t1)
        
        def fabric_eval_log(loss):
            self.cli_logger.info(f"step {self.state['iter_num']}: val loss {loss:.4f}, val time: {elapsed_time * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": loss.item()}, self.state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(loss.item())}, self.state["step_count"])
        
        fabric_eval_log(out)
        fabric.barrier()
    
    
    def _load_fabric_datasets_dataloaders(self, config):
        if config.data_dir is None:
            # TODO: Añadir funcionalidad para lidiar con datasets de HuggingFace
            raise ValueError("train_data_dir must be specified.")
        
        else:
            dataset = load_from_disk(config.data_dir)
            
            for split in dataset.keys():
                dataset[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataloaders = {}
        for split in dataset.keys():
            dataloaders[split] = DataLoader(dataset[split], batch_size=config.batch_size, shuffle=(split == "train"), num_workers=config.num_workers)
        
        return ((dataset), (dataloaders))
    
    
    def _pipeline(self, fabric):
            # DETERMINISTIC RESULTS
        if self.config.seed:
            setup_environment(self.config.seed)
            fabric.seed_everything(self.config.seed)
        
        # MONITORING
        self.monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=self.config.log_iter_interval)


        # OUTPUT DIR AND SYNC
        if fabric.global_rank == 0:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        fabric.barrier()

        # FABRIC DATALOADERS SETUP
        self.dataloaders = fabric.setup_dataloaders(self.dataloaders)
        

        # MODEL
        t0 = time.perf_counter()
        with fabric.init_module():
            self.model = FabricGeneration(self.config) 
        self.model = fabric.setup(self.model)
        
        # GRADIENT CHECKPOINTING
        if self.config.gradient_checkpointing:
            self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
        else:
            self.model.model.gradient_checkpointing_disable()

        self.cli_logger.info(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")


        # OPTIMIZER
        optimizer = select_optimizer(
            self.config.optimizer, 
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
            self.config.micro_batch_size, 
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
    
