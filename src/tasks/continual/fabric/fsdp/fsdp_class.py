import math
import time
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
# Importando Lightning y otras librerías necesarias
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from pytorch_lightning.loggers import WandbLogger

# Importando utilidades personalizadas
from src.tasks.continual.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.continual.fabric.logger import step_csv_logger
from src.tasks.continual.utils import *
from src.tasks.continual.distributed_model_classes import FabricGeneration



class Fabric_Abstract(ABC):
    def __init__(self, devices, config, resume=False):
        self.devices = devices
        self.config = config
        self.resume = resume
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
        checkpoint_path = self.config.output_dir / f"iter-{self.state['iter_num']:06d}-ckpt.pth"
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, self.state)
    
    
    def _resuming(self, fabric) -> bool:
        if self.resume:
            if self.curr_iter < self.initial_iter:
                self.curr_iter += 1
                return True  # Skip this iteration
            else:
                self.resume = False
                self.curr_iter = -1
                fabric.barrier()
                elapsed = time.perf_counter() - self.train_total_t0
                fabric.print(f"Resume finished, taken {elapsed:.2f} seconds")
        return False
    
    def _load_resume(self, fabric):
        if self.resume is True:
            self.resume = sorted(self.config.output_dir.glob("*.pth"))[-1]
        if self.resume:
            fabric.print(f"Resuming training from {self.resume}")
            fabric.load(self.resume, self.state)
            
            
    def _train_logs(self, fabric, loss):
        fabric.print(
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
    
    def _gradient_clipping(self, fabric, model, optimizer):
        """Clip gradients during training to avoid exploding gradients."""
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        fabric.print(f"Gradient norm before clipping: {grad_norm:.4f}")
        fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    
    def _accumulate_training(self, fabric, model, batch, step):
        """
            Accumulate gradients over multiple steps before backpropagating.
            Automatically called by train method if gradient_accumulation_steps > 1.
        """
        is_accumulating = (self.state["iter_num"] + 1) % self.config.gradient_accumulation_steps != 0
        
        # FORWARD PASS
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            outputs, loss = model.training_step(batch, step)
            fabric.backward(loss / self.config.gradient_accumulation_steps)

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
    
    
    def _normal_training(self, fabric, model, batch, step):
        
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
                    self._validate(fabric, model, self.dataloaders['valid'], self.config, self.monitor)
                
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
        self.curr_iter = 0
        epochs = self.config.number_epochs

        self.model.train()
        
        # TRAINING LOOP
        for epoch in range(epochs):
            if fabric.global_rank == 0:
                print(f"Running Epoch {epoch + 1} of {epochs}")
            
            batch_iterator = tqdm(self.dataloaders['train'], mininterval=0, colour="blue") if fabric.global_rank == 0 else self.dataloaders['train']
            
            # TODO: todo esto en una funcion
            epoch_batch_count = len(self.dataloaders['train'])
            if resume_iter >= epoch_batch_count:
                # This entire epoch was already processed.
                resume_iter -= epoch_batch_count
                continue
            elif resume_iter > 0:
                # Slice the iterator to start from the resume batch.
                batch_iterator = itertools.islice(batch_iterator, resume_iter, None)
                resume_iter = 0  # Reset after the first partial epoch
            
            #batch_iterator = self._resuming(fabric, batch_iterator)
            
            
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
    def _validate(self, fabric: L.Fabric) -> torch.Tensor:
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
            fabric.print(f"step {self.state['iter_num']}: val loss {loss:.4f}, val time: {elapsed_time * 1000:.2f}ms")
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

        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")


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
        self._load_resume(fabric)


        # TRAINING
        train_time = time.perf_counter()
        self._train(fabric)
        fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")

        if fabric.device.type == "cuda":
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    

class FSDP(Fabric_Abstract):
    def __init__(self, devices, config, resume=False):
        super().__init__(devices, config, resume)
        
        self.devices = devices
        self.config = config
        self.resume = resume
        
    def _setup_strategy(self):
        if self.devices > 1:
            # FSDP strategy for multiple devices
            strategy = FSDPStrategy(
                sharding_strategy=self.config.sharding_strategy,
                auto_wrap_policy=self.config.auto_wrap_policy,
                activation_checkpointing_policy=self.config.auto_wrap_policy,
                state_dict_type=self.config.state_dict_type,
                limit_all_gathers=self.config.limit_all_gathers,
                cpu_offload=self.config.cpu_offload,
            )
        else:
            strategy = "auto"
            # TODO: Poner en formato de warning
            print("Using automatic strategy for 1 device.")
            raise NotImplementedError("Automatic strategy is not yet implemented for 1 device.")
    
        return strategy            
    
    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(devices=self.devices, strategy=strategy, precision=self.config.precision, loggers=[loggers])
        
        self.hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
        fabric.print(self.hparams)
        
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
    