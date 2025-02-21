import math
import time
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

# Importando Lightning y otras librerías necesarias
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from pytorch_lightning.loggers import WandbLogger

# Importando utilidades personalizadas
from src.tasks.continual.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.continual.fabric.logger import step_csv_logger
from src.tasks.continual.utils import *
from src.tasks.continual.distributed_model_classes import FabricGeneration


def set_loggers(config):
    logger = step_csv_logger(config.output_dir, config.model_name, flush_logs_every_n_steps=config.log_iter_interval)
    wandb_logger = WandbLogger(entity=config.wandb_entity, project=config.wandb_project, log_model=config.log_model)

    return logger, wandb_logger

def setup(devices: int, config, resume: Union[bool, Path] = False) -> None:
    """
    Environment configuration for distributed training with Fabric.
    """
    if devices > 1:
        # FSDP strategy for multiple devices
        strategy = FSDPStrategy(
            sharding_strategy=config.sharding_strategy,
            auto_wrap_policy=config.auto_wrap_policy,
            activation_checkpointing_policy=config.auto_wrap_policy,
            state_dict_type=config.state_dict_type,
            limit_all_gathers=config.limit_all_gathers,
            cpu_offload=config.cpu_offload,
        )
    else:
        strategy = "auto"
        # TODO: Poner en formato de warning
        print("Using automatic strategy for 1 device.")
        raise NotImplementedError("Automatic strategy is not yet implemented for 1 device.")
    
    
    logger, wandb_logger = set_loggers(config)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=config.precision, loggers=[logger, wandb_logger])
    
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    fabric.print(hparams)
    
    fabric.launch(main, resume, config, hparams)
    
    
def main(fabric, resume, config, hparams):
    # DETERMINISTIC RESULTS
    if config.seed:
        setup_environment(config.seed)
        fabric.seed_everything(config.seed)
    
    # MONITORING
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=config.log_iter_interval)


    # OUTPUT DIR AND SYNC
    if fabric.global_rank == 0:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    fabric.barrier()


    # DATASETS
    dataset = load_from_disk(config.train_data_dir)
    train_dataset = dataset["train"]
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    validation_dataset = dataset["valid"]
    validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


    # DATALOADERS
    train_dataloader = DataLoader(train_dataset, batch_size=config.micro_batch_size, shuffle=True, num_workers=config.num_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers)


    # FABRIC DATALOADERS SETUP
    train_dataloader, validation_dataloader = fabric.setup_dataloaders(train_dataloader, validation_dataloader)


    # MODEL
    t0 = time.perf_counter()
    with fabric.init_module():
        model = FabricGeneration(config) 
    model = fabric.setup(model)
    
    # GRADIENT CHECKPOINTING
    if config.gradient_checkpointing:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
    else:
        model.model.gradient_checkpointing_disable()

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")


    # OPTIMIZER
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), foreach=True
    )
    optimizer = fabric.setup_optimizers(optimizer)
    
    
    # SCHEDULER
    scheduler = select_scheduler(optimizer, config.lr_scheduler, config.number_epochs, fabric.world_size, config.micro_batch_size, train_dataset, config.warmup_proportion, config.gradient_accumulation_steps)


    # STATE
    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0, "scheduler": scheduler}


    # RESUME
    if resume is True:
        resume = sorted(config.output_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)


    # TRAINING
    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, validation_dataloader, monitor, resume, config)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
        
def train(fabric, state, train_dataloader, validation_dataloader, monitor, resume, config):
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    total_lengths = 0
    total_t0 = time.perf_counter()
    initial_iter = state["iter_num"]
    curr_iter = 0
    epochs = config.number_epochs

    # TRAINING LOOP
    for epoch in range(epochs):
        if fabric.global_rank == 0:
            print(f"Running Epoch {epoch + 1} of {epochs}")
        batch_iterator = tqdm(train_dataloader, mininterval=0, colour="blue") if fabric.global_rank == 0 else train_dataloader
        
        
        for step, batch in enumerate(batch_iterator):
            
            # GETTING TO THE RESUME POINT
            if resume:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                else:
                    resume = False
                    curr_iter = -1
                    fabric.barrier()
                    fabric.print("Resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
            if state["iter_num"] >= config.max_iters:
                break

            iter_t0 = time.perf_counter()
            is_accumulating = (state["iter_num"] + 1) % config.gradient_accumulation_steps != 0


            # FORWARD PASS
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs, loss = model.training_step(batch, step)
                fabric.backward(loss / config.gradient_accumulation_steps)


            # BACKPROPAGATION
            if not is_accumulating:
                # Log the gradient norms before clipping for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                fabric.print(f"Gradient norm before clipping: {grad_norm:.4f}")
                fabric.clip_gradients(model, optimizer, max_norm=config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state["step_count"] += 1
                
                if state["step_count"] % config.eval_steps == 0:
                    fabric.barrier()
                    validate(fabric, model, validation_dataloader, config, monitor)
                    
                    checkpoint_path = config.output_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                    fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                    fabric.save(checkpoint_path, state)
                    

            state["iter_num"] += 1
            total_lengths += batch["input_ids"].size(1)
            t1 = time.perf_counter()

            # LOGS
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms remaining time: "
                f"{(t1 - total_t0) / (state['iter_num'] - initial_iter) * (config.max_iters - state['iter_num']) / 3600:.2f} hours. "
            )

            monitor.on_train_batch_end(
                state["iter_num"] * config.micro_batch_size,
                t1 - total_t0,
                fabric.world_size,
                state["step_count"],
                lengths=total_lengths,
                train_loss=loss.item()
            )
    
        # EPOCH END - VALIDATION
        fabric.barrier()
        validate(fabric, model, validation_dataloader, config, state, monitor)

        # EPOCH END - SAVE CHECKPOINT
        checkpoint_path = config.output_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, validation_dataloader: DataLoader, config, state, monitor) -> torch.Tensor:
    t0 = time.perf_counter()
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(config.eval_iters, device=fabric.device)
    batch_iterator = tqdm(validation_dataloader, desc=f"Running Epoch 1 of 1", mininterval=0, colour="green")

    for k, val_data in enumerate(batch_iterator):
        if k >= config.eval_iters:
            break
        outputs = model.validation_step(val_data, k)
        losses[k] = outputs.loss

    out = losses.mean()
    model.train()
    t1 = time.perf_counter() - t0
    monitor.eval_end(t1)
    
    def fabric_eval_log(fabric, state, loss):
        fabric.print(f"step {state['iter_num']}: val loss {loss:.4f}, val time: {t1 * 1000:.2f}ms")
        fabric.log_dict({f"metric/val_loss": loss.item()}, state["step_count"])
        fabric.log_dict({f"metric/val_ppl": math.exp(loss.item())}, state["step_count"])
    
    fabric_eval_log(fabric, state, out)
    fabric.barrier()
