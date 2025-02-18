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
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from pytorch_lightning.loggers import WandbLogger

# Importando utilidades personalizadas
from utils.speed_monitor import SpeedMonitorFabric as Monitor
from utils.logger import step_csv_logger
from models.models_class import FabricGeneration


# Configurar loggers
hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", args["model_name"], flush_logs_every_n_steps=args['log_iter_interval'])
wandb_logger = WandbLogger(entity=args['entity'], project=args['project'], log_model=args['log_model'])

def setup(devices: int, config, resume: Union[bool, Path] = False) -> None:
    """
    Configura el entorno para el entrenamiento distribudo.
    """
    
    # Configurar la estrategia según el número de dispositivos y si se usa TPU
    if devices > 1:
        # Estrategia FSDP para distribución de los parámetros y gradientes
        strategy = FSDPStrategy(
            sharding_strategy=config.sharding_strategy,
            auto_wrap_policy=config.auto_wrap_policy,
            activation_checkpointing_policy=config.auto_wrap_policy,
            state_dict_type=config.state_dict_type,
            limit_all_gathers=config.limit_all_gathers,
            cpu_offload=config.cpu_offload,
        )
    else:
        strategy = "auto"  # Uso automático de estrategia para 1 dispositivo

    # Configurar Fabric con los dispositivos, estrategia, precisión y loggers
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=config.precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    
    # Lanzar la función principal de entrenamiento
    fabric.launch(main, resume, config)
    
    
def main(fabric, resume, config):
    """
    Función principal para entrenamiento de modelos grandes con técnicas de paralelismo distribuido.
    """
    # Monitor para registrar el rendimiento
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args['log_iter_interval'])

    if fabric.global_rank == 0:
        args['out_dir'].mkdir(parents=True, exist_ok=True)
    fabric.barrier()  # Asegurarse de que todos los procesos esperen hasta que el directorio esté creado

    # Cargar el dataset
    dataset = load_from_disk(args["train_data_dir"])
    train_dataset = dataset["train"]
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    validation_dataset = dataset["valid"]
    validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Configurar los dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args['micro_batch_size'], shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args["eval_batch_size"], shuffle=False, num_workers=4)

    # Configurar los dataloaders con Fabric
    if validation_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, validation_dataloader)

    fabric.seed_everything(3407)  # Misma semilla para cada proceso

    # Inicializar el modelo
    t0 = time.perf_counter()
    with fabric.init_module():
        model = FabricGeneration(args) 
    model = fabric.setup(model)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    # Configurar el optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(args['beta1'], args['beta2']), foreach=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Definir el estado del entrenamiento
    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(args['out_dir'].glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # Comenzar el entrenamiento
    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, validation_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
        
def train(fabric, state, train_dataloader, validation_dataloader, monitor, resume):
    """
    Realiza el entrenamiento del modelo con las configuraciones dadas.
    """
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()
    initial_iter = state["iter_num"]
    curr_iter = 0
    epochs = args["number_epochs"]

    # Iterar sobre las épocas
    for epoch in range(epochs):
        if fabric.global_rank == 0:
            print(f"Running Epoch {epoch + 1} of {epochs}")
        batch_iterator = tqdm(train_dataloader, mininterval=0, colour="blue") if fabric.global_rank == 0 else train_dataloader
        eval_step_interval = initial_iter + int((len(batch_iterator) / 2) // args['gradient_accumulation_steps'])

        # Iterar sobre cada lote
        for step, batch in enumerate(batch_iterator):
            if resume:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                else:
                    resume = False
                    curr_iter = -1
                    fabric.barrier()
                    fabric.print("Resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
            if state["iter_num"] >= args['max_iters']:
                break

            # Determinar la tasa de aprendizaje actual
            lr = get_lr(state["iter_num"], args["lr"]) if args["decay_lr"] else args["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()
            is_accumulating = (state["iter_num"] + 1) % args["gradient_accumulation_steps"] != 0

            # Backpropagation con acumulación de gradientes
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model.training_step(batch, step)
                fabric.backward(loss / args["gradient_accumulation_steps"])

            if not is_accumulating:
                # Log the gradient norms before clipping for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                fabric.print(f"Gradient norm before clipping: {grad_norm:.4f}")
                fabric.clip_gradients(model, optimizer, max_norm=args["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1

            state["iter_num"] += 1
            total_lengths += batch["input_ids"].size(1)
            t1 = time.perf_counter()

            # Imprimir estado del entrenamiento
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms remaining time: "
                f"{(t1 - total_t0) / (state['iter_num'] - initial_iter) * (args['max_iters'] - state['iter_num']) / 3600:.2f} hours. "
            )

            monitor.on_train_batch_end(
                state["iter_num"] * args['micro_batch_size'],
                t1 - total_t0,
                fabric.world_size,
                state["step_count"],
                lengths=total_lengths,
                train_loss=loss.item()
            )

        # Validación al final de cada época
        t0 = time.perf_counter()
        val_loss = validate(fabric, model, validation_dataloader)
        t1 = time.perf_counter() - t0
        monitor.eval_end(t1)
        fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
        fabric.log_dict({"metric/val_loss": val_loss.item()}, state["step_count"])
        fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item())}, state["step_count"])
        fabric.barrier()

        # Guardar checkpoint
        checkpoint_path = args['out_dir'] / f"iter-{state['iter_num']:06d}-ckpt.pth"
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    """
    Función de validación para evaluar el rendimiento del modelo.
    """
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(args['eval_iters'], device=fabric.device)
    batch_iterator = tqdm(val_dataloader, desc=f"Running Epoch 1 of 1", mininterval=0, colour="green")

    for k, val_data in enumerate(batch_iterator):
        if k >= args['eval_iters']:
            break
        outputs = model.validation_step(val_data, k)
        losses[k] = outputs.loss

    out = losses.mean()
    model.train()
    return out


# Scheduler para el decay de la tasa de aprendizaje
def get_lr(it, learning_rate):
    if it < args['warmup_iters']:
        return learning_rate * it / args['warmup_iters']
    if it > args['lr_decay_iters']:
        return args['min_lr']
    decay_ratio = (it - args['warmup_iters']) / (args['lr_decay_iters'] - args['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args['min_lr'] + coeff * (learning_rate - args['min_lr'])