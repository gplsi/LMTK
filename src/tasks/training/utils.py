import os
import math
from typing import Optional, Union

from box import Box
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.optimization import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import numpy as np
import random
from datasets import Dataset as HFDataset

# TODO: Add more wrappers for other models, and make clear the keys for the wrappers

AUTO_WRAPPER = {
    "llama": LlamaDecoderLayer,
    "gpt2": GPT2Block
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adamax": torch.optim.Adamax,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop
}


# Scheduler for dealing with training with and without gradient accumulation
def select_scheduler(optimizer: torch.optim.Optimizer, lr_scheduler: str, number_epochs: int, world_size: int, batch_size: int, train_dataset: HFDataset, warmup_proportion: float, gradient_accumulation_steps: int = None) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Selects and returns an appropriate learning rate scheduler based on the specified configuration.
    
    This function supports three types of schedulers:
      - 'fixed': a constant learning rate scheduler.
      - 'warmup_constant': a scheduler with initial warmup followed by a constant learning rate.
      - 'warmup_linear': a scheduler with an initial warmup and subsequent linear decay.
      - 'warmup_cosine': a scheduler with an initial warmup and subsequent cosine decay.
      - 'warmup_cosine_restart': a scheduler with an initial warmup and subsequent cosine decay with hard restarts.
      - 'cosine': a pure cosine decay scheduler without warmup.
      
    It computes warmup steps and total training steps based on training dataset size, number of epochs, batch size, 
    world size, and optionally the number of gradient accumulation steps.
    
    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer instance used during training.
        lr_scheduler (str): The type of scheduler to use ('fixed', 'warmup_constant', or 'warmup_linear').
        number_epochs (int): Total number of epochs for training.
        world_size (int): Number of processing units (e.g., GPUs) used in distributed training.
        batch_size (int): Batch size per processing unit.
        train_dataset (HFDataset): The dataset used for training.
        warmup_proportion (float): Fraction of total training steps to use for warmup.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before an optimizer update.
            
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The configured learning rate scheduler.
        
    Raises:
        ValueError: If the scheduler type provided does not match any of the supported schedulers.
    """
    
    def calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset, gradient_accumulation_steps=None):
        """
        Calculates the number of warmup steps and total training steps based on the training configuration.
        
        The number of steps per epoch is computed by dividing the size of the training dataset by the product 
        of batch size and world size. Total steps are calculated by multiplying steps per epoch by the number of epochs,
        with an adjustment for gradient accumulation if applicable. Warmup steps are then determined as a fixed proportion 
        of the total training steps.
        
        Parameters:
            number_epochs (int): Total number of training epochs.
            world_size (int): Number of processing units (GPUs) used.
            batch_size (int): Batch size per unit.
            warmup_proportion (float): Proportion of steps allocated for warmup.
            train_dataset (HFDataset): The training dataset.
            gradient_accumulation_steps (int, optional): Number of gradient accumulation steps.
            
        Returns:
            tuple: A tuple containing:
                - warmup_steps (int): Number of steps allocated for warmup.
                - total_steps (int): Total number of training steps after adjusting for gradient accumulation.
        """
        batches_per_epoch = math.ceil(len(train_dataset) / (batch_size * world_size))
        accumulation = gradient_accumulation_steps or 1
        total_steps = number_epochs * math.ceil(batches_per_epoch / accumulation)
            
        if (warmup_proportion == 0): 
            return 0, total_steps
        
        warmup_steps = int(total_steps * warmup_proportion)
        return warmup_steps, total_steps

    if lr_scheduler == 'fixed':
        return get_constant_schedule(optimizer)

    warmup_steps, total_steps = calculate_warmup_steps(
        number_epochs,
        world_size,
        batch_size,
        warmup_proportion,
        train_dataset,
        gradient_accumulation_steps,
    )
        
    if lr_scheduler == 'cosine':
        # Pure cosine decay without any warmup phase
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0.0,
            last_epoch=-1
        )
    
    if lr_scheduler == 'warmup_constant':
        return get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps
        )

    if lr_scheduler == 'warmup_linear':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    if lr_scheduler == 'warmup_cosine':
        # Single-cycle cosine decay from initial LR to 0
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    if lr_scheduler == 'warmup_cosine_restart':
        # Multi-cycle cosine with hard restarts (default 1 restart cycle)
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=1
        )
        
    else:
        raise ValueError("Scheduler type not recognized.")


def select_optimizer(optimizer:str, model, lr:float, weight_decay:float, beta1:float, beta2:float) -> torch.optim.Optimizer:
    """
    Creates and returns an optimizer instance based on the specified configuration.
    
    This function selects the appropriate optimizer using a predefined dictionary mapping and configures it 
    with the model's parameters and hyperparameters like learning rate, weight decay, and beta values. 
    The 'foreach' flag is enabled to potentially streamline parameter updates on supported hardware.
    
    Parameters:
        optimizer (str): Identifier of the optimizer to be used. Must be a key in the OPTIMIZERS dictionary.
        model: The model whose parameters are to be optimized.
        lr (float): Learning rate for optimizer updates.
        weight_decay (float): Weight decay (L2 regularization coefficient).
        beta1 (float): First beta coefficient for optimizers like Adam.
        beta2 (float): Second beta coefficient for optimizers like Adam.
        
    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    
    optimizer = OPTIMIZERS[optimizer](model.parameters(), 
                                                lr=lr, 
                                                weight_decay=weight_decay,
                                                betas=(beta1, beta2),
                                                foreach=True)
    
    return optimizer


def deterministic(seed, strict=False) -> None:
    """
    Configures the environment to produce deterministic results.
    
    This function sets various backend options and seeds for PyTorch's CUDA, 
    NumPy, and Python's random module to ensure reproducible results across runs.
    
    Parameters:
        seed (int): The seed value for random number generators.
        
    Returns:
        None
    """
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Avoid the nondeterministic memory-efficient SDPA backward on older stacks.
        # Native Flash SDPA must honor strict mode or fail; never use warn_only.
        # torch.backends.cuda.enable_mem_efficient_sdp(False)


def _parse_slurm_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    trimmed = trimmed.split("(")[0].split(",")[0].strip()
    try:
        return int(trimmed)
    except ValueError:
        return None


def resolve_distributed_settings(config: Box, detected_devices: Union[int, str]) -> dict:
    num_nodes = getattr(config, "num_nodes", None)
    devices_per_node = getattr(config, "devices_per_node", None)

    if num_nodes is not None:
        num_nodes = int(num_nodes)
    if devices_per_node is not None:
        devices_per_node = int(devices_per_node)

    slurm_nnodes = _parse_slurm_int(os.getenv("SLURM_NNODES"))
    slurm_tasks_per_node = _parse_slurm_int(os.getenv("SLURM_NTASKS_PER_NODE"))
    slurm_ntasks = _parse_slurm_int(os.getenv("SLURM_NTASKS"))

    is_external_launcher = bool(
        os.getenv("SLURM_PROCID") is not None
        or os.getenv("LOCAL_RANK") is not None
        or slurm_ntasks is not None
    )

    if slurm_ntasks and slurm_ntasks > 1 and slurm_tasks_per_node is None:
        raise ValueError(
            "SLURM_NTASKS_PER_NODE is required for multi-process SLURM runs; use --ntasks-per-node."
        )

    if num_nodes is not None and num_nodes > 1 and devices_per_node is None:
        raise ValueError("devices_per_node must be set when num_nodes > 1.")

    if num_nodes is not None and slurm_nnodes is not None and num_nodes != slurm_nnodes:
        raise ValueError(
            f"num_nodes={num_nodes} does not match SLURM_NNODES={slurm_nnodes}."
        )

    if (
        devices_per_node is not None
        and slurm_tasks_per_node is not None
        and devices_per_node != slurm_tasks_per_node
    ):
        raise ValueError(
            "devices_per_node="
            f"{devices_per_node} does not match SLURM_NTASKS_PER_NODE={slurm_tasks_per_node}."
        )

    resolved_num_nodes = num_nodes if num_nodes is not None else (slurm_nnodes or 1)
    resolved_devices_per_node = (
        devices_per_node
        if devices_per_node is not None
        else slurm_tasks_per_node
    )

    available_devices = detected_devices if isinstance(detected_devices, int) else 0

    strategy = getattr(config, "parallelization_strategy", None) or "fsdp"
    if resolved_num_nodes > 1 and strategy in ("none", "dp"):
        raise ValueError(
            f"parallelization_strategy={strategy!r} is not supported for multi-node training."
        )

    if resolved_num_nodes > 1 and available_devices <= 0:
        raise ValueError("Multi-node training requires GPUs, but no CUDA devices are available.")

    if (
        slurm_tasks_per_node is not None
        and available_devices > 0
        and slurm_tasks_per_node > available_devices
    ):
        raise ValueError(
            "SLURM_NTASKS_PER_NODE="
            f"{slurm_tasks_per_node} exceeds available CUDA devices ({available_devices})."
        )

    if (
        resolved_devices_per_node is not None
        and slurm_tasks_per_node is None
        and available_devices > 0
        and resolved_devices_per_node > available_devices
    ):
        raise ValueError(
            "devices_per_node="
            f"{resolved_devices_per_node} exceeds available CUDA devices ({available_devices})."
        )

    devices = resolved_devices_per_node if resolved_devices_per_node is not None else detected_devices

    return {
        "devices": devices,
        "num_nodes": resolved_num_nodes,
        "devices_per_node": resolved_devices_per_node,
        "is_external_launcher": is_external_launcher,
        "slurm_nnodes": slurm_nnodes,
        "slurm_tasks_per_node": slurm_tasks_per_node,
        "slurm_ntasks": slurm_ntasks,
    }
    
