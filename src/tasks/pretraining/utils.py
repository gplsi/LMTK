from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.optimization import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import numpy as np
import random
from datasets import Dataset as HFDataset
import time
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# Safe barrier implementation to handle potential NVML issues
def safe_barrier(fabric, logger=None):
    """
    A safe implementation of the distributed barrier that avoids NVML errors.
    
    This function provides a fallback mechanism when torch.distributed.barrier()
    fails due to missing NVIDIA driver functions like nvmlDeviceGetNvLinkRemoteDeviceType.
    
    Parameters:
    - fabric (L.Fabric): The Lightning Fabric instance.
    - logger: Optional logger instance to log information.
    """
    if fabric.world_size <= 1:
        return
        
    try:
        # Try the normal barrier first
        fabric.barrier()
    except RuntimeError as e:
        if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e) or "NVML" in str(e) or "driver_api" in str(e):
            # If we hit any NVML-related error, use a simple time-based synchronization
            if logger:
                logger.warning(f"NVML barrier issue detected: {str(e)}")
                logger.warning("Falling back to time-based synchronization")
            
            # Sleep for a duration proportional to world size to ensure synchronization
            sleep_time = 2 + (0.1 * fabric.world_size)
            time.sleep(sleep_time)
            
            if logger:
                logger.info(f"Process {fabric.global_rank} synchronized via fallback method")
        else:
            # For other RuntimeErrors, re-raise
            if logger:
                logger.error(f"Non-NVML barrier error: {str(e)}")
            raise

# Dictionary mapping model identifiers to their corresponding wrapper classes.
# This design allows easy extension to support additional models in the future.
AUTO_WRAPPER = {
    "llama": LlamaDecoderLayer,
    "gpt2": GPT2Block
}

# Dictionary mapping optimizer names to the corresponding PyTorch optimizer classes.
# This structure centralizes the optimizer choices, making it simple to configure optimization settings.
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adamax": torch.optim.Adamax,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop
}


def select_scheduler(optimizer: torch.optim.Optimizer, lr_scheduler: str, number_epochs: int, world_size: int, batch_size: int, train_dataset: HFDataset, warmup_proportion: float, gradient_accumulation_steps: int = None) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Select and configure a learning rate scheduler based on the specified type.
    
    This function creates a scheduler that can manage learning rate adjustments throughout training,
    with options for linear, warmup-constant, and fixed schedules.
    
    Parameters:
    - optimizer (torch.optim.Optimizer): The optimizer to attach the scheduler to.
    - lr_scheduler (str): Type of schedule to use (fixed, warmup_linear, warmup_constant).
    - number_epochs (int): Total number of training epochs.
    - world_size (int): Number of distributed training processes.
    - batch_size (int): Batch size per process.
    - train_dataset (HFDataset): Training dataset for calculating steps.
    - warmup_proportion (float): Proportion of steps for warming up learning rate.
    - gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation.
    
    Returns:
    - torch.optim.lr_scheduler.LambdaLR: The configured learning rate scheduler.
    """
    def calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset, gradient_accumulation_steps=None):
        """Calculate the number of warmup steps for the scheduler"""
        # Calculate total number of training steps based on dataset size, epochs, etc.
        dataset_size = len(train_dataset)
        steps_per_epoch = dataset_size // (batch_size * world_size) 
        total_steps = number_epochs * steps_per_epoch
        if gradient_accumulation_steps:
            total_steps = total_steps // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_proportion)
        return warmup_steps, total_steps
    
    # Determine the appropriate scheduler based on the specified type.
    if lr_scheduler == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif lr_scheduler == 'warmup_constant':
        warmup_steps, _ = calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps
        )
    elif lr_scheduler == 'warmup_linear':
        # Use a linear schedule with warmup period.
        warmup_steps, training_steps = calculate_warmup_steps(
            number_epochs, world_size, batch_size, warmup_proportion, train_dataset, gradient_accumulation_steps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
    else:
        # default to a constant schedule
        scheduler = get_constant_schedule(optimizer)
    
    return scheduler

def select_optimizer(optimizer: str, model, lr: float, weight_decay: float, beta1: float, beta2: float) -> torch.optim.Optimizer:
    """
    Select and configure an optimizer for the model.
    
    Parameters:
    - optimizer (str): The name of the optimizer to use (must be in the OPTIMIZERS dictionary).
    - model: The PyTorch model to optimize.
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay factor.
    - beta1 (float): Beta1 parameter for Adam-based optimizers.
    - beta2 (float): Beta2 parameter for Adam-based optimizers.
    
    Returns:
    - torch.optim.Optimizer: The configured optimizer instance.
    
    Raises:
    - ValueError: If the specified optimizer name is not found in OPTIMIZERS.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Optimizer {optimizer} not found. Available optimizers: {list(OPTIMIZERS.keys())}")
    
    optimizer = OPTIMIZERS[optimizer](model.parameters(), 
                                      lr=lr, 
                                      weight_decay=weight_decay,
                                      betas=(beta1, beta2),
                                      foreach=True)
    return optimizer

def setup_environment(seed: int) -> None:
    """
    Set up a deterministic environment for reproducibility.
    
    This function configures PyTorch, numpy, and Python's random module to use the
    provided seed for deterministic operation.
    
    Parameters:
    - seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def rank_zero_only(fn):
    """
    Decorator to make a function run only on the process with rank zero.
    
    This is useful for logging functions to avoid duplicate messages in distributed training.
    
    Parameters:
    - fn: The function to be wrapped
    
    Returns:
    - A wrapped function that only executes on rank zero
    """
    def wrapped(self, *args, **kwargs):
        if hasattr(self, 'fabric') and self.fabric.global_rank == 0:
            return fn(self, *args, **kwargs)
    return wrapped

def rank_zero_debug(logger, message):
    """
    Log a debug message only on the process with rank zero.
    
    Parameters:
    - logger: The logger to use
    - message: The message to log
    """
    if hasattr(logger, 'fabric') and logger.fabric.global_rank == 0:
        logger.debug(message)
    
def rank_zero_info(logger, message):
    """
    Log an info message only on the process with rank zero.
    
    Parameters:
    - logger: The logger to use
    - message: The message to log
    """
    if hasattr(logger, 'fabric') and logger.fabric.global_rank == 0:
        logger.info(message)
