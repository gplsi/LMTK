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
    Selects and returns an appropriate learning rate scheduler based on the specified configuration.
    
    This function supports three types of schedulers:
      - 'fixed': a constant learning rate scheduler.
      - 'warmup_constant': a scheduler with initial warmup followed by a constant learning rate.
      - 'warmup_linear': a scheduler with an initial warmup and subsequent linear decay.
      
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
        steps_per_epoch = len(train_dataset) // (batch_size * world_size)
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
        warmup_steps, total_steps = calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset)  
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError("Scheduler type not recognized.")

    return scheduler


def select_optimizer(optimizer: str, model, lr: float, weight_decay: float, beta1: float, beta2: float) -> torch.optim.Optimizer:
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


def setup_environment(seed: int) -> None:
    """
    Configures the environment to produce deterministic results.
    
    This function sets various backend options and seeds for PyTorch's CUDA, 
    NumPy, and Python's random module to ensure reproducible results across runs.
    
    Parameters:
        seed (int): The seed value for random number generators.
        
    Returns:
        None
    """
    # Ensure deterministic behavior in CUDA backend.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Seed the standard Python random module.
    random.seed(seed)
    # Seed NumPy's random number generator.
    np.random.seed(seed)
    # Seed PyTorch's CPU and GPU random number generators.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
