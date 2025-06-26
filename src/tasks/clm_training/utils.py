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
        steps_per_epoch = len(train_dataset) // (batch_size * world_size)
        total_steps = number_epochs * steps_per_epoch
        if gradient_accumulation_steps:
            total_steps = total_steps // gradient_accumulation_steps
            
        if (warmup_proportion == 0): 
            return 0, total_steps
        
        warmup_steps = int(total_steps * warmup_proportion)
        return warmup_steps, total_steps

    if lr_scheduler == 'fixed':
        scheduler = get_constant_schedule(optimizer)
        
    warmup_steps, total_steps = calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset, gradient_accumulation_steps)
        
    if lr_scheduler == 'cosine':
        # Pure cosine decay without any warmup phase
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0.0,
            last_epoch=-1
        )
    
    if lr_scheduler == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps
        )

    if lr_scheduler == 'warmup_linear':
        scheduler = get_linear_schedule_with_warmup(
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


def deterministic(seed) -> None:
    """
    Configures the environment to produce deterministic results.
    
    This function sets various backend options and seeds for PyTorch's CUDA, 
    NumPy, and Python's random module to ensure reproducible results across runs.
    
    Parameters:
        seed (int): The seed value for random number generators.
        
    Returns:
        None
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
