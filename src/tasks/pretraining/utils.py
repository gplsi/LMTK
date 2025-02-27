from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.optimization import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import numpy as np
import random

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
def select_scheduler(optimizer, lr_scheduler, number_epochs, world_size, batch_size, train_dataset, warmup_proportion, gradient_accumulation_steps=None):
    
    def calculate_warmup_steps(number_epochs, world_size, batch_size, warmup_proportion, train_dataset, gradient_accumulation_steps=None):
        steps_per_epoch = len(train_dataset) // (batch_size * world_size)
        total_steps = number_epochs * steps_per_epoch
        if gradient_accumulation_steps:
            total_steps = total_steps // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_proportion)
        return warmup_steps, total_steps

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


def select_optimizer(optimizer:str, model, lr:float, weight_decay:float, beta1:float, beta2:float):
    optimizer = OPTIMIZERS[optimizer](model.parameters(), 
                                                lr=lr, 
                                                weight_decay=weight_decay,
                                                betas=(beta1, beta2),
                                                foreach=True)
    
    return optimizer

# For deterministic results, it will be used only if seed is provided
def setup_environment(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
