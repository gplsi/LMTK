# src/tasks/pretraining/fabric/wrappers/fsdp_config.py (new file)

import functools
from typing import Optional, Dict, Any, Set, Type, Union, Callable
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from tasks.clm_training.fabric.wrappers.policies import create_auto_wrap_policy

def get_default_fsdp_config() -> Dict[str, Any]:
    """
    Return the default configuration for Fully Sharded Data Parallel (FSDP) strategy.

    This function provides a baseline configuration for FSDP that can be used as a starting point
    for distributed training. The configuration includes default values for sharding strategy,
    auto-wrapping policies, activation checkpointing, state dictionary type, and CPU offloading.

    Returns:
        Dict[str, Any]: A dictionary containing the default FSDP configuration.
    """
    
    return {
        "sharding_strategy": "FULL_SHARD",  # Default sharding strategy
        "auto_wrap_policy": None,  # Auto-wrap policy to be determined based on the model
        "activation_checkpointing": None,  # Activation checkpointing policy to be determined
        "state_dict_type": "full",  # State dict type for saving/loading model states
        "limit_all_gathers": True,  # Limit all-gather operations to reduce communication overhead
        "cpu_offload": False,  # Disable CPU offloading by default
        "min_num_params": 1e7  # Minimum number of parameters for size-based auto-wrap policy
    }

def resolve_fsdp_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Resolve the FSDP configuration by merging user-provided settings with sensible defaults.

    This function takes a user-provided FSDP configuration and fills in missing values with defaults.
    It also determines the appropriate auto-wrap and activation checkpointing policies based on the 
    model name and other parameters.

    Args:
        config (Dict[str, Any]): User-provided FSDP configuration.
        model_name (str): Name of the model being trained.

    Returns:
        Dict[str, Any]: The complete FSDP configuration with defaults filled in.
    """
    
    # Start with default config
    fsdp_config = get_default_fsdp_config()
    # Override with user config if provided
    for key, value in config.items():
        if key in fsdp_config and value is not None:
            fsdp_config[key] = value
    
    # If auto_wrap_policy not specified, determine based on model
    if not config.get("auto_wrap_policy"):
        auto_wrap_policy = create_auto_wrap_policy(
            model_name=model_name,
            min_num_params=fsdp_config["min_num_params"]
        )
        fsdp_config["auto_wrap_policy"] = auto_wrap_policy
    
    # Instead of a tuple, use the new activation_checkpointing_policy
    if not config.get("activation_checkpointing", False) and fsdp_config["auto_wrap_policy"]:
        if hasattr(fsdp_config["auto_wrap_policy"], "keywords") and "transformer_layer_cls" in fsdp_config["auto_wrap_policy"].keywords:
            # Here we create a policy dictionary for those layers.
            layer_cls = fsdp_config["auto_wrap_policy"].keywords["transformer_layer_cls"]
            fsdp_config["activation_checkpointing_policy"] = {layer_cls: dict()}
    
    return fsdp_config


def get_transformer_layer_class(model_name: str) -> Optional[str]:
    """
    Get the transformer layer class name based on the model's name prefix.

    This function maps common model names (e.g., GPT-2, T5) to their corresponding transformer 
    layer class names. It helps identify which layers should be wrapped or checkpointed during training.

    Args:
        model_name (str): The name of the model being trained.

    Returns:
        Optional[str]: The transformer layer class name if found; otherwise, None.
    """
    
    MODEL_TO_LAYER_CLASS_MAPPING = {
        "gpt2": "GPT2Block",
        "llama": "LlamaDecoderLayer",
        "t5": "T5Block",
        "bert": "BertLayer",
        "roberta": "RobertaLayer",
        "opt": "OPTDecoderLayer",
        "bloom": "BloomBlock",
        "falcon": "FalconDecoderLayer",
        "mistral": "MistralDecoderLayer",
    }
    
    for prefix, layer_class in MODEL_TO_LAYER_CLASS_MAPPING.items():
        if prefix in model_name.lower():
            return layer_class
    
    return None

def get_transformer_layer_class_from_model(
    model: nn.Module,
) -> Optional[Set[Type[nn.Module]]]:
    """
    Attempt to find transformer layer classes within a given PyTorch model.

    This function inspects all submodules of a PyTorch model to identify classes commonly used 
    in transformer architectures (e.g., Block, Layer). It returns a set of unique layer classes 
    found in the model.

    Args:
        model (nn.Module): The PyTorch model to inspect.

    Returns:
        Optional[Set[Type[nn.Module]]]: A set of unique transformer layer classes found in the 
                                         model. Returns None if no such classes are found.
    """
    
    layer_classes = set()
    
    
    for module in model.modules():
        module_class_name = module.__class__.__name__
        if any(
            class_name in module_class_name
            for class_name in ["Block", "Layer", "DecoderLayer", "EncoderLayer"]
        ):
            layer_classes.add(module.__class__)
    return layer_classes if layer_classes else None
