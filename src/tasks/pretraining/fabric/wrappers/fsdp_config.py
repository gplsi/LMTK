# src/tasks/pretraining/fabric/wrappers/fsdp_config.py (new file)

import functools
from typing import Optional, Dict, Any, Set, Type, Union, Callable
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from tasks.pretraining.fabric.wrappers.policies import create_auto_wrap_policy

def get_default_fsdp_config() -> Dict[str, Any]:
    """Return default configuration for FSDP strategy."""
    return {
        "sharding_strategy": "FULL_SHARD",
        "auto_wrap_policy": None,  # Will be determined based on model
        "activation_checkpointing": None,  # Will be determined based on model and auto_wrap_policy
        "state_dict_type": "full",
        "limit_all_gathers": True,
        "cpu_offload": False,
        "min_num_params": 1e7  # Default threshold for size-based policy
    }

def resolve_fsdp_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Resolve FSDP configuration with sensible defaults based on the model.
    
    Args:
        config: User configuration
        model_name: Name of the model being trained
        
    Returns:
        Complete FSDP configuration with defaults filled in
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

# Keep the existing functions from policies.py
def get_transformer_layer_class(model_name: str) -> Optional[str]:
    """Get the transformer layer class name based on model name prefix."""
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
    """Attempt to find transformer layer classes in the model."""
    layer_classes = set()
    # Try to find common transformer layer classes in the model
    for module in model.modules():
        module_class_name = module.__class__.__name__
        if any(
            class_name in module_class_name
            for class_name in ["Block", "Layer", "DecoderLayer", "EncoderLayer"]
        ):
            layer_classes.add(module.__class__)
    return layer_classes if layer_classes else None
