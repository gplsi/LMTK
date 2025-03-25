# src/tasks/pretraining/fabric/wrappers/fsdp_config.py (new file)

import functools
from typing import Optional, Dict, Any, Set, Type, Union, Callable
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from tasks.pretraining.fabric.wrappers.policies import create_auto_wrap_policy

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

def resolve_fsdp_config(config: Dict[str, Any], model_name: str, logger=None) -> Dict[str, Any]:
    """
    Resolve the FSDP configuration by merging user-provided settings with sensible defaults.

    This function takes a user-provided FSDP configuration and fills in missing values with defaults.
    It also determines the appropriate auto-wrap and activation checkpointing policies based on the 
    model name and other parameters.

    Args:
        config (Dict[str, Any]): User-provided FSDP configuration.
        model_name (str): Name of the model being trained.
        logger: Optional logger instance for debug outputs.

    Returns:
        Dict[str, Any]: The complete FSDP configuration with defaults filled in.
    """
    # Start with the default configuration
    fsdp_config = get_default_fsdp_config()

    # Helper function to log only on rank 0
    def rank_zero_log(level, msg):
        if logger:
            # Check if we're running in a distributed environment and this is rank 0
            if hasattr(logger, 'fabric'):
                if logger.fabric.global_rank == 0:
                    if level == 'debug':
                        logger.debug(msg)
                    elif level == 'info':
                        logger.info(msg)
                    elif level == 'warning':
                        logger.warning(msg)
            else:
                # If no fabric attribute (single process), just log normally
                if level == 'debug':
                    logger.debug(msg)
                elif level == 'info':
                    logger.info(msg)
                elif level == 'warning':
                    logger.warning(msg)
    
    rank_zero_log('debug', f"Default FSDP config: {fsdp_config}")
    
    # Check if this is a Box object and handle it properly
    if hasattr(config, '_box_config'):
        rank_zero_log('debug', "Config is a Box object, accessing Box attributes")
        
        # In Box objects, the attributes are directly accessible as attributes
        # Try accessing common FSDP configuration keys directly
        policy_name = None
        
        # First try to access auto_wrap_policy directly on the config object
        if hasattr(config, 'auto_wrap_policy'):
            policy_name = config.auto_wrap_policy
            rank_zero_log('debug', f"Found auto_wrap_policy directly in Box config: {policy_name}")
            
            # Create appropriate wrap policy based on the string value
            if isinstance(policy_name, str):
                rank_zero_log('debug', f"Creating auto_wrap_policy for: {policy_name}")
                auto_wrap_policy = create_auto_wrap_policy(
                    model_name=model_name if policy_name == "auto" else policy_name,
                    min_num_params=fsdp_config["min_num_params"]
                )
                fsdp_config["auto_wrap_policy"] = auto_wrap_policy
        
        # Try to access other FSDP settings directly
        for key in ['sharding_strategy', 'state_dict_type', 'limit_all_gathers', 'cpu_offload']:
            if hasattr(config, key):
                value = getattr(config, key)
                rank_zero_log('debug', f"Setting {key} = {value} directly from Box config")
                fsdp_config[key] = value
    else:
        # Handle regular dictionary config
        rank_zero_log('debug', f"User config keys: {list(config.keys())}")
        
        # Check if auto_wrap_policy exists in the config
        if 'auto_wrap_policy' in config:
            rank_zero_log('debug', f"Found auto_wrap_policy in config: {config['auto_wrap_policy']}")
            policy_name = config['auto_wrap_policy']
            if isinstance(policy_name, str):
                auto_wrap_policy = create_auto_wrap_policy(
                    model_name=model_name if policy_name == "auto" else policy_name,
                    min_num_params=fsdp_config["min_num_params"]
                )
                fsdp_config["auto_wrap_policy"] = auto_wrap_policy
        else:
            # Override default values with user-provided settings if available
            for key, value in config.items():
                if key in fsdp_config and value is not None:
                    rank_zero_log('debug', f"Setting {key} = {value} from user config")
                    fsdp_config[key] = value
    
    # If no policy was specified or found, create one based on model_name
    if not fsdp_config["auto_wrap_policy"]:
        rank_zero_log('debug', f"No policy specified, creating one based on model_name: {model_name}")
        auto_wrap_policy = create_auto_wrap_policy(
            model_name=model_name,
            min_num_params=fsdp_config["min_num_params"]
        )
        fsdp_config["auto_wrap_policy"] = auto_wrap_policy
    
    # Configure activation checkpointing policy if not explicitly provided
    if fsdp_config["auto_wrap_policy"] and hasattr(fsdp_config["auto_wrap_policy"], "keywords"):
        if "transformer_layer_cls" in fsdp_config["auto_wrap_policy"].keywords:
            # Create a policy dictionary for transformer layers if applicable
            layer_cls = fsdp_config["auto_wrap_policy"].keywords["transformer_layer_cls"]
            fsdp_config["activation_checkpointing_policy"] = {layer_cls: dict()}
            rank_zero_log('debug', f"Set activation checkpointing for {layer_cls}")
    
    rank_zero_log('debug', f"Final FSDP config auto_wrap_policy: {fsdp_config['auto_wrap_policy']}")
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
    
    # Match the model name prefix to its corresponding layer class
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
    
    # Iterate through all submodules of the given model
    for module in model.modules():
        module_class_name = module.__class__.__name__
        
        # Check if the module's class name matches common transformer layer patterns
        if any(
            class_name in module_class_name
            for class_name in ["Block", "Layer", "DecoderLayer", "EncoderLayer"]
        ):
            layer_classes.add(module.__class__)
    
    return layer_classes if layer_classes else None
