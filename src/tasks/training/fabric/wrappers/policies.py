"""
Module: wrap_policies.py
This module provides functionality to determine and create an auto wrap policy
for transformer-based model architectures used in distributed training setups.
It inspects the model name or the model instance for specific transformer layer classes
and returns a partial function for auto-wrapping policies based on the model type.
"""

import functools
from typing import Set, Type, Optional, Union, Callable
import torch.nn as nn
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# Dictionary mapping model name prefixes to their respective transformer block class identifiers.
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


def get_transformer_layer_class(model_name: str) -> Optional[str]:
    """
    Retrieve the transformer layer class name based on a prefix found in the model name.
    
    This function iterates through a predefined mapping of model name prefixes to their
    corresponding transformer block class names. It returns the class name if the prefix is
    found within the lowercased model name.

    Args:
        model_name (str): The name of the model (e.g., "gpt2", "llama").

    Returns:
        Optional[str]: The transformer layer class name if a matching prefix is found;
                       otherwise, None.
    """
 
    for prefix, layer_class in MODEL_TO_LAYER_CLASS_MAPPING.items():
        if prefix in model_name.lower():
            return layer_class
    return None


def get_transformer_layer_class_from_model(
    model: nn.Module,
) -> Optional[Set[Type[nn.Module]]]:
    """
    Identify transformer layer classes within the provided model by inspecting its sub-modules.
    
    This function inspects all sub-modules of the given model and collects those whose class names
    contain common transformer-related substrings such as "Block", "Layer", "DecoderLayer", or "EncoderLayer".
    
    Args:
        model (nn.Module): The PyTorch model instance to inspect.

    Returns:
        Optional[Set[Type[nn.Module]]]: A set of transformer layer classes found in the model or
                                        None if no such classes are detected.
    """
    
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


def create_auto_wrap_policy(
    model_name: str, model: Optional[nn.Module] = None, min_num_params: int = 1e8
) -> Union[Callable, None]:
    """
    Determine and create an appropriate auto wrap policy based on the model architecture.
    
    This function generates a partial function that serves as an auto wrap policy for the provided
    model type by either dynamically importing specific transformer layers or by inspecting the model
    instance. If neither approach identifies transformer layers, it falls back to a size-based policy.
    
    Args:
        model_name (str): The name of the model (e.g., "gpt2", "llama").
        model (Optional[nn.Module]): An optional PyTorch model instance to inspect for transformer layers.
        min_num_params (int): The minimum number of parameters for triggering the size-based policy.

    Returns:
        Union[Callable, None]: A partial function representing the auto wrap policy configured with the detected
                               transformer layer(s), or None if default behavior is to be used.
    """
    
    layer_class_name = get_transformer_layer_class(model_name)

    if layer_class_name:
        try:
            if "gpt2" in model_name.lower():
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block

                return functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(GPT2Block,)
                )
            elif "llama" in model_name.lower():
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                return functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=(LlamaDecoderLayer,),
                )
            # Add more model types as needed
            # ...
        except ImportError:
            pass  # Fall through to next approach if import fails

    # If we have a model instance, try to find transformer layers by inspection
    if model is not None:
        layer_classes = get_transformer_layer_class_from_model(model)
        if layer_classes:
            return functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
            )

    # Fall back to size-based policy if we couldn't determine the transformer layers
    return functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
