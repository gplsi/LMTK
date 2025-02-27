# src/tasks/pretraining/fabric/wrap_policies.py

import functools
from typing import Set, Type, Optional, Union, Callable

import torch.nn as nn
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# Dictionary mapping model name prefixes to their transformer block classes
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
    """Get the transformer layer class name based on model name prefix."""
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


def create_auto_wrap_policy(
    model_name: str, model: Optional[nn.Module] = None, min_num_params: int = 1e8
) -> Union[Callable, None]:
    """
    Create an appropriate auto wrap policy based on the model architecture.

    Args:
        model_name: Name of the model (e.g., "gpt2", "llama", etc.)
        model: Optional model instance to inspect for layer classes
        min_num_params: Minimum number of parameters for size-based policy

    Returns:
        An auto wrap policy function or None for default behavior
    """
    # Try to get transformer layer class from model name
    layer_class_name = get_transformer_layer_class(model_name)

    # If we have a specific layer class for this model type
    if layer_class_name:
        try:
            # Import the module dynamically based on model type
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
