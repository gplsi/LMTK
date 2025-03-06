# Patch GPT2Block to accept extra keyword arguments.
from box import Box
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as OriginalGPT2Block
from transformers import GPT2Config
#Global variable to hold a default config.
_GLOBAL_GPT2_CONFIG = None

def set_global_gpt2_config(config: Box):
    global _GLOBAL_GPT2_CONFIG
    _GLOBAL_GPT2_CONFIG = config
# Save the original __init__ so we can call it
_original_gpt2block_init = OriginalGPT2Block.__init__


def patched_GPT2Block_init(self, config: Box =None, layer_idx: int =None, **kwargs) -> None:
    # If no config is provided, try to get it from kwargs.
    if config is None:
        if "module" in kwargs and hasattr(kwargs["module"], "config"):
            config = kwargs.pop("module").config
        elif _GLOBAL_GPT2_CONFIG is not None:
            config = _GLOBAL_GPT2_CONFIG
        else:
            # Fall back to a default GPT2Config (or raise an error if that's not acceptable)
            config = GPT2Config()
    _original_gpt2block_init(self, config, layer_idx=layer_idx)
    # Ignore any remaining kwargs.

# Apply the patch
OriginalGPT2Block.__init__ = patched_GPT2Block_init