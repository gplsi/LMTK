from typing import Dict, Type, Literal
from .base import PromptStyle
from src.utils.logging import get_logger

logger = get_logger(__name__)

# System blocks
from .system_blocks import (
    DefaultSystem, AlpacaSystem, StableLMSystem, Llama2System, 
    Llama3System, FreeWilly2System, VicunaSystem, CodeLlamaSystem, TinyLlamaSystem
)

# User blocks
from .user_blocks import (
    DefaultUser, AlpacaUser, FlanUser, StableLMUser, 
    StableLMZephyrUser, Llama2User, Llama3User
)

# Assistant blocks
from .assistant_blocks import (
    DefaultAssistant, StableLMAssistant, StableLMZephyrAssistant,
    TogetherComputerChatAssistant, TogetherComputerInstructAssistant,
    FalconAssistant, VicunaAssistant, Llama2Assistant, Llama3Assistant,
    FreeWilly2Assistant, PlatypusAssistant, CodeLlamaAssistant,
    Phi1Assistant, Phi2Assistant, TinyLlamaAssistant, GemmaAssistant
)

# Final blocks
from .final_blocks import (
    DefaultFinal, StableLMFinal, StableLMZephyrFinal,
    TogetherComputerChatFinal, TogetherComputerInstructFinal,
    FalconFinal, Llama3Final, Phi1Final
)

# Export these types
__all__ = [
    'get_prompt_component',
    'SYSTEM_BLOCKS',
    'USER_BLOCKS',
    'ASSISTANT_BLOCKS',
    'FINAL_BLOCKS',
    'ComponentType'
]

# Map model names to their prompt component classes
SYSTEM_BLOCKS: Dict[str, Type[PromptStyle]] = {
    "default": DefaultSystem,
    "alpaca": AlpacaSystem,
    "stablelm": StableLMSystem,
    "llama2": Llama2System,
    "llama3": Llama3System,
    "freewilly2": FreeWilly2System,
    "vicuna": VicunaSystem,
    "codellama": CodeLlamaSystem,
    "tinyllama": TinyLlamaSystem,
}

USER_BLOCKS: Dict[str, Type[PromptStyle]] = {
    "default": DefaultUser,
    "alpaca": AlpacaUser,
    "flan": FlanUser,
    "stablelm": StableLMUser,
    "stablelm-zephyr": StableLMZephyrUser,
    "llama2": Llama2User,
    "llama3": Llama3User,
}

ASSISTANT_BLOCKS: Dict[str, Type[PromptStyle]] = {
    "default": DefaultAssistant,
    "stablelm": StableLMAssistant,
    "stablelm-zephyr": StableLMZephyrAssistant,
    "together-computer-chat": TogetherComputerChatAssistant,
    "together-computer-instruct": TogetherComputerInstructAssistant,
    "falcon": FalconAssistant,
    "vicuna": VicunaAssistant,
    "llama2": Llama2Assistant,
    "llama3": Llama3Assistant,
    "freewilly2": FreeWilly2Assistant,
    "platypus": PlatypusAssistant,
    "codellama": CodeLlamaAssistant,
    "phi1": Phi1Assistant,
    "phi2": Phi2Assistant,
    "tinyllama": TinyLlamaAssistant,
    "gemma": GemmaAssistant,
}

FINAL_BLOCKS: Dict[str, Type[PromptStyle]] = {
    "default": DefaultFinal,
    "stablelm": StableLMFinal,
    "stablelm-zephyr": StableLMZephyrFinal,
    "together-computer-chat": TogetherComputerChatFinal,
    "together-computer-instruct": TogetherComputerInstructFinal,
    "falcon": FalconFinal,
    "llama3": Llama3Final,
    "phi1": Phi1Final,
}

ComponentType = Literal['system', 'user', 'assistant', 'final']

def get_prompt_component(component_type: ComponentType, model_name: str) -> Type[PromptStyle]:
    """
    Get the appropriate prompt component for the given model.
    
    Args:
        component_type: Type of prompt component ('system', 'user', 'assistant', or 'final')
        model_name: The name of the model to get the component for
        
    Returns:
        The appropriate PromptStyle class for the requested component and model
        
    Examples:
        >>> get_prompt_component('system', 'llama2')  # Returns Llama2System
        >>> get_prompt_component('user', 'alpaca')    # Returns AlpacaUser
        
    Raises:
        ValueError: If component_type is invalid
    """
    component_map = {
        'system': SYSTEM_BLOCKS,
        'user': USER_BLOCKS,
        'assistant': ASSISTANT_BLOCKS,
        'final': FINAL_BLOCKS,
    }
    
    if component_type not in component_map:
        raise ValueError(
            f"Unknown component type: {component_type}. "
            f"Must be one of: {list(component_map.keys())}"
        )
        
    components = component_map[component_type]
    model_name = model_name.lower()
    
    if model_name not in components:
        logger.warning(f"No {component_type} component found for model '{model_name}'.")
        
    return components.get(model_name, components["default"])