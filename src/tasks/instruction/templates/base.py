from abc import ABC, abstractmethod
from typing import Tuple, List
from src.utils.logging import get_logger


class PromptStyle(ABC):
    """Base interface for prompt styles."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__, config.verbose_level)

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer) -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id],)

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()

    @classmethod
    def from_config(cls, config: Config) -> "PromptStyle":
        return model_name_to_prompt_style(config.name)