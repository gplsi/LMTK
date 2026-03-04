from __future__ import annotations

import pytest

from src.tasks.tokenization.tokenizer.base import BaseTokenizer
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
from src.utils.logging import VerboseLevel


class _DummyTokenizer(BaseTokenizer):
    def tokenize(self, dataset):
        return dataset


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"
        self.sep_token = None
        self.eos_token_id = 2

    def add_special_tokens(self, payload):
        self.pad_token = payload.get("pad_token")


def test_initialize_tokenizer_falls_back_when_tokenizersbackend_class_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TokenizerConfig(
        context_length=16,
        overlap=0,
        tokenizer_name="/tmp/local-tokenizer",
        verbose_level=VerboseLevel.ERRORS,
    )
    tokenizer = _DummyTokenizer(config)

    fake = _FakeTokenizer()

    def _raise_tokenizersbackend_error(*args, **kwargs):
        raise RuntimeError(
            "Tokenizer class TokenizersBackend does not exist or is not currently imported."
        )

    monkeypatch.setattr(
        "src.tasks.tokenization.tokenizer.base.AutoTokenizer.from_pretrained",
        _raise_tokenizersbackend_error,
    )
    monkeypatch.setattr(
        "src.tasks.tokenization.tokenizer.base.PreTrainedTokenizerFast.from_pretrained",
        lambda *args, **kwargs: fake,
    )

    tokenizer._initialize_tokenizer()

    assert tokenizer._tokenizer is fake
    assert tokenizer._tokenizer.pad_token == "</s>"
