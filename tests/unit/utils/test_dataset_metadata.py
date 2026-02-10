from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.dataset.metadata import (
    build_tokenization_metadata,
    extract_eos_token_id,
    read_tokenization_metadata,
    write_tokenization_metadata,
)


def test_tokenization_metadata_roundtrip(tmp_path: Path) -> None:
    payload = build_tokenization_metadata(
        tokenizer_name="/models/salamandra-2b",
        eos_token_id=2,
        task="clm_training",
        created_by="unit-test",
    )
    metadata_path = write_tokenization_metadata(tmp_path, payload)

    loaded = read_tokenization_metadata(tmp_path)
    assert loaded is not None
    assert metadata_path.exists()
    assert loaded["tokenizer_name"] == "/models/salamandra-2b"
    assert int(loaded["eos_token_id"]) == 2
    assert extract_eos_token_id(loaded) == 2


def test_read_tokenization_metadata_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_tokenization_metadata(tmp_path) is None


def test_extract_eos_token_id_validates_input() -> None:
    with pytest.raises(ValueError, match="non-integer eos_token_id"):
        extract_eos_token_id({"eos_token_id": "abc"}, source_label="test metadata")

    with pytest.raises(ValueError, match="invalid eos_token_id"):
        extract_eos_token_id({"eos_token_id": -1}, source_label="test metadata")
