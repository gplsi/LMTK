from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOKENIZATION_METADATA_FILENAME = "lmtk_tokenization_meta.json"
TOKENIZATION_METADATA_SCHEMA_VERSION = 1


def _normalize_dataset_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root).expanduser().resolve()


def tokenization_metadata_path(dataset_root: str | Path) -> Path:
    root = _normalize_dataset_root(dataset_root)
    return root / TOKENIZATION_METADATA_FILENAME


def build_tokenization_metadata(
    *,
    tokenizer_name: str,
    eos_token_id: int,
    task: str | None = None,
    created_by: str = "lmtk-tokenization",
) -> dict[str, Any]:
    tokenizer_name_value = str(tokenizer_name).strip()
    if not tokenizer_name_value:
        raise ValueError("tokenizer_name must be a non-empty string.")

    eos_value = int(eos_token_id)
    if eos_value < 0:
        raise ValueError("eos_token_id must be >= 0.")

    payload: dict[str, Any] = {
        "schema_version": TOKENIZATION_METADATA_SCHEMA_VERSION,
        "tokenizer_name": tokenizer_name_value,
        "eos_token_id": eos_value,
        "created_by": str(created_by).strip() or "lmtk-tokenization",
    }
    if task is not None:
        payload["task"] = str(task)
    return payload


def read_tokenization_metadata(dataset_root: str | Path) -> dict[str, Any] | None:
    metadata_file = tokenization_metadata_path(dataset_root)
    if not metadata_file.exists():
        return None

    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in tokenization metadata file {metadata_file}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Tokenization metadata file {metadata_file} must contain a JSON object."
        )
    return payload


def write_tokenization_metadata(
    dataset_root: str | Path,
    payload: dict[str, Any],
) -> Path:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary.")

    metadata_file = tokenization_metadata_path(dataset_root)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_file


def extract_eos_token_id(
    payload: dict[str, Any] | None,
    *,
    source_label: str = "tokenization metadata",
) -> int | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError(f"{source_label} must be a dictionary.")

    raw = payload.get("eos_token_id", None)
    if raw is None:
        return None

    try:
        eos_value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_label} has non-integer eos_token_id value: {raw!r}."
        ) from exc

    if eos_value < 0:
        raise ValueError(f"{source_label} has invalid eos_token_id={eos_value}.")

    return eos_value
