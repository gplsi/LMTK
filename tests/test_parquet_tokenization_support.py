from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.config_loader import ConfigValidator
from src.utils.dataset.storage import DatasetStorage
from src.utils.logging import VerboseLevel


def test_tokenization_schema_accepts_parquet_file_config_format(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenization_parquet.yaml"
    config = {
        "task": "tokenization",
        "experiment_name": "test_tokenization_parquet",
        "tokenizer": {"tokenizer_name": "dummy", "task": "clm_training"},
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
            "format": "files",
            "file_config": {"format": "parquet", "text_column": "content"},
        },
        "output": {"path": "/tmp/out"},
        "test_size": 0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "tokenization")

    assert validated.dataset.file_config.format == "parquet"


def test_dataset_storage_process_files_parquet_renames_configured_text_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"content": ["hello", "world"]})
    parquet_path = tmp_path / "sample.parquet"
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(tmp_path),
        file_config={"format": "parquet", "text_key": "content"},
    )

    assert "train" in dataset
    assert "text" in dataset["train"].column_names
    assert dataset["train"][0]["text"] == "hello"


def test_dataset_storage_process_files_parquet_supports_extensionless_files(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"content": ["hello", "world"]})
    parquet_path = tmp_path / "sample"  # intentionally no extension
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(tmp_path),
        file_config={"format": "parquet", "text_key": "content"},
    )

    assert "train" in dataset
    assert "text" in dataset["train"].column_names
    assert dataset["train"][0]["text"] == "hello"


def test_dataset_storage_process_files_parquet_accepts_single_file_path(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"content": ["hello", "world"]})
    parquet_path = tmp_path / "sample.parquet"
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(parquet_path),
        file_config={"format": "parquet", "text_key": "content"},
    )

    assert "train" in dataset
    assert "text" in dataset["train"].column_names
    assert dataset["train"][0]["text"] == "hello"


def test_dataset_storage_process_files_parquet_accepts_single_extensionless_file_path(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"content": ["hello", "world"]})
    parquet_path = tmp_path / "sample"  # intentionally no extension
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(parquet_path),
        file_config={"format": "parquet", "text_key": "content"},
    )

    assert "train" in dataset
    assert "text" in dataset["train"].column_names
    assert dataset["train"][0]["text"] == "hello"
