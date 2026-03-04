from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from box import Box

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


def test_tokenization_schema_accepts_dataset_filters(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenization_parquet_filters.yaml"
    config = {
        "task": "tokenization",
        "experiment_name": "test_tokenization_parquet_filters",
        "tokenizer": {"tokenizer_name": "dummy", "task": "clm_training"},
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
            "format": "files",
            "file_config": {"format": "parquet", "text_column": "content"},
            "filters": [{"column": "label", "value": "not-fake"}],
        },
        "output": {"path": "/tmp/out"},
        "test_size": 0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "tokenization")

    assert validated.dataset.filters[0].column == "label"
    assert validated.dataset.filters[0].value == "not-fake"


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


def test_dataset_storage_apply_filters_keeps_only_matching_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "claim": ["a", "b", "c"],
            "label": ["not-fake", "fake", "not-fake"],
        }
    )
    parquet_path = tmp_path / "sample.parquet"
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(parquet_path),
        file_config={"format": "parquet", "text_column": "claim"},
    )
    filtered = storage.apply_filters(
        dataset,
        [{"column": "label", "value": "not-fake"}],
    )

    assert "train" in filtered
    assert len(filtered["train"]) == 2
    assert set(filtered["train"]["label"]) == {"not-fake"}


def test_dataset_storage_apply_filters_raises_on_missing_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"claim": ["a", "b"], "label": ["not-fake", "fake"]})
    parquet_path = tmp_path / "sample.parquet"
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(parquet_path),
        file_config={"format": "parquet", "text_column": "claim"},
    )

    with pytest.raises(ValueError, match="Filter column 'missing' not found"):
        storage.apply_filters(dataset, [{"column": "missing", "value": "x"}])


def test_dataset_storage_apply_filters_raises_when_all_rows_removed(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    table = pa.table({"claim": ["a", "b"], "label": ["fake", "fake"]})
    parquet_path = tmp_path / "sample.parquet"
    parquet.write_table(table, parquet_path)

    storage = DatasetStorage(verbose_level=VerboseLevel.ERRORS)
    dataset = storage.process_files(
        str(parquet_path),
        file_config={"format": "parquet", "text_column": "claim"},
    )

    with pytest.raises(ValueError, match="dataset.filters removed all rows"):
        storage.apply_filters(dataset, [{"column": "label", "value": "not-fake"}])


def test_tokenization_orchestrator_applies_filters_before_split(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.tokenization.orchestrator import TokenizationOrchestrator

    datasets_module = pytest.importorskip("datasets")
    base_dataset = datasets_module.Dataset.from_dict(
        {
            "text": ["a", "b", "c", "d"],
            "label": ["not-fake", "fake", "not-fake", "fake"],
        }
    )

    config = Box(
        {
            "task": "tokenization",
            "experiment_name": "test_tokenization_filter_order",
            "verbose_level": 1,
            "tokenizer": {"tokenizer_name": "dummy", "task": "clm_training"},
            "dataset": {
                "source": "local",
                "nameOrPath": "/tmp/irrelevant-for-mocked-process-files",
                "format": "files",
                "file_config": {"format": "parquet", "text_column": "claim"},
                "filters": [{"column": "label", "value": "not-fake"}],
            },
            "output": {"path": "/tmp/out"},
            "test_size": 0.5,
        },
        box_dots=True,
    )
    orchestrator = TokenizationOrchestrator(config)

    call_order: list[str] = []
    original_apply_filters = DatasetStorage.apply_filters
    original_split = DatasetStorage.split

    def _mock_process_files(self, files_path: str, file_config=None):
        _ = (files_path, file_config)
        return base_dataset

    def _tracked_apply_filters(self, dataset, filters):
        call_order.append("apply_filters")
        return original_apply_filters(self, dataset, filters)

    def _tracked_split(self, dataset, split_ratio):
        call_order.append("split")
        return original_split(self, dataset, split_ratio)

    monkeypatch.setattr(DatasetStorage, "process_files", _mock_process_files)
    monkeypatch.setattr(DatasetStorage, "apply_filters", _tracked_apply_filters)
    monkeypatch.setattr(DatasetStorage, "split", _tracked_split)

    loaded = orchestrator.load_dataset()

    assert call_order == ["apply_filters", "split"]
    assert set(loaded.keys()) == {"train", "valid"}
    assert set(loaded["train"]["label"]) == {"not-fake"}
    assert set(loaded["valid"]["label"]) == {"not-fake"}
