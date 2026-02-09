from __future__ import annotations

from unittest.mock import patch

from box import Box
import pytest

pytest.importorskip("datasets")
from datasets import Dataset, DatasetDict

from src.tasks.training.orchestrator import ContinualOrchestrator


def _base_config() -> Box:
    return Box(
        {
            "verbose_level": 1,
            "dataset": {
                "source": "local",
                "format": "hf",
            },
        },
        box_dots=True,
    )


def test_load_dataset_mixture_returns_source_dict_and_wraps_dataset() -> None:
    config = _base_config()
    config.dataset.sources = [
        {"dataset_id": "A", "nameOrPath": "/tmp/a"},
        {"dataset_id": "B", "nameOrPath": "/tmp/b"},
    ]
    config.dataset.mixture = {"enabled": True}

    ds_a = Dataset.from_dict({"input_ids": [[1, 2]], "length": [2]})
    ds_b = Dataset.from_dict({"input_ids": [[3, 4]], "length": [2]})

    with patch("src.tasks.training.orchestrator.DatasetStorage.load_from_disk", side_effect=[ds_a, ds_b]):
        orchestrator = ContinualOrchestrator(config)
        loaded = orchestrator.load_dataset()

    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == {"A", "B"}
    assert isinstance(loaded["A"], DatasetDict)
    assert isinstance(loaded["B"], DatasetDict)
    assert "train" in loaded["A"]
    assert "train" in loaded["B"]


def test_load_dataset_rejects_sources_when_mixture_disabled() -> None:
    config = _base_config()
    config.dataset.sources = [{"dataset_id": "A", "nameOrPath": "/tmp/a"}]
    config.dataset.mixture = {"enabled": False}

    orchestrator = ContinualOrchestrator(config)
    with pytest.raises(ValueError, match="dataset.sources is set but dataset.mixture.enabled is false"):
        orchestrator.load_dataset()


def test_load_dataset_rejects_mixture_enabled_without_sources() -> None:
    config = _base_config()
    config.dataset.nameOrPath = "/tmp/single"
    config.dataset.mixture = {"enabled": True}

    orchestrator = ContinualOrchestrator(config)
    with pytest.raises(ValueError, match="dataset.mixture.enabled is true but dataset.sources is missing or empty"):
        orchestrator.load_dataset()


def test_load_dataset_rejects_source_entry_without_dataset_id() -> None:
    config = _base_config()
    config.dataset.sources = [{"nameOrPath": "/tmp/a"}]
    config.dataset.mixture = {"enabled": True}

    orchestrator = ContinualOrchestrator(config)
    with pytest.raises(ValueError, match="must define dataset_id"):
        orchestrator.load_dataset()


def test_load_dataset_single_source_path_keeps_existing_behavior() -> None:
    config = _base_config()
    config.dataset.nameOrPath = "/tmp/single"

    ds_single = Dataset.from_dict({"input_ids": [[1, 2]], "length": [2]})
    with patch("src.tasks.training.orchestrator.DatasetStorage.load_from_disk", return_value=ds_single):
        orchestrator = ContinualOrchestrator(config)
        loaded = orchestrator.load_dataset()

    assert isinstance(loaded, DatasetDict)
    assert "train" in loaded
