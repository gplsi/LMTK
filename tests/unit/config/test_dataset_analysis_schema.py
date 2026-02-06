from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.config_loader import ConfigValidator


def test_dataset_analysis_schema_accepts_minimal_valid_config(tmp_path: Path) -> None:
    config = {
        "task": "dataset_analysis",
        "experiment_name": "test_dataset_analysis_schema",
        "verbose_level": 1,
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/tokenized",
            "format": "hf",
        },
        "output": {"path": "/tmp/reports"},
    }
    config_path = tmp_path / "dataset_analysis_valid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "dataset_analysis")

    assert validated.task == "dataset_analysis"
    assert validated.dataset.source == "local"


def test_dataset_analysis_schema_rejects_missing_output_path(tmp_path: Path) -> None:
    config = {
        "task": "dataset_analysis",
        "experiment_name": "test_dataset_analysis_schema_invalid",
        "verbose_level": 1,
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/tokenized",
            "format": "hf",
        },
        "output": {},
    }
    config_path = tmp_path / "dataset_analysis_invalid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError):
        validator.validate(config_path, "dataset_analysis")
