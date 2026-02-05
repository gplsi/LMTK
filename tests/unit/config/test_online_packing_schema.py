from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.config_loader import ConfigValidator


def test_tokenization_clm_doclevel_allows_omitting_overlap(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenization_doclevel.yaml"
    config = {
        "task": "tokenization",
        "experiment_name": "test_tokenization_doclevel_schema",
        "verbose_level": 1,
        "seed": 42,
        "tokenizer": {
            "tokenizer_name": "dummy",
            "task": "clm_training",
            # Intentionally omit context_length/max_sequence_length and overlap.
        },
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
            "format": "files",
            "file_config": {"format": "txt"},
        },
        "output": {"path": "/tmp/out"},
        "test_size": 0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "tokenization")

    assert validated.tokenizer.task == "clm_training"


def test_training_packing_requires_sequence_length_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_packing_missing_len.yaml"
    config = {
        "task": "clm_training",
        "experiment_name": "test_training_packing_missing_len",
        "verbose_level": 1,
        "model_name": "dummy",
        "precision": "bf16-true",
        "seed": 42,
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
            "format": "hf",
            "packing": {"enabled": True},
        },
        "number_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "validate_after_epoch": False,
        "validate_on_end": False,
        "save_on_validate": False,
        "save_on_end": False,
        "output_dir": "/tmp/out",
        "lr": 1e-4,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError):
        validator.validate(config_path, "clm_training")


def test_training_packing_accepts_optional_index_cache_and_drop_last_batch(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_packing_optional_fields.yaml"
    config = {
        "task": "clm_training",
        "experiment_name": "test_training_packing_optional_fields",
        "verbose_level": 1,
        "model_name": "dummy",
        "precision": "bf16-true",
        "seed": 42,
        "dataset": {
            "source": "local",
            "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
            "format": "hf",
            "packing": {
                "enabled": True,
                "sequence_length": 128,
                "index_cache_dir": "/tmp/packing_index",
                "drop_last_batch": True,
            },
        },
        "number_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "validate_after_epoch": False,
        "validate_on_end": False,
        "save_on_validate": False,
        "save_on_end": False,
        "output_dir": "/tmp/out",
        "lr": 1e-4,
        "lr_scheduler": "fixed",
        "warmup_proportion": 0.0,
        "gradient_accumulation": False,
        "gradient_accumulation_steps": 1,
        "lr_decay": False,
        "parallelization_strategy": "dp",
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "clm_training")
    assert validated.dataset.packing.enabled is True
