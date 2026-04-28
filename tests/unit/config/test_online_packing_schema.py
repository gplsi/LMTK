from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.config_loader import ConfigValidator


def _minimal_clm_training_config() -> dict:
    return {
        "task": "clm_training",
        "experiment_name": "test_training_schema",
        "verbose_level": 1,
        "model_name": "dummy",
        "precision": "bf16-true",
        "seed": 42,
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
    with pytest.raises(ValueError):
        validator.validate(config_path, "clm_training")


def test_clm_training_schema_accepts_optimizer_no_decay_norms(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_optimizer_no_decay_norms.yaml"
    config = _minimal_clm_training_config()
    config["optimizer_no_decay_norms"] = True
    config["dataset"] = {
        "source": "local",
        "nameOrPath": "/tmp/does-not-need-to-exist-for-validation",
        "format": "hf",
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "clm_training")
    assert validated.optimizer_no_decay_norms is True


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


def test_training_mixture_schema_accepts_sources_without_manual_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_sources.yaml"
    config = {
        "task": "clm_training",
        "experiment_name": "test_training_mixture_sources",
        "verbose_level": 1,
        "model_name": "dummy",
        "precision": "bf16-true",
        "seed": 42,
        "dataset": {
            "source": "local",
            "format": "hf",
            "sources": [
                {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
                {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
            ],
            "packing": {"enabled": True, "sequence_length": 128},
            "mixture": {"enabled": True, "budget_mode": "anchor_epochs", "anchor_epochs": 1},
        },
        "validation_split": {"proportion": 0.1, "seed": 42, "shuffle": True},
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
    assert validated.dataset.mixture.enabled is True


def test_training_mixture_schema_defaults_to_anchor_epochs_when_budget_mode_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_default_anchor_budget.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_default_anchor_budget"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
            {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True, "anchor_epochs": 1},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "clm_training")
    assert validated.dataset.mixture.anchor_epochs == 1
    assert validated.dataset.mixture.get("budget_mode", None) is None


def test_training_mixture_schema_accepts_alignment_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_alignment_policy.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_alignment_policy"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
            {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {
            "enabled": True,
            "budget_mode": "anchor_epochs",
            "anchor_epochs": 1,
            "alignment_policy": "floor",
        },
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "clm_training")
    assert validated.dataset.mixture.alignment_policy == "floor"


def test_training_mixture_schema_rejects_invalid_alignment_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_invalid_alignment_policy.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_invalid_alignment_policy"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
            {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {
            "enabled": True,
            "budget_mode": "anchor_epochs",
            "anchor_epochs": 1,
            "alignment_policy": "invalid",
        },
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="Configuration validation failed using schema"):
        validator.validate(config_path, "clm_training")


def test_training_mixture_schema_rejects_default_anchor_budget_without_anchor_epochs(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_default_anchor_budget_missing_anchor_epochs.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_default_anchor_budget_missing_anchor_epochs"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
            {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="anchor_epochs"):
        validator.validate(config_path, "clm_training")


def test_training_mixture_schema_rejects_single_and_multi_source_together(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_conflict.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_conflict"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "nameOrPath": "/tmp/single",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True, "budget_mode": "explicit_blocks", "total_blocks": 4},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="Configuration validation failed using schema"):
        validator.validate(config_path, "clm_training")


def test_training_mixture_schema_rejects_mixture_without_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_without_sources.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_without_sources"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "nameOrPath": "/tmp/single",
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True, "budget_mode": "explicit_blocks", "total_blocks": 4},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="Configuration validation failed using schema"):
        validator.validate(config_path, "clm_training")


def test_training_mixture_schema_rejects_duplicate_dataset_id(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_duplicate_dataset_id.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_duplicate_dataset_id"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a", "weight": 1.0},
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_b", "weight": 1.0},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True, "budget_mode": "explicit_blocks", "total_blocks": 4},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="duplicate dataset_id 'A'"):
        validator.validate(config_path, "clm_training")


def test_training_mixture_schema_rejects_mixed_weight_presence(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_mixture_mixed_weights.yaml"
    config = _minimal_clm_training_config()
    config["experiment_name"] = "test_training_mixture_mixed_weights"
    config["dataset"] = {
        "source": "local",
        "format": "hf",
        "sources": [
            {"dataset_id": "A", "nameOrPath": "/tmp/ds_a", "weight": 1.0},
            {"dataset_id": "B", "nameOrPath": "/tmp/ds_b"},
        ],
        "packing": {"enabled": True, "sequence_length": 128},
        "mixture": {"enabled": True, "budget_mode": "explicit_blocks", "total_blocks": 4},
    }
    config["validation_split"] = {"proportion": 0.1, "seed": 42, "shuffle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="all sources must either set weight or omit weight"):
        validator.validate(config_path, "clm_training")


def test_training_schema_accepts_min_lr_and_max_lr(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_lr_bounds_ok.yaml"
    config = _minimal_clm_training_config()
    config["dataset"] = {
        "source": "local",
        "nameOrPath": "/tmp/single",
        "format": "hf",
    }
    config["min_lr"] = 1e-6
    config["max_lr"] = 3e-4
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    validated = validator.validate(config_path, "clm_training")
    assert validated.min_lr == 1e-6
    assert validated.max_lr == 3e-4


def test_training_schema_rejects_min_lr_greater_than_peak_lr(tmp_path: Path) -> None:
    config_path = tmp_path / "clm_training_lr_bounds_invalid.yaml"
    config = _minimal_clm_training_config()
    config["dataset"] = {
        "source": "local",
        "nameOrPath": "/tmp/single",
        "format": "hf",
    }
    config["lr"] = 1e-4
    config["min_lr"] = 2e-4
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    validator = ConfigValidator()
    with pytest.raises(ValueError, match="must be <= effective peak LR"):
        validator.validate(config_path, "clm_training")
