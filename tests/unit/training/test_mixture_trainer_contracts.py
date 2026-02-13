from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from box import Box

pytest.importorskip("datasets")
pytest.importorskip("lightning")
from datasets import Dataset, DatasetDict

from src.tasks.training.data.mixture import MixturePlan
from src.tasks.training.fabric.trainer.base import (
    FabricTrainerBase,
    MIXTURE_META_VERSION,
    MIXTURE_RUNTIME_VERSION,
    MIXTURE_SAMPLING_ALGORITHM,
)
from src.utils.dataset import build_tokenization_metadata, write_tokenization_metadata


class _NoopLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def debug(self, *args, **kwargs) -> None:
        return None


class _DummyTrainer(FabricTrainerBase):
    def __init__(self) -> None:
        # Intentionally avoid FabricTrainerBase.__init__ to unit-test helpers in isolation.
        pass

    def _setup_strategy(self):  # pragma: no cover - unused in these tests
        raise NotImplementedError


@dataclass
class _FabricStub:
    world_size: int


def _make_trainer_for_contract_tests() -> _DummyTrainer:
    trainer = _DummyTrainer()
    trainer.cli_logger = _NoopLogger()
    trainer.config = Box(
        {
            "task": "clm_training",
            "experiment_name": "mixture-contract-tests",
            "model_name": "dummy-model",
            "precision": "bf16-true",
            "seed": 123,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "number_epochs": 1,
            "output_dir": "/tmp/out",
            "dataset": {
                "packing": {
                    "enabled": True,
                    "sequence_length": 8,
                    "tokenizer_name": "tok-a",
                    "eos_token_id": 2,
                }
            },
        },
        box_dots=True,
    )
    return trainer


def _write_dataset_metadata(dataset_path: Path, eos_token_id: int) -> None:
    payload = build_tokenization_metadata(
        tokenizer_name="/models/local-tokenizer",
        eos_token_id=eos_token_id,
        task="clm_training",
        created_by="unit-test",
    )
    write_tokenization_metadata(dataset_path, payload)


def _configure_mixture_resume_trainer(trainer: _DummyTrainer) -> None:
    trainer._mixture_enabled = True
    trainer._mixture_plan = MixturePlan(
        budget_mode="explicit_blocks",
        requested_total_blocks=12,
        target_blocks_per_dataset={"A": 6, "B": 6},
        source_blocks_by_id={"A": 100, "B": 100},
        weight_mode="manual",
        effective_weights_by_id={"A": 1.0, "B": 1.0},
        anchor_dataset_id=None,
    )
    trainer._mixture_configured_weights_by_id = {"A": 1.0, "B": 1.0}
    trainer._mixture_requested_total_blocks = 12
    trainer._mixture_effective_total_blocks = 12
    trainer._mixture_schedule_seed = 123
    trainer._mixture_dataset_idx_to_id = {0: "A", 1: "B"}
    trainer._mixture_dataset_id_to_idx = {"A": 0, "B": 1}
    trainer._mixture_realized_blocks_local = {"A": 0, "B": 0}
    trainer._mixture_replayed_draws_local = {"A": 0, "B": 0}
    trainer._mixture_anchor_window_start_realized_local = {"A": 0, "B": 0}
    trainer._mixture_anchor_window_start_replayed_local = {"A": 0, "B": 0}
    trainer._mixture_anchor_epoch_index = 0
    trainer._mixture_global_blocks_seen_estimate = 0
    trainer._mixture_last_val_losses = {}
    trainer._mixture_last_val_weighted = None
    trainer.state = {}


def test_validate_mixture_source_compatibility_rejects_mismatched_tokenizer_name() -> None:
    trainer = _make_trainer_for_contract_tests()
    source_map = {
        "A": Box({"dataset_id": "A", "tokenizer_name": "tok-a", "eos_token_id": 2}, box_dots=True),
        "B": Box({"dataset_id": "B", "tokenizer_name": "tok-b", "eos_token_id": 2}, box_dots=True),
    }
    with pytest.raises(ValueError, match="Incompatible source tokenizers"):
        trainer._validate_mixture_source_compatibility(source_map)


def test_validate_mixture_source_compatibility_rejects_mismatched_eos_token_id() -> None:
    trainer = _make_trainer_for_contract_tests()
    source_map = {
        "A": Box({"dataset_id": "A", "tokenizer_name": "tok-a", "eos_token_id": 2}, box_dots=True),
        "B": Box({"dataset_id": "B", "tokenizer_name": "tok-a", "eos_token_id": 3}, box_dots=True),
    }
    with pytest.raises(ValueError, match="Incompatible source eos_token_id"):
        trainer._validate_mixture_source_compatibility(source_map)


def test_validate_mixture_source_compatibility_rejects_packing_tokenizer_mismatch() -> None:
    trainer = _make_trainer_for_contract_tests()
    source_map = {
        "A": Box({"dataset_id": "A", "tokenizer_name": "tok-z", "eos_token_id": 2}, box_dots=True),
    }
    with pytest.raises(ValueError, match="dataset.packing.tokenizer_name does not match"):
        trainer._validate_mixture_source_compatibility(source_map)


def test_resolve_eos_token_id_uses_packing_value_first() -> None:
    trainer = _make_trainer_for_contract_tests()
    eos = trainer._resolve_eos_token_id(trainer.config.dataset.packing)
    assert eos == 2


def test_resolve_eos_token_id_uses_source_eos_when_packing_missing() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config.dataset.packing.eos_token_id = None
    trainer.config.dataset.sources = [
        {"dataset_id": "A", "nameOrPath": "/tmp/a", "eos_token_id": 7},
        {"dataset_id": "B", "nameOrPath": "/tmp/b", "eos_token_id": 7},
    ]
    eos = trainer._resolve_eos_token_id(trainer.config.dataset.packing)
    assert eos == 7


def test_resolve_eos_token_id_uses_single_dataset_metadata(tmp_path: Path) -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config.dataset = Box(
        {
            "source": "local",
            "nameOrPath": str(tmp_path),
            "packing": {
                "enabled": True,
                "sequence_length": 8,
                "tokenizer_name": "tok-a",
                "eos_token_id": None,
            },
        },
        box_dots=True,
    )
    _write_dataset_metadata(tmp_path, eos_token_id=11)

    eos = trainer._resolve_eos_token_id(trainer.config.dataset.packing)
    assert eos == 11


def test_resolve_eos_token_id_rejects_mixture_metadata_mismatch(tmp_path: Path) -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config.dataset.packing.eos_token_id = None

    ds_a = tmp_path / "A"
    ds_b = tmp_path / "B"
    ds_a.mkdir(parents=True, exist_ok=True)
    ds_b.mkdir(parents=True, exist_ok=True)
    _write_dataset_metadata(ds_a, eos_token_id=2)
    _write_dataset_metadata(ds_b, eos_token_id=3)

    trainer.config.dataset.sources = [
        {"dataset_id": "A", "nameOrPath": str(ds_a)},
        {"dataset_id": "B", "nameOrPath": str(ds_b)},
    ]

    with pytest.raises(ValueError, match="Incompatible eos_token_id values found in source dataset metadata"):
        trainer._resolve_eos_token_id(trainer.config.dataset.packing)


def test_resolve_eos_token_id_fails_when_unresolved_without_network() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config.dataset.packing.eos_token_id = None
    trainer.config.dataset.sources = []
    trainer.config.dataset.nameOrPath = "/tmp/does-not-exist"

    with pytest.raises(ValueError, match="could not be resolved without network access"):
        trainer._resolve_eos_token_id(trainer.config.dataset.packing)


def test_mixture_resume_meta_roundtrip_and_mismatch_fail_fast() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    meta = trainer._build_mixture_resume_meta(_FabricStub(world_size=2))
    assert meta["mixture_meta_version"] == MIXTURE_META_VERSION
    assert meta["requested_total_blocks"] == 12
    assert meta["effective_total_blocks"] == 12
    assert meta["world_size"] == 2
    assert meta["batch_size"] == 2
    assert meta["gradient_accumulation_steps"] == 4
    assert meta["sampling_algorithm"] == MIXTURE_SAMPLING_ALGORITHM
    assert meta["allocation_algorithm"] == "hamilton_lr_lexicographic_v1"
    assert meta["dataset_index_map"] == {"0": "A", "1": "B"}

    trainer.state["mixture_meta"] = dict(meta)
    trainer._validate_mixture_resume_compatibility(meta)

    bad_meta = dict(meta)
    bad_meta["world_size"] = 4
    trainer.state["mixture_meta"] = bad_meta
    with pytest.raises(ValueError, match="Mixture resume compatibility check failed"):
        trainer._validate_mixture_resume_compatibility(meta)


def test_mixture_resume_meta_requires_dict_shape() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)
    trainer.state = {"mixture_meta": "invalid-meta"}

    expected_meta = trainer._build_mixture_resume_meta(_FabricStub(world_size=2))
    with pytest.raises(ValueError, match="mixture_meta' to be a dictionary"):
        trainer._validate_mixture_resume_compatibility(expected_meta)


def test_mixture_resume_meta_rejects_legacy_checkpoint_without_version_and_sampling() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    expected_meta = trainer._build_mixture_resume_meta(_FabricStub(world_size=2))
    legacy_meta = {
        key: expected_meta[key]
        for key in (
            "weight_mode",
            "sources",
            "budget_mode",
            "anchor_dataset_id",
            "requested_total_blocks",
            "world_size",
            "batch_size",
            "gradient_accumulation_steps",
            "schedule_seed",
            "hash_algorithm",
            "allocation_algorithm",
            "split_seed_algorithm",
        )
    }
    trainer.state["mixture_meta"] = legacy_meta
    with pytest.raises(ValueError, match="mixture_meta_version mismatch"):
        trainer._validate_mixture_resume_compatibility(expected_meta)


def test_mixture_resume_meta_rejects_invalid_dataset_index_map() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    expected_meta = trainer._build_mixture_resume_meta(_FabricStub(world_size=2))
    bad_meta = dict(expected_meta)
    bad_meta["dataset_index_map"] = {"0": "A", "1": "A"}
    trainer.state["mixture_meta"] = bad_meta
    with pytest.raises(ValueError, match="dataset_index_map"):
        trainer._validate_mixture_resume_compatibility(expected_meta)


def test_restore_mixture_runtime_state_restores_counters_and_losses() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    trainer._restore_mixture_runtime_state(
        {
            "mixture_runtime_version": MIXTURE_RUNTIME_VERSION,
            "realized_blocks_local": {"A": 11, "B": 7},
            "replayed_draws_local": {"A": 3, "B": 1},
            "anchor_epoch_index": 2,
            "global_blocks_seen_estimate": 99,
            "anchor_window_start_realized_local": {"A": 5, "B": 3},
            "anchor_window_start_replayed_local": {"A": 1, "B": 1},
            "last_val_losses": {"A": 2.1, "B": 2.4},
            "last_val_weighted": 2.25,
        },
        strict=True,
    )

    assert trainer._mixture_realized_blocks_local == {"A": 11, "B": 7}
    assert trainer._mixture_replayed_draws_local == {"A": 3, "B": 1}
    assert trainer._mixture_anchor_epoch_index == 2
    assert trainer._mixture_global_blocks_seen_estimate == 99
    assert trainer._mixture_anchor_window_start_realized_local == {"A": 5, "B": 3}
    assert trainer._mixture_anchor_window_start_replayed_local == {"A": 1, "B": 1}
    assert trainer._mixture_last_val_losses == {"A": 2.1, "B": 2.4}
    assert trainer._mixture_last_val_weighted == 2.25


def test_restore_mixture_runtime_state_strict_requires_runtime_version_match() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)
    with pytest.raises(ValueError, match="mixture_runtime_version mismatch"):
        trainer._restore_mixture_runtime_state(
            {
                "mixture_runtime_version": "legacy",
                "realized_blocks_local": {"A": 0, "B": 0},
                "replayed_draws_local": {"A": 0, "B": 0},
                "anchor_window_start_realized_local": {"A": 0, "B": 0},
                "anchor_window_start_replayed_local": {"A": 0, "B": 0},
                "last_val_losses": {},
                "last_val_weighted": None,
            },
            strict=True,
        )


def test_restore_mixture_runtime_state_strict_mode_requires_runtime_payload() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    with pytest.raises(ValueError, match="mixture_runtime"):
        trainer._restore_mixture_runtime_state(None, strict=True)


def test_restore_mixture_runtime_state_legacy_mode_allows_missing_runtime_payload() -> None:
    trainer = _make_trainer_for_contract_tests()
    _configure_mixture_resume_trainer(trainer)

    trainer._restore_mixture_runtime_state(None, strict=False)
    assert trainer._mixture_realized_blocks_local == {"A": 0, "B": 0}


def test_build_run_metadata_contains_stable_training_fields() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer._mixture_enabled = False
    trainer._mixture_plan = None
    trainer._training_schedule_metadata = {
        "optimizer_steps_per_epoch": 12,
        "total_optimizer_steps": 12,
        "warmup_steps": 1,
        "min_lr": 1e-6,
        "peak_lr": 3e-4,
        "peak_lr_source": "max_lr",
    }
    metadata = trainer._build_run_metadata(_FabricStub(world_size=2))
    assert metadata["run_metadata_version"] == "v1"
    assert metadata["task"] == "clm_training"
    assert metadata["experiment_name"] == "mixture-contract-tests"
    assert metadata["world_size"] == 2
    assert metadata["gradient_accumulation_steps"] == 4
    assert metadata["training_schedule"]["total_optimizer_steps"] == 12


def test_ensure_validation_split_require_valid_fails_without_validation_split_config() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config = Box({"seed": 123}, box_dots=True)

    dataset = Dataset.from_dict({"input_ids": [[1], [2], [3]], "length": [1, 1, 1]})
    with pytest.raises(ValueError, match="Missing validation data"):
        trainer._ensure_validation_split(dataset, require_valid=True, source_id="A")


def test_ensure_validation_split_is_deterministic_with_seed_override() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config = Box(
        {
            "seed": 123,
            "validation_split": {"proportion": 0.4, "shuffle": True},
        },
        box_dots=True,
    )

    base = Dataset.from_dict(
        {
            "example_id": list(range(10)),
            "input_ids": [[i, i + 1] for i in range(10)],
            "length": [2] * 10,
        }
    )

    ds1 = trainer._ensure_validation_split(DatasetDict({"train": base}), split_seed_override=77, source_id="A")
    ds2 = trainer._ensure_validation_split(DatasetDict({"train": base}), split_seed_override=77, source_id="A")

    assert ds1["valid"]["example_id"] == ds2["valid"]["example_id"]
    assert ds1["train"]["example_id"] == ds2["train"]["example_id"]


def test_collect_source_config_map_rejects_duplicate_dataset_id() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config = Box(
        {
            "dataset": {
                "sources": [
                    {"dataset_id": "A", "nameOrPath": "/tmp/a"},
                    {"dataset_id": "A", "nameOrPath": "/tmp/b"},
                ]
            }
        },
        box_dots=True,
    )
    with pytest.raises(ValueError, match="Duplicate dataset_id"):
        trainer._collect_source_config_map()


def test_load_fabric_datasets_rejects_mixture_enabled_without_sources() -> None:
    trainer = _make_trainer_for_contract_tests()
    trainer.config.dataset.mixture = Box({"enabled": True}, box_dots=True)

    with pytest.raises(ValueError, match="dataset.mixture.enabled is true but dataset.sources is missing or empty"):
        trainer._load_fabric_datasets_dataloaders(trainer.config, {})


def test_derive_validation_split_seed_is_stable_and_source_specific() -> None:
    trainer = _make_trainer_for_contract_tests()
    seed_a_1 = trainer._derive_validation_split_seed("A")
    seed_a_2 = trainer._derive_validation_split_seed("A")
    seed_b = trainer._derive_validation_split_seed("B")
    assert seed_a_1 == seed_a_2
    assert seed_a_1 != seed_b
    assert 0 <= seed_a_1 <= 2147483646
