from __future__ import annotations

import pytest
import torch

from src.tasks.training.data.mixture import (
    MixturePackedDataset,
    allocate_exact_counts,
    blake2b_u64,
    build_expected_replay_summary,
    build_mixture_progress_metrics,
    build_replay_metrics,
    derive_anchor_epoch_targets,
    permute_index,
    resolve_aligned_total_blocks,
    resolve_effective_total_blocks,
    resolve_effective_weights,
    resolve_mixture_plan,
)


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, value: int, n: int) -> None:
        self.value = int(value)
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([self.value, idx], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([self.value, idx], dtype=torch.long),
        }


def test_allocate_exact_counts_uses_lexicographic_tie_break() -> None:
    counts = allocate_exact_counts(2, {"B": 1.0, "A": 1.0, "C": 1.0})
    assert counts == {"A": 1, "B": 1, "C": 0}


def test_resolve_effective_weights_manual_and_derived() -> None:
    manual, mode_manual = resolve_effective_weights(
        configured_weights_by_id={"A": 2.0, "B": 1.0},
        source_blocks={"A": 100, "B": 50},
    )
    assert mode_manual == "manual"
    assert manual == {"A": 2.0, "B": 1.0}

    derived, mode_derived = resolve_effective_weights(
        configured_weights_by_id={"A": None, "B": None},
        source_blocks={"A": 100, "B": 50},
    )
    assert mode_derived == "from_source_blocks"
    assert derived == {"A": 100.0, "B": 50.0}


def test_resolve_effective_weights_fails_on_mixed_presence() -> None:
    with pytest.raises(ValueError, match="either set weight or omit weight"):
        resolve_effective_weights(
            configured_weights_by_id={"A": 2.0, "B": None},
            source_blocks={"A": 100, "B": 50},
        )


def test_resolve_effective_weights_fails_on_zero_blocks() -> None:
    with pytest.raises(ValueError, match="zero train packed blocks"):
        resolve_effective_weights(
            configured_weights_by_id={"A": None, "B": None},
            source_blocks={"A": 100, "B": 0},
        )


def test_derive_anchor_epoch_targets_matches_expected_ratios() -> None:
    targets = derive_anchor_epoch_targets(
        source_blocks={"A": 100, "B": 60, "C": 40},
        weights_by_id={"A": 0.5, "B": 0.3, "C": 0.2},
        anchor_id="A",
    )
    assert targets == {"A": 100, "B": 60, "C": 40}


def test_derive_anchor_epoch_targets_matches_issue43_hamilton_on_raw_targets() -> None:
    # Regression: in anchor_epochs mode, non-anchor allocation must use Hamilton on the raw targets
    # without renormalizing them (issue 43 exact algorithm).
    targets = derive_anchor_epoch_targets(
        source_blocks={"Z": 1, "A": 1, "B": 1},
        weights_by_id={"Z": 1.0, "A": 0.5732012745071716, "B": 1.7025322218620156},
        anchor_id="Z",
    )
    assert targets == {"Z": 1, "A": 0, "B": 2}


def test_resolve_mixture_plan_anchor_epochs_defaults_to_largest_source() -> None:
    plan = resolve_mixture_plan(
        source_blocks={"A": 300, "B": 100},
        configured_weights_by_id={"A": None, "B": None},
        budget_mode="anchor_epochs",
        anchor_epochs=2,
        anchor_dataset_id=None,
        explicit_total_blocks=None,
    )
    assert plan.anchor_dataset_id == "A"
    assert plan.requested_total_blocks == sum(plan.target_blocks_per_dataset.values())
    assert plan.weight_mode == "from_source_blocks"


def test_resolve_mixture_plan_explicit_blocks_uses_hamilton_allocation() -> None:
    plan = resolve_mixture_plan(
        source_blocks={"A": 300, "B": 100},
        configured_weights_by_id={"A": 2.0, "B": 1.0},
        budget_mode="explicit_blocks",
        anchor_epochs=None,
        anchor_dataset_id=None,
        explicit_total_blocks=10,
    )
    assert plan.target_blocks_per_dataset == {"A": 7, "B": 3}
    assert plan.requested_total_blocks == 10


def test_build_mixture_progress_metrics_reports_cumulative_blocks_and_resampling_ratio() -> None:
    metrics = build_mixture_progress_metrics(
        target_blocks={"A": 10, "B": 5},
        realized_blocks={"A": 8, "B": 7},
        effective_total_blocks=15,
    )

    assert metrics["mixture/target_blocks_A"] == 10.0
    assert metrics["mixture/realized_blocks_A"] == 8.0
    assert metrics["mixture/deviation_blocks_A"] == -2.0
    assert metrics["mixture/target_ratio_A"] == pytest.approx(10.0 / 15.0)
    assert metrics["mixture/realized_ratio_A"] == pytest.approx(8.0 / 15.0)
    assert metrics["mixture/resampling_ratio_A"] == pytest.approx(0.8)

    assert metrics["mixture/target_blocks_B"] == 5.0
    assert metrics["mixture/realized_blocks_B"] == 7.0
    assert metrics["mixture/deviation_blocks_B"] == 2.0
    assert metrics["mixture/target_ratio_B"] == pytest.approx(5.0 / 15.0)
    assert metrics["mixture/realized_ratio_B"] == pytest.approx(7.0 / 15.0)
    assert metrics["mixture/resampling_ratio_B"] == pytest.approx(1.4)


def test_build_expected_replay_summary_matches_unique_first_contract() -> None:
    summary = build_expected_replay_summary(
        target_blocks={"A": 6, "B": 2},
        source_blocks_available={"A": 3, "B": 5},
    )
    per_source = summary["per_source"]
    totals = summary["totals"]

    assert per_source["A"]["expected_unique_blocks"] == 3
    assert per_source["A"]["expected_replayed_draws"] == 3
    assert per_source["A"]["expected_replay_fraction"] == pytest.approx(0.5)

    assert per_source["B"]["expected_unique_blocks"] == 2
    assert per_source["B"]["expected_replayed_draws"] == 0
    assert per_source["B"]["expected_replay_fraction"] == pytest.approx(0.0)

    assert totals["target_blocks"] == 8
    assert totals["expected_unique_blocks"] == 5
    assert totals["expected_replayed_draws"] == 3
    assert totals["expected_replay_fraction"] == pytest.approx(3.0 / 8.0)


def test_build_replay_metrics_reports_unique_replay_and_totals() -> None:
    metrics = build_replay_metrics(
        realized_blocks={"A": 10, "B": 5},
        replayed_draws={"A": 2, "B": 0},
        namespace="mixture/replay",
    )
    assert metrics["mixture/replay/unique_blocks_A"] == 8.0
    assert metrics["mixture/replay/replayed_draws_A"] == 2.0
    assert metrics["mixture/replay/fraction_A"] == pytest.approx(0.2)
    assert metrics["mixture/replay/unique_blocks_B"] == 5.0
    assert metrics["mixture/replay/replayed_draws_B"] == 0.0
    assert metrics["mixture/replay/fraction_B"] == pytest.approx(0.0)
    assert metrics["mixture/replay/unique_blocks_total"] == 13.0
    assert metrics["mixture/replay/replayed_draws_total"] == 2.0
    assert metrics["mixture/replay/fraction_total"] == pytest.approx(2.0 / 15.0)


def test_permute_index_is_a_full_permutation() -> None:
    total = 17
    permuted = [permute_index(i, total, schedule_seed=42) for i in range(total)]
    assert sorted(permuted) == list(range(total))


def test_mixture_dataset_is_deterministic_and_emits_dataset_idx() -> None:
    datasets_by_id = {
        "A": _ToyDataset(1, 3),
        "B": _ToyDataset(2, 2),
    }
    counts_by_id = {"A": 6, "B": 4}

    ds1 = MixturePackedDataset(
        datasets_by_id=datasets_by_id,
        counts_by_id=counts_by_id,
        schedule_seed=123,
    )
    ds2 = MixturePackedDataset(
        datasets_by_id=datasets_by_id,
        counts_by_id=counts_by_id,
        schedule_seed=123,
    )

    seq1 = [int(ds1[i]["dataset_idx"]) for i in range(len(ds1))]
    seq2 = [int(ds2[i]["dataset_idx"]) for i in range(len(ds2))]

    assert len(ds1) == 10
    assert seq1 == seq2
    assert sum(1 for x in seq1 if x == ds1.dataset_id_to_idx["A"]) == 6
    assert sum(1 for x in seq1 if x == ds1.dataset_id_to_idx["B"]) == 4

    sample = ds1[0]
    assert sample["dataset_idx"].dtype == torch.int64
    assert sample["mixture_is_replay"].dtype == torch.int64
    assert sample["mixture_source_local_idx"].dtype == torch.int64
    assert not hasattr(ds1, "schedule")


def test_mixture_dataset_unique_first_without_replay_when_target_within_capacity() -> None:
    datasets_by_id = {
        "A": _ToyDataset(1, 3),
        "B": _ToyDataset(2, 2),
    }
    counts_by_id = {"A": 3, "B": 2}
    ds = MixturePackedDataset(
        datasets_by_id=datasets_by_id,
        counts_by_id=counts_by_id,
        schedule_seed=7,
    )

    per_source_local_indices: dict[str, list[int]] = {"A": [], "B": []}
    per_source_replay: dict[str, int] = {"A": 0, "B": 0}
    idx_to_id = ds.dataset_idx_to_id
    for i in range(len(ds)):
        sample = ds[i]
        dataset_id = idx_to_id[int(sample["dataset_idx"].item())]
        per_source_local_indices[dataset_id].append(int(sample["mixture_source_local_idx"].item()))
        per_source_replay[dataset_id] += int(sample["mixture_is_replay"].item())

    assert per_source_replay == {"A": 0, "B": 0}
    assert len(set(per_source_local_indices["A"])) == 3
    assert len(set(per_source_local_indices["B"])) == 2


def test_mixture_dataset_replays_only_on_overflow() -> None:
    datasets_by_id = {"A": _ToyDataset(1, 3)}
    counts_by_id = {"A": 6}
    ds = MixturePackedDataset(
        datasets_by_id=datasets_by_id,
        counts_by_id=counts_by_id,
        schedule_seed=9,
    )

    replay_flags = []
    local_indices = []
    for i in range(len(ds)):
        sample = ds[i]
        replay_flags.append(int(sample["mixture_is_replay"].item()))
        local_indices.append(int(sample["mixture_source_local_idx"].item()))

    assert sum(replay_flags) == 3
    # unique-first guarantees all 3 source-local indices appear before replaying overflow
    assert len(set(local_indices)) == 3


def test_blake2b_u64_is_stable() -> None:
    assert blake2b_u64("abc") == blake2b_u64("abc")
    assert blake2b_u64("abc") != blake2b_u64("abd")


def test_resolve_effective_total_blocks_passes_when_exact_budget_is_executable() -> None:
    effective = resolve_effective_total_blocks(
        requested_total_blocks=12,
        world_size=2,
        batch_size=3,
        sampler_drop_last=True,
        drop_last_batch=True,
    )
    assert effective == 12


def test_resolve_effective_total_blocks_fails_when_drop_last_batch_would_drop_blocks() -> None:
    with pytest.raises(ValueError, match="executed blocks differ from requested blocks"):
        resolve_effective_total_blocks(
            requested_total_blocks=10,
            world_size=2,
            batch_size=3,
            sampler_drop_last=True,
            drop_last_batch=True,
        )


def test_resolve_aligned_total_blocks_anchor_mode_defaults_to_floor() -> None:
    adjusted, meta = resolve_aligned_total_blocks(
        requested_total_blocks=384546,
        budget_mode="anchor_epochs",
        world_size=8,
        batch_size=1,
        gradient_accumulation_steps=16,
        alignment_policy=None,
    )
    assert adjusted == 384512
    assert meta["policy"] == "floor"
    assert meta["alignment_unit"] == 128
    assert bool(meta["alignment_applied"]) is True


def test_resolve_aligned_total_blocks_explicit_mode_is_strict_by_default() -> None:
    with pytest.raises(ValueError, match="strict alignment"):
        resolve_aligned_total_blocks(
            requested_total_blocks=384546,
            budget_mode="explicit_blocks",
            world_size=8,
            batch_size=1,
            gradient_accumulation_steps=16,
            alignment_policy=None,
        )


def test_resolve_aligned_total_blocks_rejects_non_error_policy_in_explicit_mode() -> None:
    with pytest.raises(ValueError, match="must be 'error'"):
        resolve_aligned_total_blocks(
            requested_total_blocks=384546,
            budget_mode="explicit_blocks",
            world_size=8,
            batch_size=1,
            gradient_accumulation_steps=16,
            alignment_policy="floor",
        )
