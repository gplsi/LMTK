from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MixturePlan:
    budget_mode: str
    requested_total_blocks: int
    target_blocks_per_dataset: dict[str, int]
    source_blocks_by_id: dict[str, int]
    weight_mode: str
    effective_weights_by_id: dict[str, float]
    anchor_dataset_id: str | None


def blake2b_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _compute_stride_offset(*, total: int, schedule_seed: int) -> tuple[int, int]:
    if total <= 0:
        raise ValueError("total must be > 0.")

    stride = (blake2b_u64(f"perm_stride|{schedule_seed}") % total) or 1
    while math.gcd(stride, total) != 1:
        stride = (stride + 1) % total or 1

    offset = blake2b_u64(f"perm_offset|{schedule_seed}") % total
    return int(stride), int(offset)


def permute_index(i: int, total: int, schedule_seed: int) -> int:
    if i < 0:
        raise ValueError("Index i must be >= 0.")
    if total <= 0:
        raise ValueError("total must be > 0.")
    if total <= 1:
        return 0

    stride, offset = _compute_stride_offset(total=total, schedule_seed=schedule_seed)
    return (i * stride + offset) % total


def allocate_exact_counts(total: int, weights_by_id: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0.")
    if not weights_by_id:
        raise ValueError("weights_by_id must not be empty.")

    ids = sorted(weights_by_id)
    for dataset_id in ids:
        weight = float(weights_by_id[dataset_id])
        if weight <= 0:
            raise ValueError(f"Weight for dataset_id={dataset_id!r} must be > 0.")

    z = sum(float(weights_by_id[i]) for i in ids)
    if z <= 0:
        raise ValueError("Sum of weights must be > 0.")

    raw = {i: total * float(weights_by_id[i]) / z for i in ids}
    base = {i: int(math.floor(raw[i])) for i in ids}
    rem = total - sum(base.values())

    order = sorted(ids, key=lambda i: (-(raw[i] - base[i]), i))
    for dataset_id in order[:rem]:
        base[dataset_id] += 1

    return base


def _allocate_hamilton_on_raw(total: int, raw_by_id: Mapping[str, float]) -> dict[str, int]:
    """
    Hamilton / largest-remainder allocation on already-computed raw targets.

    This does not renormalize weights. It assumes each raw value already reflects the intended
    target count (up to rounding) and performs deterministic integer allocation to an exact total,
    breaking ties lexicographically by dataset_id.
    """
    t = int(total)
    if t < 0:
        raise ValueError("total must be >= 0.")
    if not raw_by_id:
        raise ValueError("raw_by_id must not be empty.")

    ids = sorted(raw_by_id)
    raw: dict[str, float] = {}
    for dataset_id in ids:
        value = float(raw_by_id[dataset_id])
        if value < 0:
            raise ValueError(f"raw target for dataset_id={dataset_id!r} must be >= 0.")
        raw[dataset_id] = value

    base = {dataset_id: int(math.floor(raw[dataset_id])) for dataset_id in ids}
    rem = t - sum(base.values())
    if rem < 0:
        raise ValueError("total is smaller than the sum of floor(raw) targets.")

    order = sorted(ids, key=lambda i: (-(raw[i] - base[i]), i))
    for dataset_id in order[:rem]:
        base[dataset_id] += 1

    return base


def _validate_nonzero_source_blocks(source_blocks: Mapping[str, int]) -> None:
    if not source_blocks:
        raise ValueError("source_blocks must not be empty.")
    for dataset_id, blocks in source_blocks.items():
        if int(blocks) <= 0:
            raise ValueError(
                f"Source dataset_id={dataset_id!r} has zero train packed blocks; "
                "mixture mode requires B_i > 0 for every source."
            )


def resolve_effective_weights(
    *,
    configured_weights_by_id: Mapping[str, float | None],
    source_blocks: Mapping[str, int],
) -> tuple[dict[str, float], str]:
    if set(configured_weights_by_id) != set(source_blocks):
        raise ValueError("configured_weights_by_id and source_blocks must contain identical dataset_id keys.")

    _validate_nonzero_source_blocks(source_blocks)

    ids = sorted(configured_weights_by_id)
    present = [configured_weights_by_id[i] is not None for i in ids]

    if all(present):
        weights = {i: float(configured_weights_by_id[i]) for i in ids}
        for dataset_id, weight in weights.items():
            if weight <= 0:
                raise ValueError(f"weight for dataset_id={dataset_id!r} must be > 0.")
        return weights, "manual"

    if not any(present):
        return {i: float(int(source_blocks[i])) for i in ids}, "from_source_blocks"

    raise ValueError("All sources must either set weight or omit weight.")


def derive_anchor_epoch_targets(
    *,
    source_blocks: Mapping[str, int],
    weights_by_id: Mapping[str, float],
    anchor_id: str,
) -> dict[str, int]:
    if anchor_id not in source_blocks:
        raise ValueError(f"anchor_id={anchor_id!r} is not present in source_blocks.")
    if anchor_id not in weights_by_id:
        raise ValueError(f"anchor_id={anchor_id!r} is not present in weights_by_id.")

    b_anchor = int(source_blocks[anchor_id])
    if b_anchor <= 0:
        raise ValueError("Anchor source must have B_anchor > 0.")

    w_anchor = float(weights_by_id[anchor_id])
    if w_anchor <= 0:
        raise ValueError("Anchor weight must be > 0.")

    non_anchor_ids = [did for did in sorted(weights_by_id) if did != anchor_id]
    if not non_anchor_ids:
        return {anchor_id: b_anchor}

    raw_non_anchor = {did: b_anchor * float(weights_by_id[did]) / w_anchor for did in non_anchor_ids}
    total_non_anchor = int(math.floor(sum(raw_non_anchor.values()) + 0.5))
    targets_non_anchor = _allocate_hamilton_on_raw(total_non_anchor, raw_non_anchor)

    out: dict[str, int] = {anchor_id: b_anchor}
    out.update(targets_non_anchor)
    return out


def resolve_mixture_plan(
    *,
    source_blocks: Mapping[str, int],
    configured_weights_by_id: Mapping[str, float | None],
    budget_mode: str,
    anchor_epochs: int | None,
    anchor_dataset_id: str | None,
    explicit_total_blocks: int | None,
) -> MixturePlan:
    if set(source_blocks) != set(configured_weights_by_id):
        raise ValueError("source_blocks and configured_weights_by_id must contain identical dataset_id keys.")

    source_blocks_by_id = {dataset_id: int(source_blocks[dataset_id]) for dataset_id in sorted(source_blocks)}
    effective_weights, weight_mode = resolve_effective_weights(
        configured_weights_by_id=configured_weights_by_id,
        source_blocks=source_blocks_by_id,
    )

    resolved_anchor: str | None = anchor_dataset_id

    if budget_mode == "explicit_blocks":
        if explicit_total_blocks is None:
            raise ValueError("budget_mode='explicit_blocks' requires explicit_total_blocks.")
        requested_total_blocks = int(explicit_total_blocks)
        if requested_total_blocks <= 0:
            raise ValueError("explicit_total_blocks must be > 0.")
        if resolved_anchor is not None and resolved_anchor not in source_blocks_by_id:
            raise ValueError(f"anchor_dataset_id={resolved_anchor!r} is not present in sources.")
        target_blocks = allocate_exact_counts(requested_total_blocks, effective_weights)

    elif budget_mode == "anchor_epochs":
        if anchor_epochs is None:
            raise ValueError("budget_mode='anchor_epochs' requires anchor_epochs.")
        anchor_epochs_int = int(anchor_epochs)
        if anchor_epochs_int <= 0:
            raise ValueError("anchor_epochs must be >= 1.")

        if resolved_anchor is None:
            resolved_anchor = sorted(
                source_blocks_by_id.keys(),
                key=lambda did: (-source_blocks_by_id[did], did),
            )[0]
        elif resolved_anchor not in source_blocks_by_id:
            raise ValueError(f"anchor_dataset_id={resolved_anchor!r} is not present in sources.")

        per_epoch_targets = derive_anchor_epoch_targets(
            source_blocks=source_blocks_by_id,
            weights_by_id=effective_weights,
            anchor_id=resolved_anchor,
        )
        per_epoch_total = sum(per_epoch_targets.values())
        requested_total_blocks = anchor_epochs_int * per_epoch_total
        target_blocks = {
            dataset_id: int(per_epoch_targets.get(dataset_id, 0)) * anchor_epochs_int
            for dataset_id in sorted(source_blocks_by_id)
        }
    else:
        raise ValueError(f"Unsupported budget_mode={budget_mode!r}.")

    if requested_total_blocks <= 0:
        raise ValueError("Requested total blocks must be > 0.")

    if sum(target_blocks.values()) != requested_total_blocks:
        raise RuntimeError(
            "Internal error: target block allocation does not match requested total blocks "
            f"({sum(target_blocks.values())} != {requested_total_blocks})."
        )

    return MixturePlan(
        budget_mode=budget_mode,
        requested_total_blocks=requested_total_blocks,
        target_blocks_per_dataset=target_blocks,
        source_blocks_by_id=source_blocks_by_id,
        weight_mode=weight_mode,
        effective_weights_by_id=effective_weights,
        anchor_dataset_id=resolved_anchor,
    )


def resolve_effective_total_blocks(
    *,
    requested_total_blocks: int,
    world_size: int,
    batch_size: int,
    sampler_drop_last: bool,
    drop_last_batch: bool,
) -> int:
    requested = int(requested_total_blocks)
    world = int(world_size)
    batch = int(batch_size)
    sampler_drop = bool(sampler_drop_last)
    batch_drop = bool(drop_last_batch)

    if requested <= 0:
        raise ValueError("requested_total_blocks must be > 0.")
    if world <= 0:
        raise ValueError("world_size must be > 0.")
    if batch <= 0:
        raise ValueError("batch_size must be > 0.")

    if world > 1 and requested % world != 0:
        raise ValueError(
            "Mixture requested_total_blocks must be divisible by world_size under the strict-budget contract. "
            f"Got requested_total_blocks={requested}, world_size={world}, sampler_drop_last={sampler_drop}."
        )

    per_rank_samples = requested // world
    per_rank_executed = (per_rank_samples // batch) * batch if batch_drop else per_rank_samples
    effective = per_rank_executed * world

    if effective != requested:
        raise ValueError(
            "Mixture strict-budget contract violated: executed blocks differ from requested blocks. "
            f"requested_total_blocks={requested}, effective_total_blocks={effective}, "
            f"world_size={world}, batch_size={batch}, sampler_drop_last={sampler_drop}, drop_last_batch={batch_drop}. "
            "Set drop_last_batch=false or adjust requested_total_blocks/batch_size/world_size."
        )

    return effective


def resolve_mixture_alignment_policy(
    *,
    budget_mode: str,
    alignment_policy: str | None,
) -> str:
    mode = str(budget_mode)
    if alignment_policy is None:
        return "floor" if mode == "anchor_epochs" else "error"

    policy = str(alignment_policy).strip().lower()
    if policy not in {"error", "floor", "ceil"}:
        raise ValueError(
            "dataset.mixture.alignment_policy must be one of {'error', 'floor', 'ceil'} "
            f"when provided, got {alignment_policy!r}."
        )

    if mode == "explicit_blocks" and policy != "error":
        raise ValueError(
            "dataset.mixture.alignment_policy must be 'error' when budget_mode='explicit_blocks' "
            "to preserve the exact block-budget contract."
        )
    return policy


def resolve_aligned_total_blocks(
    *,
    requested_total_blocks: int,
    budget_mode: str,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    alignment_policy: str | None,
) -> tuple[int, dict[str, int | bool | str]]:
    requested = int(requested_total_blocks)
    world = int(world_size)
    batch = int(batch_size)
    grad_accum = int(gradient_accumulation_steps)
    if requested <= 0:
        raise ValueError("requested_total_blocks must be > 0.")
    if world <= 0:
        raise ValueError("world_size must be > 0.")
    if batch <= 0:
        raise ValueError("batch_size must be > 0.")
    if grad_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0.")

    policy = resolve_mixture_alignment_policy(
        budget_mode=budget_mode,
        alignment_policy=alignment_policy,
    )
    alignment_unit = world * batch * grad_accum
    if alignment_unit <= 0:
        raise ValueError("alignment_unit must be > 0.")

    if requested % alignment_unit == 0:
        return requested, {
            "policy": policy,
            "alignment_unit": int(alignment_unit),
            "requested_total_blocks": int(requested),
            "adjusted_total_blocks": int(requested),
            "alignment_applied": False,
        }

    if policy == "error":
        raise ValueError(
            "Mixture requested_total_blocks must be divisible by world_size * batch_size * "
            "gradient_accumulation_steps under strict alignment. "
            f"Got requested_total_blocks={requested}, world_size={world}, batch_size={batch}, "
            f"gradient_accumulation_steps={grad_accum}, alignment_unit={alignment_unit}, "
            f"budget_mode={budget_mode!r}."
        )

    if policy == "floor":
        adjusted = (requested // alignment_unit) * alignment_unit
    else:
        adjusted = ((requested + alignment_unit - 1) // alignment_unit) * alignment_unit

    if adjusted <= 0:
        raise ValueError(
            "Mixture alignment produced zero executable blocks. "
            f"requested_total_blocks={requested}, alignment_unit={alignment_unit}, policy={policy!r}. "
            "Increase anchor_epochs, reduce world_size/batch_size/gradient_accumulation_steps, or switch policy to 'error'."
        )

    return int(adjusted), {
        "policy": policy,
        "alignment_unit": int(alignment_unit),
        "requested_total_blocks": int(requested),
        "adjusted_total_blocks": int(adjusted),
        "alignment_applied": int(adjusted) != int(requested),
    }


def build_mixture_progress_metrics(
    *,
    target_blocks: Mapping[str, int],
    realized_blocks: Mapping[str, int],
    effective_total_blocks: int,
) -> dict[str, float]:
    total = int(effective_total_blocks)
    ids = sorted(set(target_blocks) | set(realized_blocks))
    metrics: dict[str, float] = {}

    for dataset_id in ids:
        target = float(int(target_blocks.get(dataset_id, 0)))
        realized = float(int(realized_blocks.get(dataset_id, 0)))
        deviation = realized - target

        metrics[f"mixture/target_blocks_{dataset_id}"] = target
        metrics[f"mixture/realized_blocks_{dataset_id}"] = realized
        metrics[f"mixture/deviation_blocks_{dataset_id}"] = deviation
        metrics[f"mixture/target_ratio_{dataset_id}"] = (target / total) if total > 0 else 0.0
        metrics[f"mixture/realized_ratio_{dataset_id}"] = (realized / total) if total > 0 else 0.0
        metrics[f"mixture/resampling_ratio_{dataset_id}"] = (realized / target) if target > 0 else 0.0

    return metrics


class MixturePackedDataset(Dataset):
    def __init__(
        self,
        *,
        datasets_by_id: Mapping[str, Dataset],
        counts_by_id: Mapping[str, int],
        schedule_seed: int,
    ) -> None:
        if not datasets_by_id:
            raise ValueError("datasets_by_id must not be empty.")
        if set(datasets_by_id) != set(counts_by_id):
            raise ValueError("datasets_by_id and counts_by_id must have the same dataset_id keys.")

        self.datasets_by_id = {dataset_id: datasets_by_id[dataset_id] for dataset_id in sorted(datasets_by_id)}
        self.counts_by_id = {dataset_id: int(counts_by_id[dataset_id]) for dataset_id in sorted(counts_by_id)}
        self.schedule_seed = int(schedule_seed)

        self.dataset_id_to_idx = {dataset_id: idx for idx, dataset_id in enumerate(sorted(self.datasets_by_id))}
        self.dataset_idx_to_id = {idx: dataset_id for dataset_id, idx in self.dataset_id_to_idx.items()}

        self._ordered_ids = sorted(self.counts_by_id)
        self._cumulative_upper_bounds: list[int] = []
        running = 0
        for dataset_id in self._ordered_ids:
            count = int(self.counts_by_id[dataset_id])
            if count < 0:
                raise ValueError(f"Count for dataset_id={dataset_id!r} must be >= 0.")
            if len(self.datasets_by_id[dataset_id]) <= 0:
                raise ValueError(
                    f"dataset_id={dataset_id!r} has zero train packed blocks; mixture mode requires B_i > 0."
                )
            running += count
            self._cumulative_upper_bounds.append(running)

        self._total_blocks = running
        if self._total_blocks <= 0:
            raise ValueError("Total mixture blocks must be > 0.")
        self._perm_stride, self._perm_offset = _compute_stride_offset(
            total=self._total_blocks,
            schedule_seed=self.schedule_seed,
        )

    def __len__(self) -> int:
        return self._total_blocks

    def _dataset_id_for_position(self, global_idx: int) -> str:
        p = (int(global_idx) * self._perm_stride + self._perm_offset) % self._total_blocks
        dataset_pos = bisect_right(self._cumulative_upper_bounds, p)
        return self._ordered_ids[dataset_pos]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            raise IndexError("MixturePackedDataset does not support negative indices.")
        if idx >= self._total_blocks:
            raise IndexError(f"Index {idx} out of range for {self._total_blocks} mixture blocks.")

        dataset_id = self._dataset_id_for_position(idx)
        source_dataset = self.datasets_by_id[dataset_id]
        source_len = len(source_dataset)
        if source_len <= 0:
            raise RuntimeError(f"dataset_id={dataset_id!r} has no blocks.")

        local_idx = blake2b_u64(f"{self.schedule_seed}|{idx}|{dataset_id}") % int(source_len)
        sample = source_dataset[int(local_idx)]
        if not isinstance(sample, dict):
            raise TypeError("MixturePackedDataset expects source dataset samples to be dictionaries.")

        out = dict(sample)
        out["dataset_idx"] = torch.tensor(self.dataset_id_to_idx[dataset_id], dtype=torch.int64)
        return out
