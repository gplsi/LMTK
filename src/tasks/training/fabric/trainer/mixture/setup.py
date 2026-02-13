"""Mixture setup/config method implementations used by MixtureSetupMixin."""

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import lightning as L
from box import Box
from datasets import DatasetDict
from torch.utils.data import DataLoader

from src.tasks.training.data.mixture import (
    MixturePackedDataset,
    MixturePlan,
    allocate_exact_counts,
    blake2b_u64,
    build_expected_replay_summary,
    resolve_aligned_total_blocks,
    resolve_effective_total_blocks,
    resolve_mixture_plan,
)
from src.tasks.training.data.packing import PackedSequenceDataset, build_packing_dataloader
from src.tasks.training.data.packing_index import (
    INDEX_VERSION,
    PackingIndex,
    PackingIndexBuildError,
)

from .constants import MIXTURE_SAMPLING_ALGORITHM


def _get_mixture_config(self) -> Optional[Box]:
    dataset_cfg = getattr(self.config, "dataset", None)
    if dataset_cfg is None:
        return None
    mixture = (
        dataset_cfg.get("mixture", None)
        if hasattr(dataset_cfg, "get")
        else getattr(dataset_cfg, "mixture", None)
    )
    if mixture is None:
        return None
    if isinstance(mixture, Box):
        return mixture
    if isinstance(mixture, dict):
        return Box(mixture, box_dots=True)
    raise TypeError(f"Unsupported dataset.mixture type: {type(mixture)}")


def _get_sources_config(self) -> list[Box]:
    dataset_cfg = getattr(self.config, "dataset", None)
    if dataset_cfg is None:
        return []
    sources = (
        dataset_cfg.get("sources", None)
        if hasattr(dataset_cfg, "get")
        else getattr(dataset_cfg, "sources", None)
    )
    if not sources:
        return []

    out: list[Box] = []
    for source in sources:
        if isinstance(source, Box):
            out.append(source)
        elif isinstance(source, dict):
            out.append(Box(source, box_dots=True))
        else:
            raise TypeError(f"Unsupported dataset.sources item type: {type(source)}")
    return out


def _is_mixture_enabled(self) -> bool:
    mixture = self._get_mixture_config()
    sources = self._get_sources_config()
    return bool(mixture and mixture.get("enabled", False) and sources)


def _derive_validation_split_seed(self, dataset_id: str) -> int:
    base_seed_raw = self.config.get("seed", 0)
    base_seed = int(base_seed_raw) if base_seed_raw is not None else 0
    return int(blake2b_u64(f"valsplit|{base_seed}|{dataset_id}") % 2147483647)


def _collect_source_config_map(self) -> dict[str, Box]:
    source_configs = self._get_sources_config()
    if not source_configs:
        raise ValueError("Mixture mode requires dataset.sources.")

    out: dict[str, Box] = {}
    for source in source_configs:
        dataset_id = source.get("dataset_id", None)
        if not dataset_id:
            raise ValueError("Each dataset.sources entry must define dataset_id.")
        dataset_id = str(dataset_id)
        if dataset_id in out:
            raise ValueError(f"Duplicate dataset_id in dataset.sources: {dataset_id!r}")
        out[dataset_id] = source
    return out


def _validate_mixture_source_compatibility(self, source_config_map: Mapping[str, Box]) -> None:
    tokenizer_names = {
        dataset_id: source_cfg.get("tokenizer_name", None)
        for dataset_id, source_cfg in source_config_map.items()
        if source_cfg.get("tokenizer_name", None) is not None
    }
    if tokenizer_names:
        unique_tokenizers = sorted({str(v) for v in tokenizer_names.values()})
        if len(unique_tokenizers) > 1:
            raise ValueError(
                "Incompatible source tokenizers in dataset.sources: "
                f"{tokenizer_names}. Expected all tokenizer_name values to match."
            )

    eos_ids = {
        dataset_id: int(source_cfg.get("eos_token_id"))
        for dataset_id, source_cfg in source_config_map.items()
        if source_cfg.get("eos_token_id", None) is not None
    }
    if eos_ids:
        unique_eos_ids = sorted(set(eos_ids.values()))
        if len(unique_eos_ids) > 1:
            raise ValueError(
                "Incompatible source eos_token_id values in dataset.sources: "
                f"{eos_ids}. Expected all eos_token_id values to match."
            )

    packing = self._get_packing_config()
    if packing is None:
        return

    packing_tokenizer = packing.get("tokenizer_name", None)
    if packing_tokenizer is not None and tokenizer_names:
        source_tokenizer = next(iter(tokenizer_names.values()))
        if str(source_tokenizer) != str(packing_tokenizer):
            raise ValueError(
                "Incompatible tokenizer configuration: dataset.packing.tokenizer_name does not match "
                f"dataset.sources tokenizer_name values ({packing_tokenizer!r} vs {source_tokenizer!r})."
            )

    packing_eos = packing.get("eos_token_id", None)
    if packing_eos is not None and eos_ids:
        first_eos = next(iter(eos_ids.values()))
        if int(first_eos) != int(packing_eos):
            raise ValueError(
                "Incompatible eos configuration: dataset.packing.eos_token_id does not match "
                f"dataset.sources eos_token_id values ({packing_eos!r} vs {first_eos!r})."
            )


def _build_mixture_packing_dataloaders(self, fabric: L.Fabric) -> dict[str, Any]:
    packing = self._get_packing_config()
    mixture = self._get_mixture_config()
    if not packing or not packing.get("enabled", False):
        raise RuntimeError("_build_mixture_packing_dataloaders called but packing is not enabled.")
    if not mixture or not mixture.get("enabled", False):
        raise RuntimeError("_build_mixture_packing_dataloaders called but mixture is not enabled.")
    lock_timeout_s = self._packing_index_lock_timeout_s(packing)
    stale_lock_age_s = self._packing_index_stale_lock_age_s(packing)
    lock_lease_heartbeat_s = self._packing_index_lock_lease_heartbeat_s(packing)
    self._validate_packing_lock_timings(
        lock_timeout_s=lock_timeout_s,
        stale_lock_age_s=stale_lock_age_s,
        lock_lease_heartbeat_s=lock_lease_heartbeat_s,
    )

    sequence_length = packing.get("sequence_length", None)
    if sequence_length is None:
        raise ValueError("Mixture mode requires packing.sequence_length.")
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("packing.sequence_length must be a positive integer.")

    insert_eos = packing.get("insert_eos", None)
    if insert_eos is None:
        insert_eos = True
    insert_eos = bool(insert_eos)

    eos_token_id: Optional[int] = None
    if insert_eos:
        eos_token_id = self._resolve_eos_token_id(packing)

    seed = self.config.get("seed", None)
    seed_value = int(seed) if seed is not None else 0
    schedule_seed_raw = mixture.get("schedule_seed", None)
    schedule_seed = int(schedule_seed_raw) if schedule_seed_raw is not None else seed_value

    source_config_map = self._mixture_source_configs
    if not source_config_map:
        raise ValueError("Mixture source configuration is missing.")

    slurm_nnodes = os.getenv("SLURM_NNODES")
    nnodes = 1
    if slurm_nnodes is not None:
        try:
            nnodes = int(slurm_nnodes)
        except ValueError:
            nnodes = 1

    train_packed_by_id: dict[str, PackedSequenceDataset] = {}
    valid_packed_by_id: dict[str, PackedSequenceDataset] = {}
    source_blocks: dict[str, int] = {}
    configured_weights: dict[str, float | None] = {}

    for dataset_id in sorted(source_config_map):
        source_cfg = source_config_map[dataset_id]
        source_dataset = self.datasets.get(dataset_id)
        if source_dataset is None:
            raise ValueError(f"Source dataset {dataset_id!r} is missing from loaded datasets.")
        if not isinstance(source_dataset, DatasetDict):
            source_dataset = DatasetDict(source_dataset)
        if "train" not in source_dataset or "valid" not in source_dataset:
            raise ValueError(f"Source dataset {dataset_id!r} must contain train and valid splits.")

        dataset_path_value = source_cfg.get("nameOrPath", None)
        if not dataset_path_value:
            raise ValueError(f"Source dataset {dataset_id!r} is missing nameOrPath.")
        dataset_path = Path(str(dataset_path_value))

        index_cache_dir_value = source_cfg.get("index_cache_dir", None)
        index_cache_dir = (
            Path(str(index_cache_dir_value))
            if index_cache_dir_value
            else (dataset_path / ".packing_index")
        )
        if self._clean_index_cache_on_start(packing):
            self._maybe_clean_index_cache_version_dir(
                fabric,
                index_cache_dir,
                packing=packing,
                trigger="start",
            )

        if nnodes > 1:
            for path in (dataset_path, index_cache_dir):
                if not path.is_absolute():
                    raise ValueError(
                        "Multi-node SLURM mixture packing requires absolute dataset/index paths on a shared filesystem. "
                        f"Got non-absolute path for source {dataset_id!r}: {path}"
                    )
                if str(path).startswith("/tmp") or str(path).startswith("/dev/shm"):
                    raise ValueError(
                        "Multi-node SLURM mixture packing requires shared filesystem paths. "
                        f"Got node-local path for source {dataset_id!r}: {path}"
                    )

        indices: dict[str, PackingIndex] = {}
        build_failed_path = index_cache_dir / f"v{INDEX_VERSION}" / "BUILD_FAILED.json"
        if fabric.global_rank == 0:
            active_split_name: str | None = None
            try:
                build_failed_path.parent.mkdir(parents=True, exist_ok=True)
                build_failed_path.unlink(missing_ok=True)
                for split_name in ("train", "valid"):
                    active_split_name = split_name
                    split_t0 = time.perf_counter()
                    self.cli_logger.info(
                        "Mixture index start source=%s split=%s sequence_length=%s insert_eos=%s index_cache_dir=%s",
                        dataset_id,
                        split_name,
                        sequence_length,
                        insert_eos,
                        index_cache_dir,
                    )
                    indices[split_name] = PackingIndex.load_or_build(
                        hf_split=source_dataset[split_name],
                        split=split_name,
                        sequence_length=sequence_length,
                        insert_eos=insert_eos,
                        eos_token_id=eos_token_id,
                        cache_dir=index_cache_dir,
                        lock_timeout_s=lock_timeout_s,
                        stale_lock_age_s=stale_lock_age_s,
                        lock_lease_heartbeat_s=lock_lease_heartbeat_s,
                    )
                    self.cli_logger.info(
                        "Mixture index ready source=%s split=%s blocks=%s elapsed_s=%.2f cache_hit=%s index_cache_dir=%s",
                        dataset_id,
                        split_name,
                        int(indices[split_name].num_blocks),
                        time.perf_counter() - split_t0,
                        bool(getattr(indices[split_name], "cache_hit", False)),
                        index_cache_dir,
                    )
            except Exception as exc:
                payload = {
                    "dataset_id": dataset_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at_unix": time.time(),
                    "pid": os.getpid(),
                    "host": os.getenv("SLURMD_NODENAME") or os.uname().nodename,
                }
                if isinstance(exc, PackingIndexBuildError):
                    build_context = dict(getattr(exc, "failure_context", {}) or {})
                    payload["build_context"] = build_context
                    payload["oom_suspected"] = bool(build_context.get("oom_suspected", False))
                    self.cli_logger.error(
                        "Mixture index subprocess build failed source=%s split=%s oom_suspected=%s last_progress=%s",
                        dataset_id,
                        active_split_name,
                        payload["oom_suspected"],
                        build_context.get("last_progress", None),
                    )
                if isinstance(exc, TimeoutError):
                    payload["lock_context"] = self._collect_lock_failure_context(
                        index_cache_dir=index_cache_dir,
                        split_name=active_split_name,
                        lock_timeout_s=lock_timeout_s,
                        stale_lock_age_s=stale_lock_age_s,
                        lock_lease_heartbeat_s=lock_lease_heartbeat_s,
                    )
                try:
                    build_failed_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                try:
                    build_failed_path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass

        fabric.barrier()
        if build_failed_path.exists():
            details = build_failed_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(
                "Packing index build failed on rank 0 for mixture source "
                f"{dataset_id!r}. See {build_failed_path} for details: {details}"
            )

        if fabric.global_rank != 0:
            for split_name in ("train", "valid"):
                indices[split_name] = PackingIndex.load_or_build(
                    hf_split=source_dataset[split_name],
                    split=split_name,
                    sequence_length=sequence_length,
                    insert_eos=insert_eos,
                    eos_token_id=eos_token_id,
                    cache_dir=index_cache_dir,
                    allow_build=False,
                    lock_timeout_s=lock_timeout_s,
                    stale_lock_age_s=None,
                    lock_lease_heartbeat_s=lock_lease_heartbeat_s,
                )

        train_packed = PackedSequenceDataset(
            hf_dataset=source_dataset["train"],
            index=indices["train"],
            eos_token_id=eos_token_id,
        )
        valid_packed = PackedSequenceDataset(
            hf_dataset=source_dataset["valid"],
            index=indices["valid"],
            eos_token_id=eos_token_id,
        )
        train_packed_by_id[dataset_id] = train_packed
        valid_packed_by_id[dataset_id] = valid_packed
        source_blocks[dataset_id] = len(train_packed)
        configured_weights[dataset_id] = source_cfg.get("weight", None)

        if fabric.global_rank == 0:
            self.cli_logger.info(
                "Mixture source=%s train_blocks=%s valid_blocks=%s sequence_length=%s index_cache_dir=%s",
                dataset_id,
                len(train_packed),
                len(valid_packed),
                sequence_length,
                index_cache_dir,
            )

    budget_mode = str(mixture.get("budget_mode", "anchor_epochs"))
    anchor_epochs = mixture.get("anchor_epochs", None)
    anchor_dataset_id = mixture.get("anchor_dataset_id", None)
    explicit_total_blocks = mixture.get("total_blocks", None)

    plan = resolve_mixture_plan(
        source_blocks=source_blocks,
        configured_weights_by_id=configured_weights,
        budget_mode=budget_mode,
        anchor_epochs=anchor_epochs,
        anchor_dataset_id=anchor_dataset_id,
        explicit_total_blocks=explicit_total_blocks,
    )
    requested_total_blocks_raw = int(plan.requested_total_blocks)
    grad_accum_steps = self._resolved_gradient_accumulation_steps()
    aligned_total_blocks, alignment_metadata = resolve_aligned_total_blocks(
        requested_total_blocks=requested_total_blocks_raw,
        budget_mode=budget_mode,
        world_size=int(fabric.world_size),
        batch_size=int(self.config.batch_size),
        gradient_accumulation_steps=grad_accum_steps,
        alignment_policy=mixture.get("alignment_policy", None),
    )
    if int(aligned_total_blocks) != int(requested_total_blocks_raw):
        aligned_targets = allocate_exact_counts(
            int(aligned_total_blocks),
            plan.effective_weights_by_id,
        )
        plan = MixturePlan(
            budget_mode=plan.budget_mode,
            requested_total_blocks=int(aligned_total_blocks),
            target_blocks_per_dataset=aligned_targets,
            source_blocks_by_id=plan.source_blocks_by_id,
            weight_mode=plan.weight_mode,
            effective_weights_by_id=plan.effective_weights_by_id,
            anchor_dataset_id=plan.anchor_dataset_id,
        )
        if fabric.global_rank == 0:
            self.cli_logger.info(
                "Mixture block alignment applied budget_mode=%s alignment_policy=%s "
                "requested_total_blocks=%s adjusted_total_blocks=%s alignment_unit=%s",
                budget_mode,
                alignment_metadata.get("policy", None),
                requested_total_blocks_raw,
                aligned_total_blocks,
                alignment_metadata.get("alignment_unit", None),
            )
    requested_total_blocks = int(plan.requested_total_blocks)

    mixture_train_dataset = MixturePackedDataset(
        datasets_by_id=train_packed_by_id,
        counts_by_id=plan.target_blocks_per_dataset,
        schedule_seed=schedule_seed,
    )
    if len(mixture_train_dataset) != requested_total_blocks:
        raise RuntimeError(
            "Mixture dataset length mismatch with requested budget: "
            f"{len(mixture_train_dataset)} != {requested_total_blocks}"
        )

    shuffle = packing.get("shuffle", None)
    if shuffle is None:
        shuffle = True
    shuffle = bool(shuffle)

    sampler_drop_last = packing.get("sampler_drop_last", None)
    if sampler_drop_last is None:
        sampler_drop_last = fabric.world_size > 1
    sampler_drop_last = bool(sampler_drop_last)

    drop_last_batch = packing.get("drop_last_batch", None)
    if drop_last_batch is None:
        drop_last_batch = fabric.world_size > 1
    drop_last_batch = bool(drop_last_batch)

    effective_total_blocks = resolve_effective_total_blocks(
        requested_total_blocks=requested_total_blocks,
        world_size=int(fabric.world_size),
        batch_size=int(self.config.batch_size),
        sampler_drop_last=sampler_drop_last,
        drop_last_batch=drop_last_batch,
    )
    expected_replay_summary = build_expected_replay_summary(
        target_blocks=plan.target_blocks_per_dataset,
        source_blocks_available=source_blocks,
    )
    if fabric.global_rank == 0:
        expected_totals = expected_replay_summary.get("totals", {})
        expected_total_replayed = int(expected_totals.get("expected_replayed_draws", 0))
        total_log_fn = (
            self.cli_logger.warning if expected_total_replayed > 0 else self.cli_logger.info
        )
        total_log_fn(
            "Mixture expected replay (pre-run): target_blocks=%s expected_replayed_draws=%s expected_replay_fraction=%.6f sampling_algorithm=%s",
            int(expected_totals.get("target_blocks", 0)),
            expected_total_replayed,
            float(expected_totals.get("expected_replay_fraction", 0.0)),
            MIXTURE_SAMPLING_ALGORITHM,
        )
        per_source_expected = expected_replay_summary.get("per_source", {})
        for dataset_id in sorted(per_source_expected):
            source_summary = per_source_expected[dataset_id]
            expected_replayed = int(source_summary.get("expected_replayed_draws", 0))
            log_fn = self.cli_logger.warning if expected_replayed > 0 else self.cli_logger.info
            log_fn(
                "Mixture expected replay source=%s target_blocks=%s source_blocks_available=%s expected_unique=%s expected_replayed=%s expected_replay_fraction=%.6f",
                dataset_id,
                int(source_summary.get("target_blocks", 0)),
                int(source_summary.get("source_blocks_available", 0)),
                int(source_summary.get("expected_unique_blocks", 0)),
                expected_replayed,
                float(source_summary.get("expected_replay_fraction", 0.0)),
            )

    train_loader = build_packing_dataloader(
        dataset=mixture_train_dataset,
        split="train",
        batch_size=int(self.config.batch_size),
        num_workers=int(self.config.num_workers),
        shuffle=shuffle,
        sampler_drop_last=sampler_drop_last,
        drop_last_batch=drop_last_batch,
        seed=seed_value,
        rank=int(fabric.global_rank),
        world_size=int(fabric.world_size),
    )

    valid_by_source: dict[str, DataLoader] = {}
    for dataset_id in sorted(valid_packed_by_id):
        valid_loader = build_packing_dataloader(
            dataset=valid_packed_by_id[dataset_id],
            split="valid",
            batch_size=int(self.config.batch_size),
            num_workers=int(self.config.num_workers),
            shuffle=False,
            sampler_drop_last=False,
            drop_last_batch=False,
            seed=seed_value,
            rank=int(fabric.global_rank),
            world_size=int(fabric.world_size),
        )
        if len(valid_loader) <= 0:
            raise ValueError(
                f"Validation dataloader for source {dataset_id!r} has zero batches after setup."
            )
        valid_by_source[dataset_id] = valid_loader

    self._mixture_plan = plan
    self._mixture_requested_total_blocks = requested_total_blocks_raw
    self._mixture_effective_total_blocks = effective_total_blocks
    self._mixture_alignment_policy = str(alignment_metadata.get("policy", None))
    self._mixture_alignment_unit = int(alignment_metadata.get("alignment_unit", 0))
    self._mixture_alignment_applied = bool(alignment_metadata.get("alignment_applied", False))
    self._mixture_schedule_seed = schedule_seed
    self._mixture_dataset_id_to_idx = dict(mixture_train_dataset.dataset_id_to_idx)
    self._mixture_dataset_idx_to_id = dict(mixture_train_dataset.dataset_idx_to_id)
    self._mixture_realized_blocks_local = {dataset_id: 0 for dataset_id in sorted(source_blocks)}
    self._mixture_realized_blocks_local_tensor = None
    self._mixture_replayed_draws_local = {dataset_id: 0 for dataset_id in sorted(source_blocks)}
    self._mixture_replayed_draws_local_tensor = None
    self._mixture_configured_weights_by_id = configured_weights
    self._mixture_source_blocks_by_id = source_blocks
    self._mixture_expected_replay_summary = expected_replay_summary
    self._mixture_global_blocks_seen_estimate = 0
    self._mixture_anchor_epoch_index = 0
    if budget_mode == "anchor_epochs":
        self._mixture_anchor_epochs = max(1, int(anchor_epochs) if anchor_epochs is not None else 1)
    else:
        self._mixture_anchor_epochs = 1
    self._mixture_anchor_boundaries = []
    if self._mixture_anchor_epochs > 1:
        last_boundary = -1
        for epoch_idx in range(1, self._mixture_anchor_epochs):
            boundary = int((int(effective_total_blocks) * epoch_idx) // self._mixture_anchor_epochs)
            if 0 < boundary < int(effective_total_blocks) and boundary > last_boundary:
                self._mixture_anchor_boundaries.append(boundary)
                last_boundary = boundary
    self._mixture_anchor_window_start_realized_local = {
        dataset_id: 0 for dataset_id in sorted(source_blocks)
    }
    self._mixture_anchor_window_start_replayed_local = {
        dataset_id: 0 for dataset_id in sorted(source_blocks)
    }
    self._mixture_anchor_window_start_realized_local_tensor = None
    self._mixture_anchor_window_start_replayed_local_tensor = None

    report_path_raw = mixture.get("report_path", None)
    if report_path_raw:
        self._mixture_report_path = Path(str(report_path_raw))
    else:
        self._mixture_report_path = Path(self.config.output_dir) / "mixture_report.json"

    return {
        "train": train_loader,
        "valid_by_source": valid_by_source,
    }
