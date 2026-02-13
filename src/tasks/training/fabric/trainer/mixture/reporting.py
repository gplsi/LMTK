"""Mixture report generation implementations for MixtureRuntimeMixin."""

import json
from pathlib import Path

import lightning as L

from src.tasks.training.data.mixture import build_mixture_progress_metrics, build_replay_metrics

from .constants import MIXTURE_SAMPLING_ALGORITHM


def _write_mixture_report(self, fabric: L.Fabric) -> None:
    if not self._mixture_enabled or self._mixture_plan is None:
        return

    resolved_blocks = self._reduce_mixture_realized_blocks(fabric)
    resolved_replayed_draws = self._reduce_mixture_replayed_draws(fabric)
    window_start_realized = self._reduce_mixture_anchor_window_start_realized(fabric)
    window_start_replayed = self._reduce_mixture_anchor_window_start_replayed(fabric)
    if fabric.global_rank != 0:
        return

    report_path = self._mixture_report_path or (
        Path(self.config.output_dir) / "mixture_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    target_blocks = {
        dataset_id: int(v) for dataset_id, v in self._mixture_plan.target_blocks_per_dataset.items()
    }
    requested_total_blocks = int(self._mixture_requested_total_blocks or 0)
    effective_total_blocks = int(self._mixture_effective_total_blocks or 0)

    packing = self._get_packing_config()
    if not packing:
        raise RuntimeError("Mixture report generation requires packing configuration.")
    sequence_length = int(packing.sequence_length)

    realized_tokens = {
        dataset_id: int(blocks) * sequence_length for dataset_id, blocks in resolved_blocks.items()
    }
    observed_unique_blocks = {
        dataset_id: max(
            0,
            int(resolved_blocks.get(dataset_id, 0))
            - int(resolved_replayed_draws.get(dataset_id, 0)),
        )
        for dataset_id in sorted(target_blocks)
    }
    observed_replay_fraction = {
        dataset_id: (
            float(resolved_replayed_draws.get(dataset_id, 0))
            / float(resolved_blocks.get(dataset_id, 0))
            if int(resolved_blocks.get(dataset_id, 0)) > 0
            else 0.0
        )
        for dataset_id in sorted(target_blocks)
    }
    window_realized = {
        dataset_id: max(
            0,
            int(resolved_blocks.get(dataset_id, 0)) - int(window_start_realized.get(dataset_id, 0)),
        )
        for dataset_id in sorted(target_blocks)
    }
    window_replayed = {
        dataset_id: max(
            0,
            int(resolved_replayed_draws.get(dataset_id, 0))
            - int(window_start_replayed.get(dataset_id, 0)),
        )
        for dataset_id in sorted(target_blocks)
    }
    window_unique = {
        dataset_id: max(
            0, int(window_realized.get(dataset_id, 0)) - int(window_replayed.get(dataset_id, 0))
        )
        for dataset_id in sorted(target_blocks)
    }
    window_replay_fraction = {
        dataset_id: (
            float(window_replayed.get(dataset_id, 0)) / float(window_realized.get(dataset_id, 0))
            if int(window_realized.get(dataset_id, 0)) > 0
            else 0.0
        )
        for dataset_id in sorted(target_blocks)
    }
    realized_ratios = {
        dataset_id: (
            float(blocks) / float(effective_total_blocks) if effective_total_blocks > 0 else 0.0
        )
        for dataset_id, blocks in resolved_blocks.items()
    }
    target_ratios = {
        dataset_id: (
            float(blocks) / float(effective_total_blocks) if effective_total_blocks > 0 else 0.0
        )
        for dataset_id, blocks in target_blocks.items()
    }
    deviation_from_target_blocks = {
        dataset_id: int(resolved_blocks.get(dataset_id, 0)) - int(target_blocks.get(dataset_id, 0))
        for dataset_id in sorted(target_blocks)
    }
    resampling_ratio_to_target = {
        dataset_id: (
            float(resolved_blocks.get(dataset_id, 0)) / float(target_blocks.get(dataset_id, 0))
            if int(target_blocks.get(dataset_id, 0)) > 0
            else 0.0
        )
        for dataset_id in sorted(target_blocks)
    }

    grad_accum = self._resolved_gradient_accumulation_steps()

    source_entries = []
    for dataset_id in sorted(self._mixture_plan.effective_weights_by_id):
        source_cfg = self._mixture_source_configs.get(dataset_id, {})
        train_split = (
            self.datasets.get(dataset_id, {}).get("train")
            if isinstance(self.datasets, dict)
            else None
        )
        dataset_fingerprint = (
            getattr(train_split, "_fingerprint", None) if train_split is not None else None
        )
        source_entries.append(
            {
                "dataset_id": dataset_id,
                "nameOrPath": (
                    source_cfg.get("nameOrPath", None) if hasattr(source_cfg, "get") else None
                ),
                "configured_weight": self._mixture_configured_weights_by_id.get(dataset_id),
                "effective_weight": float(self._mixture_plan.effective_weights_by_id[dataset_id]),
                "dataset_fingerprint": dataset_fingerprint,
            }
        )

    report = {
        "sequence_length": sequence_length,
        "weight_mode": self._mixture_plan.weight_mode,
        "budget_mode": self._mixture_plan.budget_mode,
        "anchor_dataset_id": self._mixture_plan.anchor_dataset_id,
        "source_blocks_available": {
            dataset_id: int(v) for dataset_id, v in self._mixture_plan.source_blocks_by_id.items()
        },
        "requested_total_blocks": requested_total_blocks,
        "effective_total_blocks": effective_total_blocks,
        "alignment_policy": self._mixture_alignment_policy,
        "alignment_unit": (
            int(self._mixture_alignment_unit) if self._mixture_alignment_unit is not None else None
        ),
        "alignment_applied": bool(self._mixture_alignment_applied),
        "world_size": int(fabric.world_size),
        "batch_size": int(self.config.batch_size),
        "gradient_accumulation_steps": grad_accum,
        "sampling_algorithm": MIXTURE_SAMPLING_ALGORITHM,
        "dataset_index_map": {
            str(idx): dataset_id
            for idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items())
        },
        "sources": source_entries,
        "target_blocks_per_dataset": target_blocks,
        "realized_blocks_per_dataset": resolved_blocks,
        "realized_tokens_per_dataset": realized_tokens,
        "realized_ratios": realized_ratios,
        "deviation_from_target_blocks": deviation_from_target_blocks,
        "resampling_ratio_to_target": resampling_ratio_to_target,
        "expected_replay": self._mixture_expected_replay_summary,
        "observed_replay_cumulative": {
            "unique_blocks_per_dataset": observed_unique_blocks,
            "replayed_draws_per_dataset": {
                dataset_id: int(v) for dataset_id, v in sorted(resolved_replayed_draws.items())
            },
            "replay_fraction_per_dataset": observed_replay_fraction,
            "totals": {
                "unique_blocks": int(sum(observed_unique_blocks.values())),
                "replayed_draws": int(sum(int(v) for v in resolved_replayed_draws.values())),
                "replay_fraction": (
                    float(sum(int(v) for v in resolved_replayed_draws.values()))
                    / float(sum(int(v) for v in resolved_blocks.values()))
                    if int(sum(int(v) for v in resolved_blocks.values())) > 0
                    else 0.0
                ),
            },
        },
        "observed_replay_anchor_window": {
            "anchor_epoch_index": int(self._mixture_anchor_epoch_index),
            "unique_blocks_per_dataset": window_unique,
            "replayed_draws_per_dataset": window_replayed,
            "replay_fraction_per_dataset": window_replay_fraction,
            "totals": {
                "unique_blocks": int(sum(window_unique.values())),
                "replayed_draws": int(sum(int(v) for v in window_replayed.values())),
                "replay_fraction": (
                    float(sum(int(v) for v in window_replayed.values()))
                    / float(sum(int(v) for v in window_realized.values()))
                    if int(sum(int(v) for v in window_realized.values())) > 0
                    else 0.0
                ),
            },
        },
        "validation_sources": self._mixture_source_split_metadata,
        "val_loss_per_dataset": self._mixture_last_val_losses,
        "val_loss_weighted": self._mixture_last_val_weighted,
        "seed": int(self.config.seed) if self.config.get("seed", None) is not None else 0,
        "schedule_seed": int(self._mixture_schedule_seed),
        "hash_algorithm": "blake2b_u64_v1",
        "allocation_algorithm": "hamilton_lr_lexicographic_v1",
        "split_seed_algorithm": "blake2b_u64_mod_2147483647_v1",
        "run_metadata": self.state.get("run_metadata", {}),
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_metrics: dict[str, float] = {
        "mixture/requested_total_blocks": float(requested_total_blocks),
        "mixture/effective_total_blocks": float(effective_total_blocks),
        "mixture/alignment_applied": 1.0 if self._mixture_alignment_applied else 0.0,
    }
    summary_metrics.update(
        build_mixture_progress_metrics(
            target_blocks=target_blocks,
            realized_blocks=resolved_blocks,
            effective_total_blocks=effective_total_blocks,
        )
    )
    summary_metrics.update(
        build_replay_metrics(
            realized_blocks=resolved_blocks,
            replayed_draws=resolved_replayed_draws,
            namespace="mixture/replay",
        )
    )
    summary_metrics.update(
        build_replay_metrics(
            realized_blocks=window_realized,
            replayed_draws=window_replayed,
            namespace="mixture/replay_window",
        )
    )
    expected_totals = self._mixture_expected_replay_summary.get("totals", {})
    summary_metrics["mixture/replay_expected/replayed_draws_total"] = float(
        expected_totals.get("expected_replayed_draws", 0)
    )
    summary_metrics["mixture/replay_expected/fraction_total"] = float(
        expected_totals.get("expected_replay_fraction", 0.0)
    )
    summary_metrics["mixture/replay_window/anchor_epoch_index"] = float(
        self._mixture_anchor_epoch_index
    )
    if self._mixture_last_val_weighted is not None:
        summary_metrics["mixture/val_loss_weighted"] = float(self._mixture_last_val_weighted)
    fabric.log_dict(summary_metrics, int(self.state.get("step_count", 0)))
    self.cli_logger.info("Wrote mixture report to %s", report_path)
