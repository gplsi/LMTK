"""Mixture runtime method implementations used by MixtureRuntimeMixin."""

import math
import time
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
from tqdm import tqdm

from src.tasks.training.data.mixture import build_mixture_progress_metrics, build_replay_metrics


def _ensure_mixture_realized_counter_tensor(self, device: torch.device) -> torch.Tensor:
    size = len(self._mixture_dataset_idx_to_id)
    if size <= 0:
        raise RuntimeError(
            "Mixture source mapping is not initialized; cannot build realized-block counters."
        )

    local_tensor = self._mixture_realized_blocks_local_tensor
    if local_tensor is None or local_tensor.numel() != size or local_tensor.device != device:
        rebuilt = torch.zeros(size, dtype=torch.long, device=device)
        for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
            idx = int(dataset_idx)
            if 0 <= idx < size:
                rebuilt[idx] = int(self._mixture_realized_blocks_local.get(dataset_id, 0))
        self._mixture_realized_blocks_local_tensor = rebuilt

    return self._mixture_realized_blocks_local_tensor


def _ensure_mixture_replayed_counter_tensor(self, device: torch.device) -> torch.Tensor:
    size = len(self._mixture_dataset_idx_to_id)
    if size <= 0:
        raise RuntimeError(
            "Mixture source mapping is not initialized; cannot build replay-draw counters."
        )

    local_tensor = self._mixture_replayed_draws_local_tensor
    if local_tensor is None or local_tensor.numel() != size or local_tensor.device != device:
        rebuilt = torch.zeros(size, dtype=torch.long, device=device)
        for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
            idx = int(dataset_idx)
            if 0 <= idx < size:
                rebuilt[idx] = int(self._mixture_replayed_draws_local.get(dataset_id, 0))
        self._mixture_replayed_draws_local_tensor = rebuilt

    return self._mixture_replayed_draws_local_tensor


def _sync_mixture_realized_blocks_local_from_tensor(self) -> None:
    if not self._mixture_enabled:
        return
    local_tensor = self._mixture_realized_blocks_local_tensor
    if local_tensor is None:
        return

    values = local_tensor.detach().to(device="cpu", dtype=torch.long).tolist()
    synced: dict[str, int] = {}
    for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
        idx = int(dataset_idx)
        count = int(values[idx]) if 0 <= idx < len(values) else 0
        synced[dataset_id] = max(0, count)
    if synced:
        self._mixture_realized_blocks_local = synced


def _sync_mixture_replayed_draws_local_from_tensor(self) -> None:
    if not self._mixture_enabled:
        return
    local_tensor = self._mixture_replayed_draws_local_tensor
    if local_tensor is None:
        return

    values = local_tensor.detach().to(device="cpu", dtype=torch.long).tolist()
    synced: dict[str, int] = {}
    for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
        idx = int(dataset_idx)
        count = int(values[idx]) if 0 <= idx < len(values) else 0
        synced[dataset_id] = max(0, count)
    if synced:
        self._mixture_replayed_draws_local = synced


def _sync_mixture_anchor_window_start_local_from_tensors(self) -> None:
    if not self._mixture_enabled:
        return

    realized_tensor = self._mixture_anchor_window_start_realized_local_tensor
    if realized_tensor is not None:
        values = realized_tensor.detach().to(device="cpu", dtype=torch.long).tolist()
        synced_realized: dict[str, int] = {}
        for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
            idx = int(dataset_idx)
            count = int(values[idx]) if 0 <= idx < len(values) else 0
            synced_realized[dataset_id] = max(0, count)
        if synced_realized:
            self._mixture_anchor_window_start_realized_local = synced_realized

    replayed_tensor = self._mixture_anchor_window_start_replayed_local_tensor
    if replayed_tensor is not None:
        values = replayed_tensor.detach().to(device="cpu", dtype=torch.long).tolist()
        synced_replayed: dict[str, int] = {}
        for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
            idx = int(dataset_idx)
            count = int(values[idx]) if 0 <= idx < len(values) else 0
            synced_replayed[dataset_id] = max(0, count)
        if synced_replayed:
            self._mixture_anchor_window_start_replayed_local = synced_replayed


def _advance_mixture_anchor_window_progress(
    self,
    *,
    local_batch_blocks: int,
    world_size: int,
    device: torch.device,
) -> None:
    if not self._mixture_enabled or self._mixture_anchor_epochs <= 1:
        return
    if local_batch_blocks <= 0:
        return
    if world_size <= 0:
        raise ValueError("world_size must be > 0 for mixture anchor-window accounting.")

    self._mixture_global_blocks_seen_estimate += int(local_batch_blocks) * int(world_size)
    while self._mixture_anchor_epoch_index < len(
        self._mixture_anchor_boundaries
    ) and self._mixture_global_blocks_seen_estimate >= int(
        self._mixture_anchor_boundaries[self._mixture_anchor_epoch_index]
    ):
        realized = self._ensure_mixture_realized_counter_tensor(device)
        replayed = self._ensure_mixture_replayed_counter_tensor(device)
        self._mixture_anchor_window_start_realized_local_tensor = realized.clone()
        self._mixture_anchor_window_start_replayed_local_tensor = replayed.clone()
        self._mixture_anchor_epoch_index += 1


def _reduce_mixture_realized_blocks(self, fabric: L.Fabric) -> dict[str, int]:
    if not self._mixture_enabled or self._mixture_plan is None:
        return {}
    ids = sorted(self._mixture_plan.target_blocks_per_dataset)
    local_tensor = self._mixture_realized_blocks_local_tensor
    if local_tensor is not None and local_tensor.numel() >= len(self._mixture_dataset_idx_to_id):
        source = local_tensor.to(device=fabric.device, dtype=torch.long)
        local = torch.zeros(len(ids), dtype=torch.long, device=fabric.device)
        for out_idx, dataset_id in enumerate(ids):
            dataset_idx = self._mixture_dataset_id_to_idx.get(dataset_id)
            if dataset_idx is None:
                local[out_idx] = int(self._mixture_realized_blocks_local.get(dataset_id, 0))
                continue
            ds_idx = int(dataset_idx)
            if 0 <= ds_idx < source.numel():
                local[out_idx] = source[ds_idx]
            else:
                local[out_idx] = int(self._mixture_realized_blocks_local.get(dataset_id, 0))
    else:
        local = torch.tensor(
            [int(self._mixture_realized_blocks_local.get(dataset_id, 0)) for dataset_id in ids],
            dtype=torch.long,
            device=fabric.device,
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {dataset_id: int(local[idx].item()) for idx, dataset_id in enumerate(ids)}


def _reduce_mixture_replayed_draws(self, fabric: L.Fabric) -> dict[str, int]:
    if not self._mixture_enabled or self._mixture_plan is None:
        return {}
    ids = sorted(self._mixture_plan.target_blocks_per_dataset)
    local_tensor = self._mixture_replayed_draws_local_tensor
    if local_tensor is not None and local_tensor.numel() >= len(self._mixture_dataset_idx_to_id):
        source = local_tensor.to(device=fabric.device, dtype=torch.long)
        local = torch.zeros(len(ids), dtype=torch.long, device=fabric.device)
        for out_idx, dataset_id in enumerate(ids):
            dataset_idx = self._mixture_dataset_id_to_idx.get(dataset_id)
            if dataset_idx is None:
                local[out_idx] = int(self._mixture_replayed_draws_local.get(dataset_id, 0))
                continue
            ds_idx = int(dataset_idx)
            if 0 <= ds_idx < source.numel():
                local[out_idx] = source[ds_idx]
            else:
                local[out_idx] = int(self._mixture_replayed_draws_local.get(dataset_id, 0))
    else:
        local = torch.tensor(
            [int(self._mixture_replayed_draws_local.get(dataset_id, 0)) for dataset_id in ids],
            dtype=torch.long,
            device=fabric.device,
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {dataset_id: int(local[idx].item()) for idx, dataset_id in enumerate(ids)}


def _reduce_mixture_anchor_window_start_realized(self, fabric: L.Fabric) -> dict[str, int]:
    if not self._mixture_enabled or self._mixture_plan is None:
        return {}
    ids = sorted(self._mixture_plan.target_blocks_per_dataset)
    local_tensor = self._mixture_anchor_window_start_realized_local_tensor
    if local_tensor is not None and local_tensor.numel() >= len(self._mixture_dataset_idx_to_id):
        source = local_tensor.to(device=fabric.device, dtype=torch.long)
        local = torch.zeros(len(ids), dtype=torch.long, device=fabric.device)
        for out_idx, dataset_id in enumerate(ids):
            dataset_idx = self._mixture_dataset_id_to_idx.get(dataset_id)
            if dataset_idx is None:
                local[out_idx] = int(
                    self._mixture_anchor_window_start_realized_local.get(dataset_id, 0)
                )
                continue
            ds_idx = int(dataset_idx)
            if 0 <= ds_idx < source.numel():
                local[out_idx] = source[ds_idx]
            else:
                local[out_idx] = int(
                    self._mixture_anchor_window_start_realized_local.get(dataset_id, 0)
                )
    else:
        local = torch.tensor(
            [
                int(self._mixture_anchor_window_start_realized_local.get(dataset_id, 0))
                for dataset_id in ids
            ],
            dtype=torch.long,
            device=fabric.device,
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {dataset_id: int(local[idx].item()) for idx, dataset_id in enumerate(ids)}


def _reduce_mixture_anchor_window_start_replayed(self, fabric: L.Fabric) -> dict[str, int]:
    if not self._mixture_enabled or self._mixture_plan is None:
        return {}
    ids = sorted(self._mixture_plan.target_blocks_per_dataset)
    local_tensor = self._mixture_anchor_window_start_replayed_local_tensor
    if local_tensor is not None and local_tensor.numel() >= len(self._mixture_dataset_idx_to_id):
        source = local_tensor.to(device=fabric.device, dtype=torch.long)
        local = torch.zeros(len(ids), dtype=torch.long, device=fabric.device)
        for out_idx, dataset_id in enumerate(ids):
            dataset_idx = self._mixture_dataset_id_to_idx.get(dataset_id)
            if dataset_idx is None:
                local[out_idx] = int(
                    self._mixture_anchor_window_start_replayed_local.get(dataset_id, 0)
                )
                continue
            ds_idx = int(dataset_idx)
            if 0 <= ds_idx < source.numel():
                local[out_idx] = source[ds_idx]
            else:
                local[out_idx] = int(
                    self._mixture_anchor_window_start_replayed_local.get(dataset_id, 0)
                )
    else:
        local = torch.tensor(
            [
                int(self._mixture_anchor_window_start_replayed_local.get(dataset_id, 0))
                for dataset_id in ids
            ],
            dtype=torch.long,
            device=fabric.device,
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {dataset_id: int(local[idx].item()) for idx, dataset_id in enumerate(ids)}


def _validate_mixture(self, fabric: L.Fabric) -> None:
    if not self._mixture_val_dataloaders:
        raise RuntimeError(
            "Mixture validation called but no per-source validation dataloaders are available."
        )
    if self._mixture_plan is None:
        raise RuntimeError("Mixture validation called but mixture plan metadata is missing.")

    t0 = time.perf_counter()
    self.model.eval()
    losses_by_source: dict[str, float] = {}

    try:
        for dataset_id in sorted(self._mixture_val_dataloaders):
            dataloader = self._mixture_val_dataloaders[dataset_id]
            local_sum = torch.zeros((), dtype=torch.float32, device=fabric.device)
            local_count = torch.zeros((), dtype=torch.long, device=fabric.device)

            iterator = (
                tqdm(
                    dataloader,
                    desc=f"Validating {dataset_id}...",
                    mininterval=0,
                    colour="green",
                )
                if fabric.global_rank == 0
                else dataloader
            )

            for batch_idx, batch in enumerate(iterator):
                validation_output = self.model.validation_step(batch, batch_idx)
                loss = validation_output["loss"].detach().to(torch.float32)
                local_sum += loss
                local_count += 1

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

            if int(local_count.item()) <= 0:
                raise RuntimeError(
                    f"Validation dataloader for source {dataset_id!r} produced zero batches."
                )

            mean_loss = float((local_sum / local_count).item())
            losses_by_source[dataset_id] = mean_loss
            fabric.log_dict(
                {f"metric/val_loss_{dataset_id}": mean_loss},
                self.state["step_count"],
            )

        weights = self._mixture_plan.effective_weights_by_id
        z = sum(float(weights[dataset_id]) for dataset_id in losses_by_source)
        if z <= 0:
            raise RuntimeError("Mixture validation weights sum to zero.")
        val_loss_weighted = sum(
            (float(weights[dataset_id]) / z) * float(losses_by_source[dataset_id])
            for dataset_id in losses_by_source
        )

        fabric.log_dict({"metric/val_loss_weighted": val_loss_weighted}, self.state["step_count"])
        if math.isfinite(val_loss_weighted):
            fabric.log_dict(
                {"metric/val_ppl_weighted": math.exp(val_loss_weighted)}, self.state["step_count"]
            )

        resolved_blocks = self._reduce_mixture_realized_blocks(fabric)
        resolved_replayed_draws = self._reduce_mixture_replayed_draws(fabric)
        effective_total_blocks = int(
            self._mixture_effective_total_blocks or self._mixture_requested_total_blocks or 0
        )
        mixture_progress_metrics = build_mixture_progress_metrics(
            target_blocks=self._mixture_plan.target_blocks_per_dataset,
            realized_blocks=resolved_blocks,
            effective_total_blocks=effective_total_blocks,
        )
        replay_metrics = build_replay_metrics(
            realized_blocks=resolved_blocks,
            replayed_draws=resolved_replayed_draws,
            namespace="mixture/replay",
        )
        expected_replay_metrics: dict[str, float] = {}
        expected_summary = self._mixture_expected_replay_summary.get("per_source", {})
        for dataset_id in sorted(expected_summary):
            src = expected_summary[dataset_id]
            expected_replay_metrics[f"mixture/replay_expected/replayed_draws_{dataset_id}"] = float(
                src.get("expected_replayed_draws", 0)
            )
            expected_replay_metrics[f"mixture/replay_expected/fraction_{dataset_id}"] = float(
                src.get("expected_replay_fraction", 0.0)
            )
        expected_totals = self._mixture_expected_replay_summary.get("totals", {})
        expected_replay_metrics["mixture/replay_expected/replayed_draws_total"] = float(
            expected_totals.get("expected_replayed_draws", 0)
        )
        expected_replay_metrics["mixture/replay_expected/fraction_total"] = float(
            expected_totals.get("expected_replay_fraction", 0.0)
        )

        window_start_realized = self._reduce_mixture_anchor_window_start_realized(fabric)
        window_start_replayed = self._reduce_mixture_anchor_window_start_replayed(fabric)
        window_realized = {
            dataset_id: max(
                0,
                int(resolved_blocks.get(dataset_id, 0))
                - int(window_start_realized.get(dataset_id, 0)),
            )
            for dataset_id in sorted(resolved_blocks)
        }
        window_replayed = {
            dataset_id: max(
                0,
                int(resolved_replayed_draws.get(dataset_id, 0))
                - int(window_start_replayed.get(dataset_id, 0)),
            )
            for dataset_id in sorted(resolved_replayed_draws)
        }
        replay_window_metrics = build_replay_metrics(
            realized_blocks=window_realized,
            replayed_draws=window_replayed,
            namespace="mixture/replay_window",
        )

        mixture_metrics = dict(mixture_progress_metrics)
        mixture_metrics.update(replay_metrics)
        mixture_metrics.update(replay_window_metrics)
        mixture_metrics.update(expected_replay_metrics)
        mixture_metrics["mixture/replay_window/anchor_epoch_index"] = float(
            self._mixture_anchor_epoch_index
        )
        fabric.log_dict(mixture_metrics, self.state["step_count"])

        self._mixture_last_val_losses = {k: float(v) for k, v in losses_by_source.items()}
        self._mixture_last_val_weighted = float(val_loss_weighted)

        elapsed_time = (time.perf_counter() - t0) * 1000.0
        replay_fraction_total = float(replay_metrics.get("mixture/replay/fraction_total", 0.0))
        replay_window_fraction_total = float(
            replay_window_metrics.get("mixture/replay_window/fraction_total", 0.0)
        )
        self.cli_logger.info(
            "step %s: val_loss_weighted %.4f, replay_fraction_total=%.6f, replay_window_fraction_total=%.6f, anchor_epoch_index=%s, per-source=%s, val time: %.2fms",
            self.state.get("iter_num", 0),
            val_loss_weighted,
            replay_fraction_total,
            replay_window_fraction_total,
            self._mixture_anchor_epoch_index,
            self._mixture_last_val_losses,
            elapsed_time,
        )
        fabric.barrier()
    finally:
        self.model.train()
