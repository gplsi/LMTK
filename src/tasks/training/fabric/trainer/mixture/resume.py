"""Mixture resume/runtime-state compatibility implementations for MixtureRuntimeMixin."""

import json
from typing import Any

import lightning as L

from .constants import MIXTURE_META_VERSION, MIXTURE_RUNTIME_VERSION, MIXTURE_SAMPLING_ALGORITHM


def _build_mixture_runtime_state(self) -> dict[str, Any]:
    if not self._mixture_enabled:
        return {}
    self._sync_mixture_realized_blocks_local_from_tensor()
    self._sync_mixture_replayed_draws_local_from_tensor()
    self._sync_mixture_anchor_window_start_local_from_tensors()
    return {
        "mixture_runtime_version": MIXTURE_RUNTIME_VERSION,
        "realized_blocks_local": {
            dataset_id: int(v)
            for dataset_id, v in sorted(self._mixture_realized_blocks_local.items())
        },
        "replayed_draws_local": {
            dataset_id: int(v)
            for dataset_id, v in sorted(self._mixture_replayed_draws_local.items())
        },
        "anchor_epoch_index": int(self._mixture_anchor_epoch_index),
        "global_blocks_seen_estimate": int(self._mixture_global_blocks_seen_estimate),
        "anchor_window_start_realized_local": {
            dataset_id: int(v)
            for dataset_id, v in sorted(self._mixture_anchor_window_start_realized_local.items())
        },
        "anchor_window_start_replayed_local": {
            dataset_id: int(v)
            for dataset_id, v in sorted(self._mixture_anchor_window_start_replayed_local.items())
        },
        "last_val_losses": {
            dataset_id: float(v) for dataset_id, v in sorted(self._mixture_last_val_losses.items())
        },
        "last_val_weighted": (
            float(self._mixture_last_val_weighted)
            if self._mixture_last_val_weighted is not None
            else None
        ),
    }


def _restore_mixture_runtime_state(
    self, runtime_state: dict[str, Any] | None, *, strict: bool
) -> None:
    if not self._mixture_enabled:
        return
    if not runtime_state:
        if strict:
            raise ValueError(
                "Mixture resume requires checkpoint key 'mixture_runtime' for mixture_meta_version='v3'."
            )
        self.cli_logger.warning(
            "Mixture resume checkpoint is missing 'mixture_runtime'; "
            "continuing with zeroed local counters for this rank."
        )
        return
    if not isinstance(runtime_state, dict):
        if strict:
            raise ValueError(
                "Mixture resume requires 'mixture_runtime' to be a dictionary for mixture_meta_version='v3'."
            )
        self.cli_logger.warning(
            "Mixture resume checkpoint has invalid 'mixture_runtime' type (%s); "
            "continuing with zeroed local counters for this rank.",
            type(runtime_state).__name__,
        )
        return
    loaded_runtime_version = runtime_state.get("mixture_runtime_version", None)
    if strict and loaded_runtime_version != MIXTURE_RUNTIME_VERSION:
        raise ValueError(
            "Mixture resume compatibility check failed: mixture_runtime_version mismatch "
            f"(expected={MIXTURE_RUNTIME_VERSION!r}, loaded={loaded_runtime_version!r})."
        )

    loaded_counts = runtime_state.get("realized_blocks_local", {})
    if strict and not isinstance(loaded_counts, dict):
        raise ValueError(
            "Mixture resume requires 'mixture_runtime.realized_blocks_local' to be a dictionary."
        )
    if isinstance(loaded_counts, dict):
        restored: dict[str, int] = {}
        for dataset_id in sorted(self._mixture_realized_blocks_local):
            if strict and dataset_id not in loaded_counts:
                raise ValueError(
                    f"Mixture resume requires realized_blocks_local entry for dataset_id={dataset_id!r}."
                )
            raw = loaded_counts.get(dataset_id, 0)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        f"Mixture resume has non-integer realized_blocks_local value for dataset_id={dataset_id!r}: {raw!r}."
                    )
                restored[dataset_id] = 0
                continue
            if strict and parsed < 0:
                raise ValueError(
                    f"Mixture resume has negative realized_blocks_local value for dataset_id={dataset_id!r}: {raw!r}."
                )
            restored[dataset_id] = max(0, parsed)
        self._mixture_realized_blocks_local = restored
        self._mixture_realized_blocks_local_tensor = None

    loaded_replayed = runtime_state.get("replayed_draws_local", {})
    if strict and not isinstance(loaded_replayed, dict):
        raise ValueError(
            "Mixture resume requires 'mixture_runtime.replayed_draws_local' to be a dictionary."
        )
    if isinstance(loaded_replayed, dict):
        restored_replayed: dict[str, int] = {}
        for dataset_id in sorted(self._mixture_replayed_draws_local):
            if strict and dataset_id not in loaded_replayed:
                raise ValueError(
                    f"Mixture resume requires replayed_draws_local entry for dataset_id={dataset_id!r}."
                )
            raw = loaded_replayed.get(dataset_id, 0)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        "Mixture resume has non-integer replayed_draws_local value "
                        f"for dataset_id={dataset_id!r}: {raw!r}."
                    )
                restored_replayed[dataset_id] = 0
                continue
            if strict and parsed < 0:
                raise ValueError(
                    "Mixture resume has negative replayed_draws_local value "
                    f"for dataset_id={dataset_id!r}: {raw!r}."
                )
            restored_replayed[dataset_id] = max(0, parsed)
        self._mixture_replayed_draws_local = restored_replayed
        self._mixture_replayed_draws_local_tensor = None

    loaded_anchor_epoch_index = runtime_state.get("anchor_epoch_index", 0)
    try:
        parsed_anchor_epoch_index = int(loaded_anchor_epoch_index)
    except (TypeError, ValueError):
        if strict:
            raise ValueError(
                f"Mixture resume has invalid anchor_epoch_index value: {loaded_anchor_epoch_index!r}."
            )
        self._mixture_anchor_epoch_index = 0
    else:
        if strict and parsed_anchor_epoch_index < 0:
            raise ValueError(
                f"Mixture resume has invalid negative anchor_epoch_index value: {loaded_anchor_epoch_index!r}."
            )
        self._mixture_anchor_epoch_index = max(0, parsed_anchor_epoch_index)

    loaded_blocks_seen = runtime_state.get("global_blocks_seen_estimate", 0)
    try:
        parsed_blocks_seen = int(loaded_blocks_seen)
    except (TypeError, ValueError):
        if strict:
            raise ValueError(
                "Mixture resume has invalid global_blocks_seen_estimate value: "
                f"{loaded_blocks_seen!r}."
            )
        self._mixture_global_blocks_seen_estimate = 0
    else:
        if strict and parsed_blocks_seen < 0:
            raise ValueError(
                "Mixture resume has invalid negative global_blocks_seen_estimate value: "
                f"{loaded_blocks_seen!r}."
            )
        self._mixture_global_blocks_seen_estimate = max(0, parsed_blocks_seen)

    loaded_window_start_realized = runtime_state.get("anchor_window_start_realized_local", {})
    if strict and not isinstance(loaded_window_start_realized, dict):
        raise ValueError(
            "Mixture resume requires 'mixture_runtime.anchor_window_start_realized_local' to be a dictionary."
        )
    if isinstance(loaded_window_start_realized, dict):
        restored_window_realized: dict[str, int] = {}
        for dataset_id in sorted(self._mixture_anchor_window_start_realized_local):
            if strict and dataset_id not in loaded_window_start_realized:
                raise ValueError(
                    "Mixture resume requires anchor_window_start_realized_local entry "
                    f"for dataset_id={dataset_id!r}."
                )
            raw = loaded_window_start_realized.get(dataset_id, 0)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        "Mixture resume has non-integer anchor_window_start_realized_local value "
                        f"for dataset_id={dataset_id!r}: {raw!r}."
                    )
                restored_window_realized[dataset_id] = 0
                continue
            if strict and parsed < 0:
                raise ValueError(
                    "Mixture resume has negative anchor_window_start_realized_local value "
                    f"for dataset_id={dataset_id!r}: {raw!r}."
                )
            restored_window_realized[dataset_id] = max(0, parsed)
        self._mixture_anchor_window_start_realized_local = restored_window_realized
        self._mixture_anchor_window_start_realized_local_tensor = None

    loaded_window_start_replayed = runtime_state.get("anchor_window_start_replayed_local", {})
    if strict and not isinstance(loaded_window_start_replayed, dict):
        raise ValueError(
            "Mixture resume requires 'mixture_runtime.anchor_window_start_replayed_local' to be a dictionary."
        )
    if isinstance(loaded_window_start_replayed, dict):
        restored_window_replayed: dict[str, int] = {}
        for dataset_id in sorted(self._mixture_anchor_window_start_replayed_local):
            if strict and dataset_id not in loaded_window_start_replayed:
                raise ValueError(
                    "Mixture resume requires anchor_window_start_replayed_local entry "
                    f"for dataset_id={dataset_id!r}."
                )
            raw = loaded_window_start_replayed.get(dataset_id, 0)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        "Mixture resume has non-integer anchor_window_start_replayed_local value "
                        f"for dataset_id={dataset_id!r}: {raw!r}."
                    )
                restored_window_replayed[dataset_id] = 0
                continue
            if strict and parsed < 0:
                raise ValueError(
                    "Mixture resume has negative anchor_window_start_replayed_local value "
                    f"for dataset_id={dataset_id!r}: {raw!r}."
                )
            restored_window_replayed[dataset_id] = max(0, parsed)
        self._mixture_anchor_window_start_replayed_local = restored_window_replayed
        self._mixture_anchor_window_start_replayed_local_tensor = None

    loaded_losses = runtime_state.get("last_val_losses", {})
    if strict and not isinstance(loaded_losses, dict):
        raise ValueError(
            "Mixture resume requires 'mixture_runtime.last_val_losses' to be a dictionary."
        )
    if isinstance(loaded_losses, dict):
        restored_losses: dict[str, float] = {}
        for dataset_id, value in loaded_losses.items():
            try:
                restored_losses[str(dataset_id)] = float(value)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        f"Mixture resume has invalid last_val_losses value for dataset_id={dataset_id!r}: {value!r}."
                    )
                continue
        self._mixture_last_val_losses = restored_losses

    loaded_weighted = runtime_state.get("last_val_weighted", None)
    if loaded_weighted is None:
        self._mixture_last_val_weighted = None
    else:
        try:
            self._mixture_last_val_weighted = float(loaded_weighted)
        except (TypeError, ValueError):
            if strict:
                raise ValueError(
                    f"Mixture resume has invalid last_val_weighted value: {loaded_weighted!r}."
                )
            self._mixture_last_val_weighted = None


def _build_mixture_resume_meta(self, fabric: L.Fabric) -> dict[str, Any]:
    if not self._mixture_enabled or self._mixture_plan is None:
        return {}
    grad_accum = self._resolved_gradient_accumulation_steps()

    return {
        "mixture_meta_version": MIXTURE_META_VERSION,
        "weight_mode": self._mixture_plan.weight_mode,
        "sources": [
            {
                "dataset_id": dataset_id,
                "configured_weight": self._mixture_configured_weights_by_id.get(dataset_id),
                "effective_weight": float(self._mixture_plan.effective_weights_by_id[dataset_id]),
            }
            for dataset_id in sorted(self._mixture_plan.effective_weights_by_id)
        ],
        "budget_mode": self._mixture_plan.budget_mode,
        "anchor_dataset_id": self._mixture_plan.anchor_dataset_id,
        "requested_total_blocks": int(self._mixture_requested_total_blocks or 0),
        "effective_total_blocks": int(self._mixture_effective_total_blocks or 0),
        "sampling_algorithm": MIXTURE_SAMPLING_ALGORITHM,
        "world_size": int(fabric.world_size),
        "batch_size": int(self.config.batch_size),
        "gradient_accumulation_steps": grad_accum,
        "dataset_index_map": {
            str(idx): dataset_id
            for idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items())
        },
        "schedule_seed": int(self._mixture_schedule_seed),
        "hash_algorithm": "blake2b_u64_v1",
        "allocation_algorithm": "hamilton_lr_lexicographic_v1",
        "split_seed_algorithm": "blake2b_u64_mod_2147483647_v1",
    }


def _validate_mixture_resume_compatibility(self, expected_meta: dict[str, Any]) -> None:
    loaded_meta = self.state.get("mixture_meta")
    if loaded_meta is None:
        raise ValueError(
            "Mixture resume requires checkpoint metadata key 'mixture_meta', but it was not found."
        )
    if not isinstance(loaded_meta, dict):
        raise ValueError(
            "Mixture resume requires checkpoint key 'mixture_meta' to be a dictionary, "
            f"got {type(loaded_meta).__name__}."
        )

    loaded_version = loaded_meta.get("mixture_meta_version", None)
    expected_version = expected_meta.get("mixture_meta_version", None)

    if loaded_version != expected_version:
        raise ValueError(
            "Mixture resume compatibility check failed: mixture_meta_version mismatch "
            f"(expected={expected_version!r}, loaded={loaded_version!r})."
        )

    loaded_sampling_algorithm = loaded_meta.get("sampling_algorithm", None)
    expected_sampling_algorithm = expected_meta.get("sampling_algorithm", None)
    if loaded_sampling_algorithm != expected_sampling_algorithm:
        raise ValueError(
            "Mixture resume compatibility check failed: sampling_algorithm mismatch "
            f"(expected={expected_sampling_algorithm!r}, loaded={loaded_sampling_algorithm!r})."
        )

    dataset_index_map = loaded_meta.get("dataset_index_map", None)
    if not isinstance(dataset_index_map, dict):
        raise ValueError(
            "Mixture resume compatibility check failed: 'dataset_index_map' is missing or invalid."
        )
    mapped_ids = [str(v) for v in dataset_index_map.values()]
    if len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError(
            "Mixture resume compatibility check failed: 'dataset_index_map' contains duplicate dataset_id values."
        )

    if loaded_meta != expected_meta:
        raise ValueError(
            "Mixture resume compatibility check failed: checkpoint mixture_meta does not match current "
            f"configuration.\nExpected: {json.dumps(expected_meta, sort_keys=True)}\n"
            f"Loaded: {json.dumps(loaded_meta, sort_keys=True)}"
        )
