"""Mixin exposing mixture runtime/resume/report methods for FabricTrainerBase."""

from src.tasks.training.fabric.trainer.mixture import reporting as mixture_reporting
from src.tasks.training.fabric.trainer.mixture import resume as mixture_resume
from src.tasks.training.fabric.trainer.mixture import runtime as mixture_runtime


class MixtureRuntimeMixin:
    _ensure_mixture_realized_counter_tensor = (
        mixture_runtime._ensure_mixture_realized_counter_tensor
    )
    _ensure_mixture_replayed_counter_tensor = (
        mixture_runtime._ensure_mixture_replayed_counter_tensor
    )
    _sync_mixture_realized_blocks_local_from_tensor = (
        mixture_runtime._sync_mixture_realized_blocks_local_from_tensor
    )
    _sync_mixture_replayed_draws_local_from_tensor = (
        mixture_runtime._sync_mixture_replayed_draws_local_from_tensor
    )
    _sync_mixture_anchor_window_start_local_from_tensors = (
        mixture_runtime._sync_mixture_anchor_window_start_local_from_tensors
    )
    _advance_mixture_anchor_window_progress = (
        mixture_runtime._advance_mixture_anchor_window_progress
    )
    _reduce_mixture_realized_blocks = mixture_runtime._reduce_mixture_realized_blocks
    _reduce_mixture_replayed_draws = mixture_runtime._reduce_mixture_replayed_draws
    _reduce_mixture_anchor_window_start_realized = (
        mixture_runtime._reduce_mixture_anchor_window_start_realized
    )
    _reduce_mixture_anchor_window_start_replayed = (
        mixture_runtime._reduce_mixture_anchor_window_start_replayed
    )
    _validate_mixture = mixture_runtime._validate_mixture

    _build_mixture_runtime_state = mixture_resume._build_mixture_runtime_state
    _restore_mixture_runtime_state = mixture_resume._restore_mixture_runtime_state
    _build_mixture_resume_meta = mixture_resume._build_mixture_resume_meta
    _validate_mixture_resume_compatibility = mixture_resume._validate_mixture_resume_compatibility

    _write_mixture_report = mixture_reporting._write_mixture_report
