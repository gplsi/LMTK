"""Mixin exposing mixture setup/config methods for FabricTrainerBase."""

from src.tasks.training.fabric.trainer.mixture import setup as mixture_setup


class MixtureSetupMixin:
    _get_mixture_config = mixture_setup._get_mixture_config
    _get_sources_config = mixture_setup._get_sources_config
    _is_mixture_enabled = mixture_setup._is_mixture_enabled
    _derive_validation_split_seed = mixture_setup._derive_validation_split_seed
    _collect_source_config_map = mixture_setup._collect_source_config_map
    _validate_mixture_source_compatibility = mixture_setup._validate_mixture_source_compatibility
    _build_mixture_packing_dataloaders = mixture_setup._build_mixture_packing_dataloaders
