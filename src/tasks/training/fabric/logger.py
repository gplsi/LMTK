"""
Utilities for Fabric logging backends used across training tasks.

This module exposes helpers for:
  * Creating CSV loggers whose entries are merged per training step.
  * Building fully-configured WandB loggers with standardized metadata handling.
"""

from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
from types import MethodType

from lightning.fabric.loggers import CSVLogger
from box import Box

try:  # pragma: no cover - Lightning import paths differ by version
    from lightning.fabric.loggers.wandb import WandbLogger
except ImportError:  # pragma: no cover
    try:
        from lightning.pytorch.loggers import WandbLogger  # type: ignore
    except ImportError:  # pragma: no cover
        from pytorch_lightning.loggers import WandbLogger  # type: ignore

T = TypeVar("T")


def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    """
    Create a customized logger instance with an overridden experiment.save method.

    This function instantiates a logger (by default a CSVLogger) using provided positional
    and keyword arguments. It then replaces the logger's experiment.save method with a
    custom implementation that merges CSV log rows by a specified key, 'step'.

    Args:
        *args: Positional arguments to pass to the CSVLogger (or a subclass) constructor.
        cls (Type[T], optional): The logger class to instantiate. Defaults to CSVLogger.
        **kwargs: Keyword arguments to pass to the CSVLogger (or a subclass) constructor.

    Returns:
        T: An instance of the logger with the customized 'save' method.

    Note:
        The customized 'save' method leverages a helper function `merge_by` to combine logs
        sharing the same step number before writing them to a CSV file.
    """
    logger = cls(*args, **kwargs)

    def merge_by(dicts: Iterable[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        """
        Merge a list of dictionaries by a common key.

        This helper function iterates over a list of dictionaries and, for each dictionary
        that contains the specified key, merges dictionaries that share the same key value.
        The result is returned as a list of merged dictionaries, sorted by the key's value.
        """
        from collections import defaultdict

        out: Dict[Any, Dict[str, Any]] = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]

    def save(self) -> None:
        """
        Customized save method for the experiment.

        This method overrides the default experiment.save behavior. It merges the logged
        metrics based on the 'step' key, determines the complete set of CSV columns from
        the merged data, and then writes the resulting list of dictionaries to the CSV file.
        """
        import csv

        if not self.metrics:
            return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

    logger.experiment.save = MethodType(save, logger.experiment)

    return logger


def _sanitize_for_wandb(config: Any) -> Any:
    """
    Convert Box or complex objects into WandB-friendly primitives.
    """
    if isinstance(config, Box):
        return {k: _sanitize_for_wandb(v) for k, v in config.items()}
    if isinstance(config, dict):
        return {k: _sanitize_for_wandb(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_sanitize_for_wandb(v) for v in config]
    if isinstance(config, (str, int, float, bool)) or config is None:
        return config
    return str(config)


def _config_get(config: Union[Box, Dict[str, Any]], key: str, default: Any = None) -> Any:
    """
    Access configuration values agnostic to Box or dict backing types.
    """
    if isinstance(config, Box):
        return getattr(config, key, default)
    return config.get(key, default)


def create_wandb_logger(
    config: Union[Box, Dict[str, Any]],
    *,
    allow_val_change: bool = True,
    logger: Optional[Any] = None,
) -> WandbLogger:
    """
    Create a fully-configured WandB logger from the experiment configuration.

    Args:
        config: Experiment configuration (Box or dict) containing WandB fields.
        allow_val_change: Forwarded to WandB logger to enable config updates.
        logger: Optional logger for emitting warnings (expects .warning).

    Returns:
        WandbLogger: Instantiated WandB logger.
    """
    wandb_project = _config_get(config, "wandb_project")
    wandb_entity = _config_get(config, "wandb_entity")

    if not wandb_project or not wandb_entity:
        raise ValueError("wandb_project and wandb_entity must be provided when using WandB logging.")

    tags = _config_get(config, "wandb_tags")
    if isinstance(tags, str):
        tags = [tags]

    wandb_kwargs: Dict[str, Any] = {
        "project": wandb_project,
        "entity": wandb_entity,
        "log_model": bool(_config_get(config, "log_model", False)),
        "name": _config_get(config, "wandb_run_name"),
        "tags": tags,
        "group": _config_get(config, "wandb_group"),
        "job_type": _config_get(config, "wandb_job_type"),
        "notes": _config_get(config, "wandb_notes"),
        "mode": _config_get(config, "wandb_mode"),
        "id": _config_get(config, "wandb_id"),
        "resume": _config_get(config, "wandb_resume", "allow"),
        "config": _sanitize_for_wandb(config),
        "allow_val_change": allow_val_change,
    }

    output_dir = _config_get(config, "output_dir")
    if output_dir:
        wandb_kwargs["save_dir"] = output_dir

    # Remove None entries so WandB receives only explicit overrides
    wandb_kwargs = {key: value for key, value in wandb_kwargs.items() if value is not None}

    # Emit friendly message if logging is disabled
    if wandb_kwargs.get("mode") == "disabled" and logger is not None:
        logger.warning("WandB logger created with `mode='disabled'`. Run data will not be uploaded.")

    return WandbLogger(**wandb_kwargs)
