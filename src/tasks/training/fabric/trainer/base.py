"""
This module defines a base trainer class using Lightning Fabric for training deep learning models.
It encapsulates functionalities such as dataset loading, model instantiation, training loop, validation,
checkpointing, logging, and gradient accumulation. The FabricTrainerBase class is designed as an abstract base
class, providing a template for custom training strategies.
"""

import math
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from datasets import Dataset as HFDataset
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
from typing import Any, Mapping, Tuple, Union, Optional
from box import Box
import lightning as L
import os
import shutil

# Import custom utilities
from src.tasks.training.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.training.fabric.logger import step_csv_logger, create_wandb_logger
from src.tasks.training.utils import *
from utils.logging import get_logger
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, DeepSpeedStrategy, DataParallelStrategy
from src.tasks.training.utils import select_optimizer, select_scheduler, deterministic



# Specific Model classes for the framework
from src.tasks.training.fabric.model.clm import FabricCLM
from src.tasks.training.fabric.model.mlm import FabricMLM
from src.tasks.training.fabric.model.instruction import FabricInstruction
from src.tasks.training.data.packing import PackedSequenceDataset, build_packing_dataloader
from src.tasks.training.data.packing_index import INDEX_VERSION, PackingIndex
from src.tasks.training.data.mixture import (
    MixturePackedDataset,
    MixturePlan,
    blake2b_u64,
    resolve_effective_total_blocks,
    resolve_mixture_plan,
)
from src.utils.dataset import (
    TOKENIZATION_METADATA_FILENAME,
    extract_eos_token_id,
    read_tokenization_metadata,
)

MODEL_CLASS_MAP = {
    "clm_training": FabricCLM,
    "mlm_training": FabricMLM,
    "instruction": FabricInstruction,
}

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MIXTURE_META_VERSION = "v2"
MIXTURE_RUNTIME_VERSION = "v1"
RUN_METADATA_VERSION = "v1"

class FabricTrainerBase(ABC):
    """
    Abstract base trainer class for managing training using Lightning Fabric.
    
    This class provides the basic structure required for training by handling dataset preparation,
    strategy setup, logging, checkpointing, gradient accumulation, and validation.
    It is intended to be subclassed with a concrete implementation of the _setup_strategy method.
    """
    def __init__(
        self,
        devices: Union[int, str],
        config: Box,
        dataset: HFDataset,
        checkpoint_path: str = None,
        num_nodes: Union[int, None] = None,
        devices_per_node: Union[int, None] = None,
    ) -> None:
        """
        Initialize the FabricTrainerBase instance.

        Parameters:
        - devices (int | str): The number of devices to use for training or "cpu".
        - config (Box): Configuration object containing training parameters. Can include:
            - checkpoint: Path to resume complete training state
            - initial_weights_checkpoint: Path to load only model weights for transfer learning
        - dataset (HFDataset): The dataset (or DatasetDict) used for training.
        - checkpoint_path (str, optional): Path to a checkpoint to resume training, if applicable.
        - num_nodes (int, optional): Number of nodes to use for training.
        - devices_per_node (int, optional): Number of devices per node.

        Raises:
        - ValueError: If dataset is None or if both checkpoint types are specified.
        """
        self.cli_logger = get_logger(__name__, config.verbose_level)
        
        if dataset is None:
            raise ValueError("Dataset must be provided for training.")
        
        # Validate checkpoint parameters
        checkpoint = getattr(config, 'checkpoint', None)
        initial_weights_checkpoint = getattr(config, 'initial_weights_checkpoint', None)
        if checkpoint is not None and initial_weights_checkpoint is not None:
            raise ValueError("Cannot specify both 'checkpoint' and 'initial_weights_checkpoint'. Use 'checkpoint' to resume training or 'initial_weights_checkpoint' for transfer learning.")
        
        self.devices = devices
        self.num_nodes = num_nodes if num_nodes is not None else 1
        self.devices_per_node = devices_per_node
        self.config = config
        self._packing_enabled = False
        self._mixture_enabled = False
        self.checkpoint_path = checkpoint_path
        self.state = {}
        self.dataset = dataset
        self._mixture_plan: MixturePlan | None = None
        self._mixture_source_configs: dict[str, Box] = {}
        self._mixture_source_split_metadata: dict[str, dict[str, Any]] = {}
        self._mixture_dataset_idx_to_id: dict[int, str] = {}
        self._mixture_dataset_id_to_idx: dict[str, int] = {}
        self._mixture_val_dataloaders: dict[str, DataLoader] = {}
        self._mixture_realized_blocks_local: dict[str, int] = {}
        self._mixture_realized_blocks_local_tensor: torch.Tensor | None = None
        self._mixture_last_val_losses: dict[str, float] = {}
        self._mixture_last_val_weighted: float | None = None
        self._mixture_requested_total_blocks: int | None = None
        self._mixture_effective_total_blocks: int | None = None
        self._mixture_report_path: Path | None = None
        self._mixture_schedule_seed: int = 0
        self._mixture_configured_weights_by_id: dict[str, float | None] = {}
        self._mixture_source_blocks_by_id: dict[str, int] = {}
        self._training_schedule_metadata: dict[str, Any] = {}
        
        # Load datasets and create dataloaders
        result = self._load_fabric_datasets_dataloaders(self.config, self.dataset)
        self.datasets = result["datasets"]
        self.dataloaders = result["dataloaders"]
    
    @abstractmethod
    def _setup_strategy(self) -> Union[FSDPStrategy, DDPStrategy, DeepSpeedStrategy, DataParallelStrategy]:
        """
        Abstract method to set up the training strategy.
        
        This method should be implemented in subclasses to return the desired training strategy instance.
        """
        pass

    def _instantiate_model(self):
        model_type = self.config.get("task", "")
        if model_type not in MODEL_CLASS_MAP:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_class = MODEL_CLASS_MAP[model_type]
        # Convert Box config to dict to ensure proper unpacking
        config_dict = dict(self.config)
        # Debug: print available keys
        self.cli_logger.debug(f"Available config keys: {list(config_dict.keys())}")
        self.cli_logger.debug(f"Looking for model_name, found: {config_dict.get('model_name', 'NOT FOUND')}")
        return model_class(**config_dict)
    
    def setup(self) -> None:
        """
        Set up and launch the training pipeline.

        This method configures the training strategy, sets up loggers, and then launches the training pipeline using Lightning Fabric.
        """
        self.cli_logger.info("Setting up FSDP strategy.")
        self.cli_logger.info(
            "Fabric config: devices=%s num_nodes=%s devices_per_node=%s",
            self.devices,
            self.num_nodes,
            self.devices_per_node,
        )
        torch.set_float32_matmul_precision("high")
        # Debug logging for configuration values that might cause type issues
        config_keys_to_check = ['gradient_accumulation_steps', 'validations_per_epoch', 'max_epochs', 'max_steps', 'batch_size', 'eval_batch_size']
        for key in config_keys_to_check:
            if hasattr(self.config, key):
                value = getattr(self.config, key)
                self.cli_logger.debug(f"Config {key}: value={value}, type={type(value)}")
        
        strategy = self._setup_strategy()
        loggers = self._set_loggers()

        fabric = L.Fabric(
            devices=self.devices,
            num_nodes=self.num_nodes,
            strategy=strategy,
            precision=self.config.precision,
            loggers=loggers,
        )

        self.cli_logger.info(f"Precision {self.config.precision}")

        self.hparams = {
            "task": str(self.config.get("task", "")),
            "experiment_name": str(self.config.get("experiment_name", "")),
            "model_name": str(self.config.get("model_name", "")),
            "precision": str(self.config.get("precision", "")),
            "seed": int(self.config.get("seed", 0) or 0),
            "devices": str(self.devices),
            "num_nodes": int(self.num_nodes),
            "devices_per_node": (
                int(self.devices_per_node) if self.devices_per_node is not None else None
            ),
        }
        self.cli_logger.debug(self.hparams)

        fabric.launch(self._pipeline)

    def _set_loggers(self) -> list:
        """
        Set up training loggers based on configuration.

        Returns:
        - list: A list of logger objects to be used during training.
        """
        logger = step_csv_logger(
            self.config.output_dir, 
            self.config.model_name, 
            flush_logs_every_n_steps=self.config.get("log_iter_interval", 100)
        )
        loggers = [logger]

        if self.config.get("logging_config", None) == "wandb":
            wandb_logger = create_wandb_logger(self.config, logger=self.cli_logger)
            loggers.append(wandb_logger)

        return loggers

    def _get_packing_config(self) -> Optional[Box]:
        dataset_cfg = getattr(self.config, "dataset", None)
        if dataset_cfg is None:
            return None
        packing = dataset_cfg.get("packing", None) if hasattr(dataset_cfg, "get") else getattr(dataset_cfg, "packing", None)
        if packing is None:
            return None
        if isinstance(packing, Box):
            return packing
        if isinstance(packing, dict):
            return Box(packing, box_dots=True)
        raise TypeError(f"Unsupported dataset.packing type: {type(packing)}")

    def _packing_index_lock_timeout_s(self, packing: Box) -> int:
        value = packing.get("lock_timeout_s", None)
        if value is None:
            return 1800
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"dataset.packing.lock_timeout_s must be an integer, got {value!r}.")
        if parsed <= 0:
            raise ValueError("dataset.packing.lock_timeout_s must be > 0.")
        return parsed

    def _packing_index_stale_lock_age_s(self, packing: Box) -> int | None:
        value = packing.get("stale_lock_age_s", None)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"dataset.packing.stale_lock_age_s must be an integer, got {value!r}.")
        if parsed <= 0:
            raise ValueError("dataset.packing.stale_lock_age_s must be > 0 when set.")
        return parsed

    def _packing_index_lock_lease_heartbeat_s(self, packing: Box) -> int:
        value = packing.get("lock_lease_heartbeat_s", None)
        if value is None:
            return 30
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"dataset.packing.lock_lease_heartbeat_s must be an integer, got {value!r}.")
        if parsed <= 0:
            raise ValueError("dataset.packing.lock_lease_heartbeat_s must be > 0.")
        return parsed

    def _allow_shared_cache_cleanup(self, packing: Box) -> bool:
        return bool(packing.get("allow_shared_cache_cleanup", False))

    def _clean_index_cache_on_start(self, packing: Box) -> bool:
        return bool(packing.get("clean_index_cache_on_start", False))

    def _clean_index_cache_on_success(self, packing: Box) -> bool:
        return bool(packing.get("clean_index_cache_on_success", False))

    def _read_json_file(self, path: Path) -> dict[str, Any] | None:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                return None
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _collect_lock_failure_context(
        self,
        *,
        index_cache_dir: Path,
        split_name: str | None,
        lock_timeout_s: int,
        stale_lock_age_s: int | None,
        lock_lease_heartbeat_s: int,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "lock_timeout_s": int(lock_timeout_s),
            "stale_lock_age_s": int(stale_lock_age_s) if stale_lock_age_s is not None else None,
            "lock_lease_heartbeat_s": int(lock_lease_heartbeat_s),
            "index_cache_dir": str(index_cache_dir),
            "split": split_name,
        }
        if split_name is None:
            return context

        root = index_cache_dir / f"v{INDEX_VERSION}" / str(split_name)
        lock_path = root / "LOCK"
        lease_path = root / "LEASE.json"
        context["lock_path"] = str(lock_path)
        context["lease_path"] = str(lease_path)
        context["lock_payload"] = self._read_json_file(lock_path)
        lease_payload = self._read_json_file(lease_path)
        context["lease_payload"] = lease_payload

        lease_updated = None
        lease_age = None
        if isinstance(lease_payload, dict):
            candidate = lease_payload.get("updated_at_unix", lease_payload.get("created_at_unix", None))
            try:
                lease_updated = float(candidate) if candidate is not None else None
            except Exception:
                lease_updated = None
            if lease_updated is not None:
                lease_age = max(0.0, time.time() - lease_updated)
        context["lease_updated_at_unix"] = lease_updated
        context["lease_age_s"] = lease_age
        return context

    def _validate_packing_lock_timings(
        self,
        *,
        lock_timeout_s: int,
        stale_lock_age_s: int | None,
        lock_lease_heartbeat_s: int,
    ) -> None:
        if lock_timeout_s <= 0:
            raise ValueError("dataset.packing.lock_timeout_s must be > 0.")
        if lock_lease_heartbeat_s <= 0:
            raise ValueError("dataset.packing.lock_lease_heartbeat_s must be > 0.")
        if stale_lock_age_s is not None and stale_lock_age_s <= lock_lease_heartbeat_s:
            raise ValueError(
                "dataset.packing.stale_lock_age_s must be greater than dataset.packing.lock_lease_heartbeat_s "
                "to avoid breaking active builders."
            )

    def _is_run_scoped_index_cache_dir(self, index_cache_dir: Path) -> bool:
        output_dir_value = self.config.get("output_dir", None)
        if output_dir_value is None:
            return False
        try:
            output_dir = Path(str(output_dir_value)).resolve()
            cache_dir = index_cache_dir.resolve()
        except Exception:
            return False
        return cache_dir == output_dir or output_dir in cache_dir.parents

    def _maybe_clean_index_cache_version_dir(
        self,
        fabric: L.Fabric,
        index_cache_dir: Path,
        *,
        packing: Box,
        trigger: str,
    ) -> None:
        if fabric.global_rank != 0:
            return

        if not self._is_run_scoped_index_cache_dir(index_cache_dir) and not self._allow_shared_cache_cleanup(packing):
            self.cli_logger.warning(
                "Skipping packing index cache cleanup (%s) for shared cache dir %s. "
                "Set dataset.packing.allow_shared_cache_cleanup=true to override.",
                trigger,
                index_cache_dir,
            )
            return

        version_dir = index_cache_dir / f"v{INDEX_VERSION}"
        if not version_dir.exists():
            return
        active_locks = [path for path in version_dir.rglob("LOCK") if path.exists()]
        if active_locks:
            self.cli_logger.warning(
                "Skipping packing index cache cleanup (%s) for %s because active lock files exist: %s",
                trigger,
                version_dir,
                [str(path) for path in active_locks],
            )
            return
        try:
            shutil.rmtree(version_dir)
            self.cli_logger.info("Removed packing index cache version dir %s", version_dir)
        except Exception as exc:
            self.cli_logger.warning(
                "Failed to remove packing index cache version dir %s: %s",
                version_dir,
                str(exc),
            )

    def _cleanup_index_caches_on_success(self, fabric: L.Fabric, packing: Box) -> None:
        if not self._clean_index_cache_on_success(packing):
            return
        fabric.barrier()
        if fabric.global_rank != 0:
            return

        cache_dirs: set[Path] = set()
        if self._mixture_enabled:
            for dataset_id in sorted(self._mixture_source_configs):
                source_cfg = self._mixture_source_configs[dataset_id]
                dataset_path_value = source_cfg.get("nameOrPath", None)
                if not dataset_path_value:
                    continue
                dataset_path = Path(str(dataset_path_value))
                index_cache_dir_value = source_cfg.get("index_cache_dir", None)
                index_cache_dir = (
                    Path(str(index_cache_dir_value)) if index_cache_dir_value else (dataset_path / ".packing_index")
                )
                cache_dirs.add(index_cache_dir)
        else:
            dataset_cfg = getattr(self.config, "dataset", None)
            dataset_path_value = None
            if dataset_cfg is not None:
                dataset_path_value = (
                    dataset_cfg.get("nameOrPath", None)
                    if hasattr(dataset_cfg, "get")
                    else getattr(dataset_cfg, "nameOrPath", None)
                )
            if dataset_path_value:
                dataset_path = Path(str(dataset_path_value))
                index_cache_dir_value = packing.get("index_cache_dir", None)
                index_cache_dir = (
                    Path(str(index_cache_dir_value)) if index_cache_dir_value else (dataset_path / ".packing_index")
                )
                cache_dirs.add(index_cache_dir)

        for index_cache_dir in sorted(cache_dirs):
            self._maybe_clean_index_cache_version_dir(
                fabric,
                index_cache_dir,
                packing=packing,
                trigger="success",
            )

    def _is_packing_enabled(self) -> bool:
        packing = self._get_packing_config()
        return bool(packing and packing.get("enabled", False))

    def _get_mixture_config(self) -> Optional[Box]:
        dataset_cfg = getattr(self.config, "dataset", None)
        if dataset_cfg is None:
            return None
        mixture = dataset_cfg.get("mixture", None) if hasattr(dataset_cfg, "get") else getattr(dataset_cfg, "mixture", None)
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
        sources = dataset_cfg.get("sources", None) if hasattr(dataset_cfg, "get") else getattr(dataset_cfg, "sources", None)
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

    def _resolve_eos_token_id(self, packing: Box) -> int:
        insert_eos = packing.get("insert_eos", None)
        if insert_eos is None:
            insert_eos = True
        if not bool(insert_eos):
            raise ValueError("_resolve_eos_token_id called with insert_eos disabled.")

        if packing.get("eos_token_id", None) is not None:
            return int(packing.eos_token_id)

        sources = self._get_sources_config()
        if sources:
            source_eos_by_id = {
                str(source_cfg.get("dataset_id")): int(source_cfg.get("eos_token_id"))
                for source_cfg in sources
                if source_cfg.get("eos_token_id", None) is not None
            }
            if source_eos_by_id:
                unique_source_eos = sorted(set(source_eos_by_id.values()))
                if len(unique_source_eos) > 1:
                    raise ValueError(
                        "Incompatible source eos_token_id values in dataset.sources: "
                        f"{source_eos_by_id}. Expected all eos_token_id values to match."
                    )
                return int(unique_source_eos[0])

            metadata_eos_by_id: dict[str, int] = {}
            for source_cfg in sources:
                dataset_id = str(source_cfg.get("dataset_id", "<missing-dataset-id>"))
                dataset_path = source_cfg.get("nameOrPath", None)
                if not dataset_path:
                    raise ValueError(f"Source {dataset_id!r} is missing nameOrPath.")
                eos = self._read_eos_from_dataset_metadata(
                    dataset_path=Path(str(dataset_path)),
                    dataset_label=f"source {dataset_id!r}",
                )
                if eos is None:
                    raise ValueError(
                        "Packing insert_eos is enabled but eos_token_id could not be resolved for "
                        f"{dataset_id!r}. Provide dataset.packing.eos_token_id, set "
                        f"dataset.sources[{dataset_id!r}].eos_token_id, or add "
                        f"{TOKENIZATION_METADATA_FILENAME} to {dataset_path!r}."
                    )
                metadata_eos_by_id[dataset_id] = eos

            unique_metadata_eos = sorted(set(metadata_eos_by_id.values()))
            if len(unique_metadata_eos) > 1:
                raise ValueError(
                    "Incompatible eos_token_id values found in source dataset metadata: "
                    f"{metadata_eos_by_id}. Expected all sources to share the same eos_token_id."
                )
            return int(unique_metadata_eos[0])

        dataset_cfg = getattr(self.config, "dataset", None)
        dataset_path_value = None
        if dataset_cfg is not None:
            dataset_path_value = (
                dataset_cfg.get("nameOrPath", None)
                if hasattr(dataset_cfg, "get")
                else getattr(dataset_cfg, "nameOrPath", None)
            )
        if dataset_path_value:
            eos_from_metadata = self._read_eos_from_dataset_metadata(
                dataset_path=Path(str(dataset_path_value)),
                dataset_label=f"dataset path {dataset_path_value!r}",
            )
            if eos_from_metadata is not None:
                return int(eos_from_metadata)

        raise ValueError(
            "Packing insert_eos is enabled but eos_token_id could not be resolved without network access. "
            "Provide dataset.packing.eos_token_id, set dataset.sources[*].eos_token_id, or add "
            f"{TOKENIZATION_METADATA_FILENAME} to the tokenized dataset path."
        )

    def _read_eos_from_dataset_metadata(
        self,
        *,
        dataset_path: Path,
        dataset_label: str,
    ) -> int | None:
        metadata = read_tokenization_metadata(dataset_path)
        if metadata is None:
            self.cli_logger.debug(
                "No tokenization metadata found for %s at %s",
                dataset_label,
                dataset_path,
            )
            return None
        return extract_eos_token_id(
            metadata,
            source_label=f"{dataset_label} metadata ({dataset_path / TOKENIZATION_METADATA_FILENAME})",
        )

    def _build_packing_dataloaders(self, fabric: L.Fabric) -> dict[str, DataLoader]:
        packing = self._get_packing_config()
        if not packing or not packing.get("enabled", False):
            raise RuntimeError("_build_packing_dataloaders called but packing is not enabled.")
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
            raise ValueError("Packing is enabled but packing.sequence_length is missing.")
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

        dataset_cfg = getattr(self.config, "dataset", None)
        dataset_path_value = None
        if dataset_cfg is not None:
            dataset_path_value = dataset_cfg.get("nameOrPath", None) if hasattr(dataset_cfg, "get") else getattr(dataset_cfg, "nameOrPath", None)
        if not dataset_path_value:
            raise ValueError("Packing is enabled but dataset.nameOrPath is missing.")
        dataset_path = Path(str(dataset_path_value))

        index_cache_dir_value = packing.get("index_cache_dir", None)
        index_cache_dir = Path(str(index_cache_dir_value)) if index_cache_dir_value else (dataset_path / ".packing_index")
        if self._clean_index_cache_on_start(packing):
            self._maybe_clean_index_cache_version_dir(
                fabric,
                index_cache_dir,
                packing=packing,
                trigger="start",
            )

        slurm_nnodes = os.getenv("SLURM_NNODES")
        if slurm_nnodes is not None:
            try:
                nnodes = int(slurm_nnodes)
            except ValueError:
                nnodes = 1
            if nnodes > 1:
                for path in (dataset_path, index_cache_dir):
                    if not path.is_absolute():
                        raise ValueError(
                            "Multi-node SLURM packing requires absolute dataset/index paths on a shared filesystem. "
                            f"Got non-absolute path: {path}"
                        )
                    if str(path).startswith("/tmp") or str(path).startswith("/dev/shm"):
                        raise ValueError(
                            "Multi-node SLURM packing requires a shared filesystem path. "
                            f"Got a node-local path: {path}"
                        )

        seed = self.config.get("seed", None)
        seed_value = int(seed) if seed is not None else 0

        indices: dict[str, PackingIndex] = {}
        build_failed_path = index_cache_dir / f"v{INDEX_VERSION}" / "BUILD_FAILED.json"
        if fabric.global_rank == 0:
            build_failed_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                build_failed_path.unlink(missing_ok=True)
            except Exception:
                pass

            active_split_name: str | None = None
            try:
                for split_name, split_dataset in self.datasets.items():
                    active_split_name = split_name
                    split_t0 = time.perf_counter()
                    self.cli_logger.info(
                        "Packing index start split=%s sequence_length=%s insert_eos=%s index_cache_dir=%s",
                        split_name,
                        sequence_length,
                        insert_eos,
                        index_cache_dir,
                    )
                    indices[split_name] = PackingIndex.load_or_build(
                        hf_split=split_dataset,
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
                        "Packing index ready split=%s blocks=%s elapsed_s=%.2f cache_hit=%s index_cache_dir=%s",
                        split_name,
                        int(indices[split_name].num_blocks),
                        time.perf_counter() - split_t0,
                        bool(getattr(indices[split_name], "cache_hit", False)),
                        index_cache_dir,
                    )
            except Exception as exc:
                payload = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at_unix": time.time(),
                    "pid": os.getpid(),
                    "host": os.getenv("SLURMD_NODENAME") or os.uname().nodename,
                }
                if isinstance(exc, TimeoutError):
                    payload["lock_context"] = self._collect_lock_failure_context(
                        index_cache_dir=index_cache_dir,
                        split_name=active_split_name,
                        lock_timeout_s=lock_timeout_s,
                        stale_lock_age_s=stale_lock_age_s,
                        lock_lease_heartbeat_s=lock_lease_heartbeat_s,
                    )
                try:
                    build_failed_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                except Exception:
                    pass

        fabric.barrier()

        if build_failed_path.exists():
            try:
                details = build_failed_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                details = "<unreadable>"
            raise RuntimeError(
                "Packing index build failed on rank 0. "
                f"See {build_failed_path} for details: {details}"
            )

        if fabric.global_rank != 0:
            for split_name, split_dataset in self.datasets.items():
                indices[split_name] = PackingIndex.load_or_build(
                    hf_split=split_dataset,
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

        dataloaders: dict[str, DataLoader] = {}
        for split_name, split_dataset in self.datasets.items():
            shuffle = packing.get("shuffle", None)
            if shuffle is None:
                shuffle = split_name == "train"
            shuffle = bool(shuffle)

            sampler_drop_last = packing.get("sampler_drop_last", None)
            if sampler_drop_last is None:
                sampler_drop_last = fabric.world_size > 1
            sampler_drop_last = bool(sampler_drop_last)

            drop_last_batch = packing.get("drop_last_batch", None)
            if drop_last_batch is None:
                drop_last_batch = split_name == "train" and fabric.world_size > 1
            drop_last_batch = bool(drop_last_batch)

            # Packing requires doc-level tokenization output with `input_ids` + `length`.
            packed = PackedSequenceDataset(
                hf_dataset=split_dataset,
                index=indices[split_name],
                eos_token_id=eos_token_id,
            )
            dataloaders[split_name] = build_packing_dataloader(
                dataset=packed,
                split=split_name,
                batch_size=int(self.config.batch_size),
                num_workers=int(self.config.num_workers),
                shuffle=shuffle,
                sampler_drop_last=sampler_drop_last,
                drop_last_batch=drop_last_batch,
                seed=seed_value,
                rank=int(fabric.global_rank),
                world_size=int(fabric.world_size),
            )

            if fabric.global_rank == 0:
                sampler = getattr(dataloaders[split_name], "sampler", None)
                self.cli_logger.info(
                    "Packing dataset split=%s blocks=%s sequence_length=%s insert_eos=%s eos_token_id=%s "
                    "shuffle=%s sampler=%s sampler_drop_last=%s drop_last_batch=%s "
                    "skipped_empty_docs=%s tail_tokens_dropped=%s index_cache_dir=%s",
                    split_name,
                    len(packed),
                    sequence_length,
                    insert_eos,
                    eos_token_id,
                    shuffle,
                    type(sampler).__name__ if sampler is not None else None,
                    sampler_drop_last,
                    drop_last_batch,
                    packed.stats.skipped_empty_docs,
                    packed.stats.tail_tokens_dropped,
                    index_cache_dir,
                )

        return dataloaders

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
            index_cache_dir = Path(str(index_cache_dir_value)) if index_cache_dir_value else (dataset_path / ".packing_index")
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
        self._mixture_requested_total_blocks = requested_total_blocks
        self._mixture_effective_total_blocks = effective_total_blocks
        self._mixture_schedule_seed = schedule_seed
        self._mixture_dataset_id_to_idx = dict(mixture_train_dataset.dataset_id_to_idx)
        self._mixture_dataset_idx_to_id = dict(mixture_train_dataset.dataset_idx_to_id)
        self._mixture_realized_blocks_local = {dataset_id: 0 for dataset_id in sorted(source_blocks)}
        self._mixture_realized_blocks_local_tensor = None
        self._mixture_configured_weights_by_id = configured_weights
        self._mixture_source_blocks_by_id = source_blocks

        report_path_raw = mixture.get("report_path", None)
        if report_path_raw:
            self._mixture_report_path = Path(str(report_path_raw))
        else:
            self._mixture_report_path = Path(self.config.output_dir) / "mixture_report.json"

        return {
            "train": train_loader,
            "valid_by_source": valid_by_source,
        }

    def _ensure_validation_split(
        self,
        dataset: Union[DatasetDict, HFDataset],
        *,
        split_seed_override: int | None = None,
        require_valid: bool = False,
        source_id: str | None = None,
    ) -> DatasetDict:
        """
        Ensure the dataset provides a 'valid' split for validation.

        If a 'valid' split already exists it is used as-is. If a 'validation' split
        is present, it is re-keyed to 'valid'. Otherwise, a new validation split is
        created from the training data when the configuration specifies
        `validation_split`.

        Args:
            dataset (Union[DatasetDict, HFDataset]): The dataset to inspect.

        Returns:
            DatasetDict: Dataset dictionary guaranteed to contain a 'train' split and,
                when configured, a 'valid' split.
        """

        source_label = f"source={source_id!r} " if source_id is not None else ""

        if isinstance(dataset, HFDataset):
            self.cli_logger.info("%sSingle dataset provided, wrapping as training data only", source_label)
            dataset = DatasetDict({"train": dataset})
        else:
            dataset = DatasetDict(dataset)

        if "valid" in dataset:
            return dataset

        if "validation" in dataset:
            self.cli_logger.info("%sFound 'validation' split; reusing it as 'valid'.", source_label)
            validation_dataset = dataset["validation"]
            del dataset["validation"]
            dataset["valid"] = validation_dataset
            return dataset

        split_config = getattr(self.config, "validation_split", None)
        if not split_config:
            if require_valid:
                raise ValueError(
                    f"{source_label}Missing validation data: source has no 'valid'/'validation' split "
                    "and no validation_split configuration was provided."
                )
            return dataset

        if isinstance(split_config, Box):
            split_config = split_config.to_dict()

        shuffle = bool(split_config.get("shuffle", True))
        seed = split_seed_override if split_seed_override is not None else split_config.get("seed", getattr(self.config, "seed", None))

        proportion = split_config.get("proportion")
        count = split_config.get("count")

        if proportion is None and count is None:
            raise ValueError(
                f"{source_label}validation_split configuration must include 'proportion' or 'count'."
            )

        train_dataset = dataset.get("train")
        if train_dataset is None:
            raise ValueError(f"{source_label}Training split 'train' is required to derive a validation split.")

        total_examples = len(train_dataset)
        if total_examples < 2:
            raise ValueError(
                f"{source_label}Not enough training examples to create a validation split (need at least 2)."
            )

        candidate_sizes: list[int] = []
        if proportion is not None:
            if not isinstance(proportion, (int, float)):
                raise TypeError(
                    f"{source_label}validation_split.proportion must be a numeric value between 0 and 1."
                )
            proportion_value = float(proportion)
            if not 0 < proportion_value < 1:
                raise ValueError(f"{source_label}validation_split.proportion must be between 0 and 1.")
            candidate_sizes.append(max(1, int(round(total_examples * proportion_value))))

        if count is not None:
            if not isinstance(count, int):
                raise TypeError(f"{source_label}validation_split.count must be an integer.")
            if count <= 0:
                raise ValueError(f"{source_label}validation_split.count must be greater than 0.")
            candidate_sizes.append(count)

        if not candidate_sizes:
            raise ValueError(f"{source_label}Unable to determine validation split size from configuration.")

        val_count = min(candidate_sizes)
        if val_count >= total_examples:
            adjusted_val_count = total_examples - 1
            if adjusted_val_count <= 0:
                raise ValueError(
                    f"{source_label}Requested validation size {val_count} is incompatible with training size {total_examples}."
                )
            self.cli_logger.warning(
                f"{source_label}Requested validation size {val_count} >= training size {total_examples}. "
                f"Reducing validation size to {adjusted_val_count}."
            )
            val_count = adjusted_val_count

        split = train_dataset.train_test_split(
            test_size=val_count,
            shuffle=shuffle,
            seed=seed,
        )

        dataset["train"] = split["train"]
        dataset["valid"] = split["test"]

        self.cli_logger.info(
            f"{source_label}Created validation split with {len(dataset['valid'])} examples "
            f"({len(dataset['valid']) / total_examples:.2%} of the original training data)."
        )

        return dataset
    
    def _save(self, fabric: L.Fabric, epochFinished: bool = False, trainingFinished: bool = False) -> None:
        """
        Save the training checkpoint.

        This method saves the current training state to the output directory if provided.
        The checkpoint name uses the format: epoch-<epoch number>-<global iteration number>
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the distributed training.
        - epochFinished (bool): Flag indicating if the current epoch has finished.
        - trainingFinished (bool): Flag indicating if training has completed.
        """
        if self.config.output_dir is None:
            self.cli_logger.warning("Output directory not provided. Skipping checkpoint saving.")
            return
        
        try:
            self.state["run_metadata"] = self._build_run_metadata(fabric)
            if self._mixture_enabled:
                self.state["mixture_meta"] = self._build_mixture_resume_meta(fabric)
                self.state["mixture_runtime"] = self._build_mixture_runtime_state()

            # Generate checkpoint name using consistent nomenclature: epoch-<epoch>-<global_iteration>
            current_epoch = self.state.get('current_epoch', 0)
            global_iteration = self.state.get('step_count', 0)
            checkpoint_name = f"e-{current_epoch:03d}-gs-{global_iteration:06d}.pth"
            
            output_checkpoint_path = Path(self.config.output_dir, checkpoint_name)

            self.cli_logger.info(f"Saving checkpoint to {output_checkpoint_path}")
            
            # Log checkpoint saving info
            progress_info = ""
            if epochFinished:
                progress_info = " (end of epoch)"
            elif trainingFinished:
                progress_info = " (training complete)"
            else:
                # Calculate progress within epoch for intra-epoch saves
                try:
                    train_dataset = self.datasets["train"]
                    batch_size = max(1, int(self.config.batch_size))
                    world_size = max(1, int(fabric.world_size))
                    gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
                    if gradient_accumulation_steps is None or int(gradient_accumulation_steps) <= 0:
                        gradient_accumulation_steps = 1
                    else:
                        gradient_accumulation_steps = int(gradient_accumulation_steps)

                    total_batches = math.ceil(len(train_dataset) / (batch_size * world_size))
                    steps_per_epoch = max(1, math.ceil(total_batches / gradient_accumulation_steps))
                    steps_in_previous_epochs = (current_epoch - 1) * steps_per_epoch
                    step_in_current_epoch = global_iteration - steps_in_previous_epochs
                    epoch_progress = (step_in_current_epoch / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0
                    progress_info = f" ({epoch_progress:.1f}% of epoch {current_epoch})"
                except Exception as e:
                    self.cli_logger.debug(f"Could not calculate epoch progress: {str(e)}")
                    progress_info = f" (step {global_iteration} in epoch {current_epoch})"
            
            self.cli_logger.info(f"Saving checkpoint to {checkpoint_name!r}{progress_info}")
            fabric.save(output_checkpoint_path, self.state)
            self.cli_logger.info(f"Checkpoint saved successfully to {str(output_checkpoint_path)}")
            
        except Exception as e:
            self.cli_logger.error(f"Failed to save checkpoint: {str(e)}")
            raise  # Re-raise to be handled by calling code
    
    def _get_resume_iterator(self, iterator: int, resume_iter: int) -> Tuple[int, int]:
        """
        Get the resumed iterator state for training.

        Parameters:
        - iterator (int): The current iterator over the dataset.
        - resume_iter (int): The iteration number from which to resume training.

        Returns:
        - tuple: A tuple containing the possibly sliced iterator and the updated resume_iter.
        """
        epoch_batch_count = len(iterator)
        if resume_iter >= epoch_batch_count:
            return None, resume_iter - epoch_batch_count        
        elif resume_iter > 0:
            return itertools.islice(iterator, resume_iter, None), 0
        else:
            return iterator, resume_iter

    def _resolved_gradient_accumulation_steps(self) -> int:
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        if grad_accum is None or int(grad_accum) <= 0:
            return 1
        return int(grad_accum)

    def _build_run_metadata(self, fabric: L.Fabric) -> dict[str, Any]:
        lr_raw = self.config.get("lr", None)
        warmup_raw = self.config.get("warmup_proportion", None)
        min_lr_raw = self.config.get("min_lr", None)
        max_lr_raw = self.config.get("max_lr", None)
        state_obj = getattr(self, "state", {})
        metadata: dict[str, Any] = {
            "run_metadata_version": RUN_METADATA_VERSION,
            "task": str(self.config.get("task", "")),
            "experiment_name": str(self.config.get("experiment_name", "")),
            "model_name": str(self.config.get("model_name", "")),
            "precision": str(self.config.get("precision", "")),
            "seed": int(self.config.get("seed", 0) or 0),
            "batch_size": int(self.config.batch_size),
            "gradient_accumulation_steps": self._resolved_gradient_accumulation_steps(),
            "number_epochs": int(self.config.number_epochs),
            "world_size": int(fabric.world_size),
            "num_nodes": int(self.num_nodes),
            "devices_per_node": (
                int(self.devices_per_node) if self.devices_per_node is not None else None
            ),
            "output_dir": str(self.config.output_dir),
            "mixture_enabled": bool(self._mixture_enabled),
            "lr": float(lr_raw) if lr_raw is not None else None,
            "lr_scheduler": str(self.config.get("lr_scheduler", "")),
            "warmup_proportion": float(warmup_raw) if warmup_raw is not None else None,
            "min_lr": float(min_lr_raw) if min_lr_raw is not None else 0.0,
            "max_lr": float(max_lr_raw) if max_lr_raw is not None else None,
            "iter_num": int(state_obj.get("iter_num", 0)) if isinstance(state_obj, dict) else 0,
            "step_count": int(state_obj.get("step_count", 0)) if isinstance(state_obj, dict) else 0,
        }
        schedule_meta = getattr(self, "_training_schedule_metadata", {})
        if isinstance(schedule_meta, dict) and schedule_meta:
            metadata["training_schedule"] = dict(schedule_meta)
        if self._mixture_enabled and self._mixture_plan is not None:
            metadata.update(
                {
                    "budget_mode": self._mixture_plan.budget_mode,
                    "anchor_dataset_id": self._mixture_plan.anchor_dataset_id,
                    "requested_total_blocks": int(self._mixture_requested_total_blocks or 0),
                    "effective_total_blocks": int(self._mixture_effective_total_blocks or 0),
                    "hash_algorithm": "blake2b_u64_v1",
                    "allocation_algorithm": "hamilton_lr_lexicographic_v1",
                    "split_seed_algorithm": "blake2b_u64_mod_2147483647_v1",
                }
            )
        return metadata

    def _ensure_mixture_realized_counter_tensor(self, device: torch.device) -> torch.Tensor:
        size = len(self._mixture_dataset_idx_to_id)
        if size <= 0:
            raise RuntimeError("Mixture source mapping is not initialized; cannot build realized-block counters.")

        local_tensor = self._mixture_realized_blocks_local_tensor
        if (
            local_tensor is None
            or local_tensor.numel() != size
            or local_tensor.device != device
        ):
            rebuilt = torch.zeros(size, dtype=torch.long, device=device)
            for dataset_idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items()):
                idx = int(dataset_idx)
                if 0 <= idx < size:
                    rebuilt[idx] = int(self._mixture_realized_blocks_local.get(dataset_id, 0))
            self._mixture_realized_blocks_local_tensor = rebuilt

        return self._mixture_realized_blocks_local_tensor

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

    def _build_mixture_runtime_state(self) -> dict[str, Any]:
        if not self._mixture_enabled:
            return {}
        self._sync_mixture_realized_blocks_local_from_tensor()
        return {
            "mixture_runtime_version": MIXTURE_RUNTIME_VERSION,
            "realized_blocks_local": {
                dataset_id: int(v)
                for dataset_id, v in sorted(self._mixture_realized_blocks_local.items())
            },
            "last_val_losses": {
                dataset_id: float(v)
                for dataset_id, v in sorted(self._mixture_last_val_losses.items())
            },
            "last_val_weighted": (
                float(self._mixture_last_val_weighted)
                if self._mixture_last_val_weighted is not None
                else None
            ),
        }

    def _restore_mixture_runtime_state(self, runtime_state: dict[str, Any] | None, *, strict: bool) -> None:
        if not self._mixture_enabled:
            return
        if not runtime_state:
            if strict:
                raise ValueError(
                    "Mixture resume requires checkpoint key 'mixture_runtime' for mixture_meta_version='v2'."
                )
            self.cli_logger.warning(
                "Mixture resume checkpoint is missing 'mixture_runtime'; "
                "continuing with zeroed local counters for this rank."
            )
            return
        if not isinstance(runtime_state, dict):
            if strict:
                raise ValueError(
                    "Mixture resume requires 'mixture_runtime' to be a dictionary for mixture_meta_version='v2'."
                )
            self.cli_logger.warning(
                "Mixture resume checkpoint has invalid 'mixture_runtime' type (%s); "
                "continuing with zeroed local counters for this rank.",
                type(runtime_state).__name__,
            )
            return

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
                    restored[dataset_id] = max(0, int(raw))
                except (TypeError, ValueError):
                    if strict:
                        raise ValueError(
                            f"Mixture resume has non-integer realized_blocks_local value for dataset_id={dataset_id!r}: {raw!r}."
                        )
                    restored[dataset_id] = 0
            self._mixture_realized_blocks_local = restored
            self._mixture_realized_blocks_local_tensor = None

        loaded_losses = runtime_state.get("last_val_losses", {})
        if strict and not isinstance(loaded_losses, dict):
            raise ValueError("Mixture resume requires 'mixture_runtime.last_val_losses' to be a dictionary.")
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

        # Backward-compatible mode for checkpoints created before metadata v2.
        if loaded_version is None:
            legacy_keys = [
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
            ]
            loaded_legacy = {k: loaded_meta.get(k) for k in legacy_keys}
            expected_legacy = {k: expected_meta.get(k) for k in legacy_keys}
            if loaded_legacy != expected_legacy:
                raise ValueError(
                    "Mixture resume compatibility check failed for legacy checkpoint metadata.\n"
                    f"Expected (legacy keys): {json.dumps(expected_legacy, sort_keys=True)}\n"
                    f"Loaded: {json.dumps(loaded_legacy, sort_keys=True)}"
                )
            self.cli_logger.warning(
                "Loaded legacy mixture metadata without version tag; "
                "continuing with compatibility checks on the legacy field subset only."
            )
            return

        if loaded_version != expected_version:
            raise ValueError(
                "Mixture resume compatibility check failed: mixture_meta_version mismatch "
                f"(expected={expected_version!r}, loaded={loaded_version!r})."
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

    def _write_mixture_report(self, fabric: L.Fabric) -> None:
        if not self._mixture_enabled or self._mixture_plan is None:
            return

        resolved_blocks = self._reduce_mixture_realized_blocks(fabric)
        if fabric.global_rank != 0:
            return

        report_path = self._mixture_report_path or (Path(self.config.output_dir) / "mixture_report.json")
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
        realized_ratios = {
            dataset_id: (float(blocks) / float(effective_total_blocks) if effective_total_blocks > 0 else 0.0)
            for dataset_id, blocks in resolved_blocks.items()
        }
        target_ratios = {
            dataset_id: (float(blocks) / float(effective_total_blocks) if effective_total_blocks > 0 else 0.0)
            for dataset_id, blocks in target_blocks.items()
        }
        deviation_from_target_blocks = {
            dataset_id: int(resolved_blocks.get(dataset_id, 0)) - int(target_blocks.get(dataset_id, 0))
            for dataset_id in sorted(target_blocks)
        }

        grad_accum = self._resolved_gradient_accumulation_steps()

        source_entries = []
        for dataset_id in sorted(self._mixture_plan.effective_weights_by_id):
            source_cfg = self._mixture_source_configs.get(dataset_id, {})
            train_split = self.datasets.get(dataset_id, {}).get("train") if isinstance(self.datasets, dict) else None
            dataset_fingerprint = getattr(train_split, "_fingerprint", None) if train_split is not None else None
            source_entries.append(
                {
                    "dataset_id": dataset_id,
                    "nameOrPath": source_cfg.get("nameOrPath", None) if hasattr(source_cfg, "get") else None,
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
            "world_size": int(fabric.world_size),
            "batch_size": int(self.config.batch_size),
            "gradient_accumulation_steps": grad_accum,
            "dataset_index_map": {str(idx): dataset_id for idx, dataset_id in sorted(self._mixture_dataset_idx_to_id.items())},
            "sources": source_entries,
            "target_blocks_per_dataset": target_blocks,
            "realized_blocks_per_dataset": resolved_blocks,
            "realized_tokens_per_dataset": realized_tokens,
            "realized_ratios": realized_ratios,
            "deviation_from_target_blocks": deviation_from_target_blocks,
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
        }
        if self._mixture_last_val_weighted is not None:
            summary_metrics["mixture/val_loss_weighted"] = float(self._mixture_last_val_weighted)
        for dataset_id in sorted(target_blocks):
            summary_metrics[f"mixture/target_ratio_{dataset_id}"] = float(target_ratios.get(dataset_id, 0.0))
            summary_metrics[f"mixture/realized_ratio_{dataset_id}"] = float(realized_ratios.get(dataset_id, 0.0))
            summary_metrics[f"mixture/deviation_blocks_{dataset_id}"] = float(
                deviation_from_target_blocks.get(dataset_id, 0)
            )
        fabric.log_dict(summary_metrics, int(self.state.get("step_count", 0)))
        self.cli_logger.info("Wrote mixture report to %s", report_path)
    
    def _load_from_checkpoint(self, fabric: L.Fabric, expected_mixture_meta: dict[str, Any] | None = None) -> None:
        """
        Load model and optimizer state from a checkpoint if a checkpoint path is provided.

        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the training.
        """
        if self.checkpoint_path is not None:
            self.cli_logger.info(f"Resuming training from '{self.checkpoint_path}'")
            if self._mixture_enabled:
                self.state.pop("mixture_meta", None)
                self.state.pop("mixture_runtime", None)
            # Use strict=False to allow loading checkpoints that may not have all current state keys
            fabric.load(self.checkpoint_path, self.state, strict=False)
            
            # Ensure that essential keys have default values if they weren't in the checkpoint
            if 'current_epoch' not in self.state:
                self.state['current_epoch'] = 0
                self.cli_logger.info("'current_epoch' not found in checkpoint, defaulting to 0")
            
            if 'iter_num' not in self.state:
                self.state['iter_num'] = 0
                self.cli_logger.info("'iter_num' not found in checkpoint, defaulting to 0")
                
            if 'step_count' not in self.state:
                self.state['step_count'] = 0
                self.cli_logger.info("'step_count' not found in checkpoint, defaulting to 0")

            if self._mixture_enabled:
                if not expected_mixture_meta:
                    raise ValueError("Mixture resume requires expected mixture metadata but none was provided.")
                self._validate_mixture_resume_compatibility(expected_mixture_meta)
                loaded_meta = self.state.get("mixture_meta", {})
                strict_runtime = isinstance(loaded_meta, dict) and loaded_meta.get("mixture_meta_version") == MIXTURE_META_VERSION
                self._restore_mixture_runtime_state(
                    self.state.get("mixture_runtime"),
                    strict=bool(strict_runtime),
                )

            if "run_metadata" not in self.state:
                self.state["run_metadata"] = self._build_run_metadata(fabric)
                self.cli_logger.warning(
                    "Checkpoint is missing 'run_metadata'; generated run metadata from current configuration."
                )
            
            # Log the loaded state for debugging
            self.cli_logger.info(f"Loaded state keys: {list(self.state.keys())}")
            self.cli_logger.info(f"Resuming from iteration {self.state.get('iter_num', 0)}, step {self.state.get('step_count', 0)}, epoch {self.state.get('current_epoch', 0)}")
    
    def _load_initial_weights(self, fabric: L.Fabric) -> None:
        """
        Load only model weights from a checkpoint for transfer learning (no training state).
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the training.
        """
        initial_weights_checkpoint = getattr(self.config, 'initial_weights_checkpoint', None)
        if initial_weights_checkpoint is not None:
            self.cli_logger.info(f"Loading initial model weights from '{initial_weights_checkpoint}' for transfer learning")
            
            # Load the checkpoint using torch.load since we only need model weights
            checkpoint = torch.load(initial_weights_checkpoint, map_location="cpu")
            
            # Extract only the model state dict
            model_state_dict = None
            if isinstance(checkpoint, dict):
                # Try different keys where model weights might be stored
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                else:
                    # Assume the checkpoint is a direct state dict
                    model_state_dict = checkpoint
            else:
                model_state_dict = checkpoint
            
            if model_state_dict is None:
                self.cli_logger.error("Could not find model weights in checkpoint")
                return
            
            # Load only the model weights, not the training state
            if hasattr(self.state, 'model') and self.state['model'] is not None:
                try:
                    missing_keys, unexpected_keys = self.state['model'].load_state_dict(model_state_dict, strict=False)
                    if missing_keys:
                        self.cli_logger.warning(f"Missing keys when loading initial weights: {len(missing_keys)} keys")
                        self.cli_logger.debug(f"Sample missing keys: {missing_keys[:5]}")
                    if unexpected_keys:
                        self.cli_logger.warning(f"Unexpected keys when loading initial weights: {len(unexpected_keys)} keys")
                        self.cli_logger.debug(f"Sample unexpected keys: {unexpected_keys[:5]}")
                    self.cli_logger.info("✅ Successfully loaded initial model weights for transfer learning")
                except Exception as e:
                    self.cli_logger.error(f"Failed to load initial weights: {str(e)}")
                    raise
            else:
                self.cli_logger.warning("Model not yet initialized, cannot load initial weights")
    def _train_logs(self, fabric: L.Fabric, loss: torch.Tensor) -> None:
        """
        Log training metrics for monitoring.
        """

        self.cli_logger.debug(
            f"iter {self.state['iter_num']} step {self.state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(self.train_t1 - self.train_iter_t0) * 1000:.2f}ms remaining time: "
            # f"{(self.train_t1 - self.train_total_t0) / (self.state['iter_num'] - self.initial_iter) * (self.config.max_iters - self.state['iter_num']) / 3600:.2f} hours. "
        )
        self.monitor.on_train_batch_end(
            self.state["iter_num"] * self.config.batch_size,
            self.train_t1 - self.train_total_t0,
            fabric.world_size,
            self.state["step_count"],
            lengths=self.total_lengths,
            train_loss=loss.item()
        )

    def _log_learning_rates(self, fabric: L.Fabric) -> None:
        """Log current learning rates for each optimizer param group."""
        optimizer = self.state.get("optimizer")
        if optimizer is None:
            return

        lr_metrics = {}
        primary_lr = None
        for idx, group in enumerate(optimizer.param_groups):
            lr = group.get("lr")
            if lr is None:
                continue
            lr_value = float(lr)
            lr_metrics[f"lr/group_{idx}"] = lr_value
            if primary_lr is None:
                primary_lr = lr_value

        if not lr_metrics:
            return

        if primary_lr is not None:
            lr_metrics.setdefault("lr", primary_lr)

        fabric.log_dict(lr_metrics, self.state["step_count"])
    
    def _gradient_clipping(self, fabric: L.Fabric, model: L.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """
        Clip model gradients to avoid exploding gradients.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - optimizer (torch.optim.Optimizer): The optimizer used for training.
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.cli_logger.debug(f"Gradient norm before clipping: {grad_norm:.4f}")
        fabric.clip_gradients(model, optimizer, max_norm=self.config.grad_clip)
    
    def _accumulate_training(self, fabric: L.Fabric, model: L.LightningModule, batch: Tuple[torch.Tensor, ...], step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform training step with gradient accumulation.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - batch (tuple): A batch of training data.
        - step (int): The current training step.

        Returns:
        - tuple: Contains the outputs from the training step and the loss tensor.
        """
        # Debug logging for gradient accumulation steps
        gradient_accumulation_steps_raw = self.config.gradient_accumulation_steps
        self.cli_logger.debug(f"gradient_accumulation_steps raw value: {gradient_accumulation_steps_raw}, type: {type(gradient_accumulation_steps_raw)}")
        
        gradient_accumulation_steps = int(self.config.gradient_accumulation_steps)
        self.cli_logger.debug(f"gradient_accumulation_steps converted: {gradient_accumulation_steps}, type: {type(gradient_accumulation_steps)}")
        self.cli_logger.debug(f"iter_num value: {self.state['iter_num']}, type: {type(self.state['iter_num'])}")
        self.cli_logger.debug(f"About to check: (iter_num + 1) % gradient_accumulation_steps != 0 -> ({self.state['iter_num']} + 1) % {gradient_accumulation_steps} != 0")
        
        is_accumulating = (self.state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            real_loss = (loss / gradient_accumulation_steps) if is_accumulating else loss
            fabric.backward(real_loss)
        if not is_accumulating:
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            self._gradient_clipping(fabric, model, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.state["step_count"] += 1
            self._log_learning_rates(fabric)
            self._try_validate(fabric)
        self.state["iter_num"] += 1
        return outputs, loss
    
    def _try_validate(self, fabric: L.Fabric, epochFinished: bool = False, trainingFinished: bool = False) -> None:
        """
        Determine whether to run validation based on configured conditions and perform validation/checkpointing if necessary.

        This method implements a training-first approach that:
        1. Validates according to per-epoch scheduling and explicit end-of-epoch/end-of-training flags
        2. Saves checkpoints based on dedicated cadence settings (including checkpoints_per_epoch)
        3. Runs validation only when validation data is available
        4. Handles training-only datasets gracefully (common in pre-training scenarios)

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - epochFinished (bool): Flag indicating if the current epoch has finished.
        - trainingFinished (bool): Flag indicating if training has completed.
        """
        validations_per_epoch = self.config.get("validations_per_epoch", 1)
        checkpoints_per_epoch = self.config.get("checkpoints_per_epoch", None)
        validate_after_epoch = self.config.get("validate_after_epoch", True)
        validate_on_end = self.config.get("validate_on_end", True)
        save_on_validate = self.config.get("save_on_validate", False)
        save_on_end = self.config.get("save_on_end", False)
        validate_after_k_steps = self.config.get("validate_after_k_steps", None)
        total_steps_completed = self.state.get("step_count", 0)

        try:
            validations_per_epoch = max(1, int(validations_per_epoch))
        except (TypeError, ValueError):
            self.cli_logger.warning(f"Invalid validations_per_epoch value {validations_per_epoch!r}; defaulting to 1")
            validations_per_epoch = 1

        parsed_checkpoints = None
        if checkpoints_per_epoch is not None:
            try:
                parsed_checkpoints = max(1, int(checkpoints_per_epoch))
            except (TypeError, ValueError):
                self.cli_logger.warning(f"Invalid checkpoints_per_epoch value {checkpoints_per_epoch!r}; ignoring setting")
                parsed_checkpoints = None
        checkpoints_per_epoch = parsed_checkpoints

        step_interval = None
        if validate_after_k_steps is not None:
            try:
                step_interval = int(validate_after_k_steps)
                if step_interval <= 0:
                    self.cli_logger.warning(
                        f"validate_after_k_steps must be a positive integer, got {validate_after_k_steps!r}; disabling setting"
                    )
                    step_interval = None
            except (TypeError, ValueError):
                self.cli_logger.warning(
                    f"Invalid validate_after_k_steps value {validate_after_k_steps!r}; expected integer"
                )
                step_interval = None

        should_validate = False
        should_save = False
        validation_steps = []
        checkpoint_steps = []
        step_in_current_epoch = None
        steps_per_epoch = None

        # Validation/save logic for end of training
        if trainingFinished:
            should_validate = bool(validate_on_end)
            should_save = bool(save_on_end or (save_on_validate and should_validate))

        # Validation/save logic for end of epoch
        elif epochFinished:
            should_validate = bool(validate_after_epoch)
            if should_validate and save_on_validate:
                should_save = True

        # Validation/save logic during epoch based on step count
        else:
            # Safety check: ensure we have the required data structures
            if not hasattr(self, "dataloaders") or "train" not in self.dataloaders:
                self.cli_logger.warning("Cannot perform intra-epoch validation/checkpoint scheduling: missing dataset or dataloader structure")
                return

            current_epoch = self.state.get("current_epoch", 1)
            batch_size = max(1, int(self.config.batch_size))
            world_size = max(1, int(fabric.world_size))
            gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

            if gradient_accumulation_steps is None or int(gradient_accumulation_steps) <= 0:
                gradient_accumulation_steps = 1
            else:
                gradient_accumulation_steps = int(gradient_accumulation_steps)

            if self._packing_enabled:
                train_dataloader = self.dataloaders.get("train")
                if train_dataloader is None:
                    self.cli_logger.warning(
                        "Cannot schedule intra-epoch validation/checkpoints: missing train dataloader in packing mode"
                    )
                    return
                train_num_batches = len(train_dataloader)
                steps_per_epoch = max(1, train_num_batches // gradient_accumulation_steps)
            else:
                if not hasattr(self, "datasets") or "train" not in self.datasets:
                    self.cli_logger.warning(
                        "Cannot schedule intra-epoch validation/checkpoints: missing train dataset in non-packing mode"
                    )
                    return
                train_dataset = self.datasets["train"]
                total_batches = math.ceil(len(train_dataset) / (batch_size * world_size))
                steps_per_epoch = max(1, math.ceil(total_batches / gradient_accumulation_steps))

            def _build_epoch_schedule(events_per_epoch: int, total_steps: int) -> list[int]:
                schedule = []
                for i in range(1, events_per_epoch + 1):
                    step_in_epoch = max(1, int((i / events_per_epoch) * total_steps))
                    schedule.append(step_in_epoch)
                return sorted(set(schedule))

            validation_steps = _build_epoch_schedule(validations_per_epoch, steps_per_epoch)

            checkpoint_steps = []
            if checkpoints_per_epoch is not None:
                checkpoint_steps = _build_epoch_schedule(checkpoints_per_epoch, steps_per_epoch)

            steps_in_previous_epochs = (current_epoch - 1) * steps_per_epoch
            step_in_current_epoch = total_steps_completed - steps_in_previous_epochs

            if step_in_current_epoch in validation_steps:
                should_validate = True
                self.cli_logger.debug(
                    f"Validation triggered at optimizer step {step_in_current_epoch}/{steps_per_epoch} in epoch {current_epoch} "
                    f"(validations_per_epoch={validations_per_epoch}, checkpoints={validation_steps})"
                )

            if checkpoints_per_epoch is not None and step_in_current_epoch in checkpoint_steps:
                should_save = True
                self.cli_logger.debug(
                    f"Checkpoint scheduled at optimizer step {step_in_current_epoch}/{steps_per_epoch} in epoch {current_epoch} "
                    f"(checkpoints_per_epoch={checkpoints_per_epoch}, schedule={checkpoint_steps})"
                )

            if should_validate and save_on_validate:
                should_save = True

        if step_interval is not None and total_steps_completed > 0:
            if total_steps_completed % step_interval == 0:
                should_validate = True
                if save_on_validate:
                    should_save = True
                self.cli_logger.debug(
                    f"Validation triggered at global optimizer step {total_steps_completed} "
                    f"(validate_after_k_steps={step_interval})"
                )

        validation_completed = False
        attempted_validation = should_validate

        if should_validate:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fabric.barrier()

            if self._mixture_enabled or "valid" in getattr(self, "dataloaders", {}):
                try:
                    self._validate(fabric)
                    validation_completed = True
                    self.cli_logger.debug("Validation completed successfully")
                except Exception as e:
                    self.cli_logger.warning(f"Validation failed: {str(e)}")
            else:
                self.cli_logger.debug("No validation data available, skipping validation step")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if should_save:
            if not should_validate:
                fabric.barrier()
            try:
                self._save(fabric, epochFinished, trainingFinished)
                if validation_completed:
                    validation_status = "with validation"
                elif attempted_validation:
                    validation_status = "with failed validation"
                else:
                    validation_status = "without validation"
                self.cli_logger.info(f"Checkpoint saved successfully ({validation_status})")
            except Exception as e:
                self.cli_logger.error(f"Failed to save checkpoint: {str(e)}")
                # Don't raise here to avoid stopping training for checkpoint save failures
    
    def _normal_training(self, fabric: L.Fabric, model: L.LightningModule, batch: Tuple[torch.Tensor, ...], step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a standard training step without gradient accumulation.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - batch (tuple): A batch of training data.
        - step (int): The current training step.

        Returns:
        - tuple: Contains the outputs from the training step and the loss tensor.
        """      
        
        with self.autocast_context():
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            gradient_accumulation_steps = int(self.config.gradient_accumulation_steps)
            fabric.backward(loss / gradient_accumulation_steps)
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            self._gradient_clipping(fabric, model, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.state["step_count"] += 1

            self._log_learning_rates(fabric)
            self._try_validate(fabric)
            self.state["iter_num"] += 1
            return outputs, loss
            
    def _train(self, fabric: L.Fabric) -> None:
        """
        Execute the main training loop over the specified number of epochs.

        This method iterates over the training dataset, handling both gradient accumulation and normal training,
        resuming from checkpoints when applicable, and logging training progress as well as performing validation.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance driving the training.
        """
        model = self.state["model"]
        self.total_lengths = 0
        self.train_total_t0 = time.perf_counter()
        self.initial_iter = self.state["iter_num"]
        epochs = self.config.number_epochs
        self.model.train()
        resume_iter = self.state["iter_num"]
        
        # Ensure current_epoch is initialized (for backwards compatibility with old checkpoints)
        if "current_epoch" not in self.state:
            self.state["current_epoch"] = 0
        
        for epoch in range(epochs):
            # Update current epoch in state for checkpoint naming
            self.state["current_epoch"] = epoch + 1

            sampler = getattr(self.dataloaders.get("train"), "sampler", None)
            if sampler is None:
                batch_sampler = getattr(self.dataloaders.get("train"), "batch_sampler", None)
                sampler = getattr(batch_sampler, "sampler", None) if batch_sampler is not None else None
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            
            if fabric.global_rank == 0:
                self.cli_logger.debug(f"Running Epoch {epoch + 1} of {epochs}")
            batch_iterator = tqdm(self.dataloaders['train'], mininterval=0, colour="blue") \
                if fabric.global_rank == 0 else self.dataloaders['train']
            batch_iterator, resume_iter = self._get_resume_iterator(batch_iterator, resume_iter)
            if batch_iterator is None:
                continue            
            for step, batch in enumerate(batch_iterator):
                self.train_iter_t0 = time.perf_counter()
                if (
                    fabric.global_rank == 0
                    and epoch == 0
                    and step == 0
                    and int(self.config.get("verbose_level", 0)) >= 4
                ):
                    shapes = {
                        k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")
                    }
                    self.cli_logger.debug("First batch shapes: %s", shapes)
                if self.config.gradient_accumulation_steps:
                    _, loss = self._accumulate_training(fabric, model, batch, step)
                else:
                    _, loss = self._normal_training(fabric, model, batch, step)
                self.total_lengths += batch["input_ids"].size(1)
                if self._mixture_enabled and "dataset_idx" in batch:
                    dataset_idx_tensor = batch["dataset_idx"].reshape(-1).to(torch.long)
                    local_counter = self._ensure_mixture_realized_counter_tensor(dataset_idx_tensor.device)
                    local_counts = torch.bincount(
                        dataset_idx_tensor,
                        minlength=local_counter.numel(),
                    )
                    local_counter.add_(local_counts)
                self.train_t1 = time.perf_counter()
                self._train_logs(fabric, loss)
                
            self._try_validate(fabric, epochFinished=True)
        self._try_validate(fabric, trainingFinished=True)
    
    @torch.no_grad()
    def _validate_mixture(self, fabric: L.Fabric) -> None:
        if not self._mixture_val_dataloaders:
            raise RuntimeError("Mixture validation called but no per-source validation dataloaders are available.")
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

                iterator = tqdm(
                    dataloader,
                    desc=f"Validating {dataset_id}...",
                    mininterval=0,
                    colour="green",
                ) if fabric.global_rank == 0 else dataloader

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
                fabric.log_dict({"metric/val_ppl_weighted": math.exp(val_loss_weighted)}, self.state["step_count"])

            self._mixture_last_val_losses = {k: float(v) for k, v in losses_by_source.items()}
            self._mixture_last_val_weighted = float(val_loss_weighted)

            elapsed_time = (time.perf_counter() - t0) * 1000.0
            self.cli_logger.info(
                "step %s: val_loss_weighted %.4f, per-source=%s, val time: %.2fms",
                self.state.get("iter_num", 0),
                val_loss_weighted,
                self._mixture_last_val_losses,
                elapsed_time,
            )
            fabric.barrier()
        finally:
            self.model.train()

    @torch.no_grad()
    def _validate(self, fabric: L.Fabric) -> None:
        """
        Validate the model on the validation dataset.

        This method switches the model to evaluation mode, processes the validation data, computes
        the mean loss, logs the validation metrics, and synchronizes across processes.

        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        
        Raises:
        - RuntimeError: If validation fails due to data or model issues.
        """       
        
        if self._mixture_enabled:
            self._validate_mixture(fabric)
            return

        if 'valid' not in self.dataloaders:
            raise RuntimeError("Validation called but no validation dataloader available")
        
        t0 = time.perf_counter()
        self.model.eval()
        losses = []
        
        try:
            batch_iterator = tqdm(
                self.dataloaders['valid'],
                desc="Validating...",
                mininterval=0,
                colour="green"
            ) if fabric.global_rank == 0 else self.dataloaders['valid']
            
            for k, val_data in enumerate(batch_iterator):
                validation_output = self.model.validation_step(val_data, k)
                loss = validation_output["loss"]
                losses.append(loss.detach())
                
        except Exception as e:
            self.cli_logger.error(f"Error during validation at batch {k}: {str(e)}")
            raise RuntimeError(f"Validation failed: {str(e)}") from e
        finally:
            # Ensure model is back in training mode
            self.model.train()
            
        if not losses:
            self.cli_logger.warning("No validation batches processed")
            return
            
        out = torch.mean(torch.stack(losses))
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        self.monitor.eval_end(t1)
        
        def fabric_eval_log(loss):
            self.cli_logger.info(f"step {self.state['iter_num']}: val loss {loss:.4f}, val time: {elapsed_time * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": loss.item()}, self.state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(loss.item())}, self.state["step_count"])
        
        fabric_eval_log(out)
        fabric.barrier()
      
    def _load_fabric_datasets_dataloaders(
        self,
        config: Box,
        dataset: Union[HFDataset, DatasetDict, dict[str, Union[HFDataset, DatasetDict]]],
    ) -> dict[str, Any]:
        """
        Load datasets and create dataloaders from the given dataset and configuration.

        This method validates the dataset, sets the required format, and creates DataLoader objects for each split.
        If train_data_ratio is specified in config and less than 1.0, only that proportion of the training data will be used.

        Parameters:
        - config (Box): Configuration parameters including batch_size, num_workers, and optionally train_data_ratio.
        - dataset (Union[HFDataset, DatasetDict]): The dataset or dictionary of datasets to use.

        Returns:
        - dict: A dictionary containing the processed datasets and corresponding dataloaders.

        Raises:
        - TypeError: If the dataset is not a DatasetDict or HFDataset.
        - ValueError: If required config parameters or dataset splits/columns are missing, or if train_data_ratio results in empty training set.
        - RuntimeError: If setting the format or creating a DataLoader fails.
        """
        mixture_cfg = self._get_mixture_config()
        sources_cfg = self._get_sources_config()
        mixture_requested = bool(mixture_cfg and mixture_cfg.get("enabled", False))
        if mixture_requested and not sources_cfg:
            raise ValueError("dataset.mixture.enabled is true but dataset.sources is missing or empty.")
        self._mixture_enabled = bool(mixture_requested and sources_cfg)

        if self._mixture_enabled:
            if config.get("task", None) != "clm_training":
                raise ValueError("dataset.mixture is enabled but is only supported for task='clm_training'.")
            if not self._is_packing_enabled():
                raise ValueError("dataset.mixture is enabled but dataset.packing.enabled is false.")
            if int(config.number_epochs) != 1:
                raise ValueError("Mixture mode requires number_epochs == 1.")
            if not isinstance(dataset, dict):
                raise TypeError("Mixture mode expects a mapping dataset_id -> DatasetDict/Dataset from the orchestrator.")
            if not hasattr(config, "num_workers") or not isinstance(config.num_workers, int) or config.num_workers < 0:
                raise ValueError("config.num_workers must be a non-negative integer")
            if not hasattr(config, "batch_size") or not isinstance(config.batch_size, int) or config.batch_size <= 0:
                raise ValueError("config.batch_size must be a positive integer")

            source_config_map = self._collect_source_config_map()
            self._validate_mixture_source_compatibility(source_config_map)
            self._mixture_source_configs = source_config_map

            dataset_ids = sorted(source_config_map)
            if set(dataset.keys()) != set(dataset_ids):
                raise ValueError(
                    "Loaded source datasets do not match dataset.sources entries. "
                    f"Expected {dataset_ids}, got {sorted(dataset.keys())}."
                )

            train_data_ratio = getattr(config, "train_data_ratio", 1.0)
            if train_data_ratio < 1.0:
                raise ValueError("train_data_ratio is not supported in mixture mode.")

            required_columns = ["input_ids", "length"]
            processed_sources: dict[str, DatasetDict] = {}
            self._mixture_source_split_metadata = {}

            for dataset_id in dataset_ids:
                source_dataset = dataset[dataset_id]
                if not isinstance(source_dataset, (DatasetDict, HFDataset)):
                    raise TypeError(
                        f"Source dataset {dataset_id!r} must be a DatasetDict or Dataset, got {type(source_dataset)}."
                    )

                had_existing_valid = False
                if isinstance(source_dataset, DatasetDict):
                    had_existing_valid = "valid" in source_dataset or "validation" in source_dataset

                split_seed_i = self._derive_validation_split_seed(dataset_id)
                source_dataset = self._ensure_validation_split(
                    source_dataset,
                    split_seed_override=split_seed_i,
                    require_valid=True,
                    source_id=dataset_id,
                )

                for split_name in ("train", "valid"):
                    if split_name not in source_dataset:
                        raise ValueError(
                            f"Source {dataset_id!r} is missing required split {split_name!r} in mixture mode."
                        )
                    missing_columns = [
                        col for col in required_columns if col not in source_dataset[split_name].column_names
                    ]
                    if missing_columns:
                        raise ValueError(
                            f"Source {dataset_id!r} split {split_name!r} is missing required columns "
                            f"{missing_columns} for packing mode."
                        )

                processed_sources[dataset_id] = source_dataset
                if had_existing_valid:
                    self._mixture_source_split_metadata[dataset_id] = {
                        "mode": "existing_valid",
                    }
                else:
                    self._mixture_source_split_metadata[dataset_id] = {
                        "mode": "auto_generated_valid",
                        "seed": split_seed_i,
                    }

            self._packing_enabled = True
            return {
                "datasets": processed_sources,
                "dataloaders": {},
            }

        if not isinstance(dataset, (DatasetDict, HFDataset)):
            raise TypeError("Expected dataset to be a DatasetDict or Dataset")
        if not hasattr(config, 'batch_size') or not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise ValueError("config.batch_size must be a positive integer")
        if not hasattr(config, 'num_workers') or not isinstance(config.num_workers, int) or config.num_workers < 0:
            raise ValueError("config.num_workers must be a non-negative integer")
        
        dataset = self._ensure_validation_split(dataset)
        packing_enabled = self._is_packing_enabled()
        self._packing_enabled = packing_enabled

        if packing_enabled and config.get("task", None) != "clm_training":
            raise ValueError(
                "dataset.packing is enabled but is only supported for task='clm_training'."
            )

        if not dataset.keys():
            raise ValueError("Dataset is empty, no splits found")
        
        # Log available splits for transparency
        available_splits = list(dataset.keys())
        self.cli_logger.info(f"Available dataset splits: {available_splits}")
        
        if 'valid' not in available_splits:
            self.cli_logger.info("No validation split available. Training will proceed without validation steps.")
        
        # Apply train_data_ratio if specified and less than 1.0
        train_data_ratio = getattr(config, 'train_data_ratio', 1.0)
        if train_data_ratio < 1.0 and 'train' in dataset:
            original_size = len(dataset['train'])
            subset_size = int(original_size * train_data_ratio)
            if subset_size > 0:
                # Use select method to get a subset of the training data
                dataset['train'] = dataset['train'].select(range(subset_size))
                self.cli_logger.info(f"Using {subset_size}/{original_size} ({train_data_ratio:.2%}) of training data")
            else:
                raise ValueError(f"train_data_ratio {train_data_ratio} results in empty training set")
                
        required_columns = (
            ["input_ids", "length"] if packing_enabled else ["input_ids", "attention_mask", "labels"]
        )
        for split in dataset.keys():
            missing_columns = [col for col in required_columns if col not in dataset[split].column_names]
            if missing_columns:
                raise ValueError(f"Missing required columns {missing_columns} in {split} split")

        if packing_enabled:
            packing = self._get_packing_config()
            sequence_length = packing.get("sequence_length", None) if packing else None
            if sequence_length is None:
                raise ValueError("Packing is enabled but packing.sequence_length is missing.")
            sequence_length = int(sequence_length)

            # Fail-fast guard for offline-packed datasets (avoid false positives).
            for split in dataset.keys():
                cols = set(dataset[split].column_names)
                if "attention_mask" not in cols and "labels" not in cols:
                    continue
                sample_n = min(64, len(dataset[split]))
                if sample_n == 0:
                    continue
                lengths = []
                for i in range(sample_n):
                    row = dataset[split][i]
                    lengths.append(len(row["input_ids"]))
                if len(set(lengths)) == 1 and lengths[0] == sequence_length:
                    raise ValueError(
                        "Packing is enabled but the dataset appears to be already offline packed "
                        f"(found attention_mask/labels and fixed-length input_ids == sequence_length={sequence_length}). "
                        "Disable packing or point to doc-level tokenization output."
                    )

            # Keep HF dataset in Python format for variable-length rows; dataloaders will be built in _pipeline.
            dataloaders: dict[str, DataLoader] = {}
        else:
            for split in dataset.keys():
                try:
                    dataset[split].set_format(type="torch", columns=required_columns)
                except Exception as e:
                    raise RuntimeError(f"Failed to set format for {split} split: {str(e)}")
            dataloaders = {}
            for split in dataset.keys():
                try:
                    dataloaders[split] = DataLoader(
                        dataset[split],
                        batch_size=config.batch_size,
                        shuffle=(split == "train"),
                        num_workers=config.num_workers,
                        pin_memory=True,
                        drop_last=False,
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to create DataLoader for {split} split: {str(e)}")
        return {
            "datasets": dataset,
            "dataloaders": dataloaders
        }
        
    def _pipeline(self, fabric: L.Fabric) -> None:
        """
        Orchestrate the complete training pipeline.

        This method sets deterministic seeds if provided, sets up monitoring and log directories,
        prepares dataloaders for Fabric, instantiates and configures the model (including gradient checkpointing),
        sets up the optimizer and scheduler, loads from a checkpoint if available, and finally starts training.

        Parameters:
        - fabric (L.Fabric): The Fabric instance coordinating distributed training.
        """
        packing = self._get_packing_config()
        cleanup_on_success = bool(packing and self._clean_index_cache_on_success(packing))
        # DETERMINISTIC RESULTS
        if self.config.get("seed", None) is not None:
            deterministic(self.config.seed)
            fabric.seed_everything(self.config.seed)

        # MONITORING
        self.monitor = Monitor(
            fabric,
            window_size=2,
            time_unit="seconds",
            log_iter_interval=self.config.get("log_iter_interval", 100),
        )

        # OUTPUT DIR AND SYNC
        if fabric.global_rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
        fabric.barrier()

        # FABRIC DATALOADERS SETUP
        if self._mixture_enabled:
            raw = self._build_mixture_packing_dataloaders(fabric)
            self.dataloaders = {
                "train": fabric.setup_dataloaders(raw["train"], use_distributed_sampler=False),
            }
            self._mixture_val_dataloaders = {
                dataset_id: fabric.setup_dataloaders(dl, use_distributed_sampler=False)
                for dataset_id, dl in raw["valid_by_source"].items()
            }
        elif self._packing_enabled:
            raw_dataloaders = self._build_packing_dataloaders(fabric)
            self.dataloaders = {
                k: fabric.setup_dataloaders(v, use_distributed_sampler=False)
                for k, v in raw_dataloaders.items()
            }
        else:
            self.dataloaders = {k: fabric.setup_dataloaders(v) for k, v in self.dataloaders.items()}

        if fabric.global_rank == 0 and int(self.config.get("verbose_level", 0)) >= 4:
            train_sampler = getattr(self.dataloaders.get("train"), "sampler", None)
            self.cli_logger.debug(
                "Dataloader setup: packing_enabled=%s train_sampler=%s train_num_batches=%s",
                self._packing_enabled,
                type(train_sampler).__name__ if train_sampler is not None else None,
                len(self.dataloaders["train"]) if "train" in self.dataloaders else None,
            )

        # MODEL: instantiate within the fabric.init_module() context
        t0 = time.perf_counter()
        with fabric.init_module():
            # Instantiate the model that inheriths from LightningModule
            self.model = self._instantiate_model()
        

            
        # Properly set up the model with fabric for FSDP TODO: check if this is the problem with Salamandra
        self.model = fabric.setup(self.model)
        self.cli_logger.info(f"Time to SetUp model: {time.perf_counter() - t0:.02f} seconds.")

        self.cli_logger.info(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        # OPTIMIZER
        optimizer = select_optimizer(
            self.config.get("optimizer", "adamw"), 
            self.model, 
            self.config.lr, 
            self.config.weight_decay, 
            self.config.beta1, 
            self.config.beta2
        )
        optimizer = fabric.setup_optimizers(optimizer)
        
        # SCHEDULER
        train_dataloader = self.dataloaders.get("train")
        if train_dataloader is None:
            raise RuntimeError("Training dataloader 'train' is missing.")
        train_num_batches = len(train_dataloader)
        if train_num_batches <= 0:
            raise ValueError(
                "Training dataloader yielded 0 batches. "
                "Reduce batch_size, disable sampler_drop_last, or use more data."
            )

        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        if gradient_accumulation_steps is None or int(gradient_accumulation_steps) <= 0:
            gradient_accumulation_steps = 1
        else:
            gradient_accumulation_steps = int(gradient_accumulation_steps)

        if self._packing_enabled and gradient_accumulation_steps > 1:
            if train_num_batches % gradient_accumulation_steps != 0:
                raise ValueError(
                    "Packing mode requires len(train_dataloader) to be divisible by gradient_accumulation_steps "
                    "to avoid silent lost optimizer steps. Adjust batch_size, sampler_drop_last, or gradient_accumulation_steps."
                )

        optimizer_steps_per_epoch = train_num_batches // gradient_accumulation_steps
        total_optimizer_steps = int(self.config.number_epochs) * optimizer_steps_per_epoch

        min_lr_value = float(self.config.get("min_lr", 0.0) or 0.0)
        max_lr_raw = self.config.get("max_lr", None)
        max_lr_value = float(max_lr_raw) if max_lr_raw is not None else None
        peak_lr_value = max_lr_value if max_lr_value is not None else float(self.config.lr)
        warmup_steps_value = int(max(0, total_optimizer_steps * float(self.config.warmup_proportion)))
        self._training_schedule_metadata = {
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "total_optimizer_steps": int(total_optimizer_steps),
            "warmup_steps": int(warmup_steps_value),
            "min_lr": float(min_lr_value),
            "peak_lr": float(peak_lr_value),
            "peak_lr_source": "max_lr" if max_lr_value is not None else "lr",
        }

        scheduler = select_scheduler(
            optimizer, 
            self.config.lr_scheduler, 
            self.config.number_epochs, 
            fabric.world_size, 
            self.config.batch_size, 
            train_dataloader.dataset,
            self.config.warmup_proportion, 
            base_lr=float(self.config.lr),
            min_lr=min_lr_value,
            max_lr=max_lr_value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            total_steps=total_optimizer_steps,
        )
        self.cli_logger.info(
            "Scheduler config: type=%s total_steps=%s warmup_steps=%s min_lr=%.8g peak_lr=%.8g peak_source=%s",
            self.config.lr_scheduler,
            total_optimizer_steps,
            warmup_steps_value,
            min_lr_value,
            peak_lr_value,
            "max_lr" if max_lr_value is not None else "lr",
        )
        
        # STATE
        expected_mixture_meta = self._build_mixture_resume_meta(fabric) if self._mixture_enabled else None
        run_metadata = self._build_run_metadata(fabric)
        self.state = {
            "model": self.model, 
            "optimizer": optimizer, 
            "hparams": dict(self.hparams),
            "run_metadata": run_metadata,
            "iter_num": 0, 
            "step_count": 0, 
            "current_epoch": 0,
            "scheduler": scheduler
        }
        if expected_mixture_meta is not None:
            self.state["mixture_meta"] = expected_mixture_meta
            self.state["mixture_runtime"] = self._build_mixture_runtime_state()
        # LOAD INITIAL WEIGHTS (for continual training learning)
        self._load_initial_weights(fabric)
        
        # RESUME (for continuing training)
        self._load_from_checkpoint(fabric, expected_mixture_meta=expected_mixture_meta)
        fabric.log_dict(
            {
                "train/optimizer_steps_per_epoch": float(optimizer_steps_per_epoch),
                "train/total_optimizer_steps": float(total_optimizer_steps),
                "train/warmup_steps": float(warmup_steps_value),
                "train/lr_min": float(min_lr_value),
                "train/lr_peak": float(peak_lr_value),
            },
            int(self.state.get("step_count", 0)),
        )
        
        # TRAINING
        train_time = time.perf_counter()
        self._train(fabric)
        self.state["run_metadata"] = self._build_run_metadata(fabric)
        if self._mixture_enabled:
            self._write_mixture_report(fabric)
        self.cli_logger.info(f"Training time: {(time.perf_counter() - train_time):.2f}s")
        if fabric.device.type == "cuda":
            self.cli_logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

        if cleanup_on_success and packing is not None:
            self._cleanup_index_caches_on_success(fabric, packing)
