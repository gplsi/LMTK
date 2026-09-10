"""
This module defines a base trainer class using Lightning Fabric for training deep learning models.
It encapsulates functionalities such as dataset loading, model instantiation, training loop, validation,
checkpointing, logging, and gradient accumulation. The FabricTrainerBase class is designed as an abstract base
class, providing a template for custom training strategies.
"""

import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler
from src.tasks.training.reproducibility import strict_enabled, seed_worker, prepare_process, write_manifest
from datasets import load_from_disk, DatasetDict
from datasets import Dataset as HFDataset
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
from typing import Tuple, Union
from box import Box
from transformers import AutoTokenizer
import lightning as L
import os

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

MODEL_CLASS_MAP = {
    "clm_training": FabricCLM,
    "mlm_training": FabricMLM,
    "instruction": FabricInstruction,
}

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        self.strict_determinism = strict_enabled(config)
        prepare_process(config)
        self.checkpoint_path = checkpoint_path
        self.state = {}
        self.dataset = dataset
        
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
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
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

    def _ensure_validation_split(self, dataset: Union[DatasetDict, HFDataset]) -> DatasetDict:
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

        if isinstance(dataset, HFDataset):
            self.cli_logger.info("Single dataset provided, wrapping as training data only")
            dataset = DatasetDict({"train": dataset})
        else:
            dataset = DatasetDict(dataset)

        if "valid" in dataset:
            return dataset

        if "validation" in dataset:
            self.cli_logger.info("Found 'validation' split; reusing it as 'valid'.")
            validation_dataset = dataset["validation"]
            del dataset["validation"]
            dataset["valid"] = validation_dataset
            return dataset

        split_config = getattr(self.config, "validation_split", None)
        if not split_config:
            return dataset

        if isinstance(split_config, Box):
            split_config = split_config.to_dict()

        shuffle = bool(split_config.get("shuffle", True))
        seed = split_config.get("seed", getattr(self.config, "seed", None))

        proportion = split_config.get("proportion")
        count = split_config.get("count")

        if proportion is None and count is None:
            raise ValueError("validation_split configuration must include 'proportion' or 'count'.")

        train_dataset = dataset.get("train")
        if train_dataset is None:
            raise ValueError("Training split 'train' is required to derive a validation split.")

        total_examples = len(train_dataset)
        if total_examples < 2:
            raise ValueError("Not enough training examples to create a validation split (need at least 2).")

        candidate_sizes: list[int] = []
        if proportion is not None:
            if not isinstance(proportion, (int, float)):
                raise TypeError("validation_split.proportion must be a numeric value between 0 and 1.")
            proportion_value = float(proportion)
            if not 0 < proportion_value < 1:
                raise ValueError("validation_split.proportion must be between 0 and 1.")
            candidate_sizes.append(max(1, int(round(total_examples * proportion_value))))

        if count is not None:
            if not isinstance(count, int):
                raise TypeError("validation_split.count must be an integer.")
            if count <= 0:
                raise ValueError("validation_split.count must be greater than 0.")
            candidate_sizes.append(count)

        if not candidate_sizes:
            raise ValueError("Unable to determine validation split size from configuration.")

        val_count = min(candidate_sizes)
        if val_count >= total_examples:
            adjusted_val_count = total_examples - 1
            if adjusted_val_count <= 0:
                raise ValueError(
                    f"Requested validation size {val_count} is incompatible with training size {total_examples}."
                )
            self.cli_logger.warning(
                f"Requested validation size {val_count} >= training size {total_examples}. "
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
            f"Created validation split with {len(dataset['valid'])} examples "
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
    
    def _load_from_checkpoint(self, fabric: L.Fabric) -> None:
        """
        Load model and optimizer state from a checkpoint if a checkpoint path is provided.

        Parameters:
        - fabric (L.Fabric): The Fabric instance handling the training.
        """
        if self.checkpoint_path is not None:
            self.cli_logger.info(f"Resuming training from '{self.checkpoint_path}'")
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
        
        # Average every microbatch, including the update-triggering one.
        # Flush and normalize a shorter final group at the end of each epoch.
        total_batches = len(self.dataloaders['train'])
        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, total_batches - group_start)
        is_accumulating = (step + 1) % gradient_accumulation_steps != 0 and step + 1 < total_batches
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            training_output = model.training_step(batch, step)
            outputs = training_output["outputs"]
            loss = training_output["loss"]
            
            real_loss = loss / group_size
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
            if not hasattr(self, "datasets") or not hasattr(self, "dataloaders") or "train" not in self.datasets:
                self.cli_logger.warning("Cannot perform intra-epoch validation/checkpoint scheduling: missing dataset or dataloader structure")
                return

            current_epoch = self.state.get("current_epoch", 1)
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

            if "valid" in getattr(self, "dataloaders", {}):
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
            if self.strict_determinism:
                self.dataloaders['train'].sampler.set_epoch(epoch)
            
            if fabric.global_rank == 0:
                self.cli_logger.debug(f"Running Epoch {epoch + 1} of {epochs}")
            batch_iterator = tqdm(self.dataloaders['train'], mininterval=0, colour="blue") \
                if fabric.global_rank == 0 else self.dataloaders['train']
            batch_iterator, resume_iter = self._get_resume_iterator(batch_iterator, resume_iter)
            if batch_iterator is None:
                continue            
            for step, batch in enumerate(batch_iterator):
                self.train_iter_t0 = time.perf_counter()
                if self.config.gradient_accumulation_steps:
                    _, loss = self._accumulate_training(fabric, model, batch, step)
                else:
                    _, loss = self._normal_training(fabric, model, batch, step)
                self.total_lengths += batch["input_ids"].size(1)
                self.train_t1 = time.perf_counter()
                self._train_logs(fabric, loss)
                
            self._try_validate(fabric, epochFinished=True)
        self._try_validate(fabric, trainingFinished=True)
    
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
      
    def _load_fabric_datasets_dataloaders(self, config: Box, dataset: Union[HFDataset, DatasetDict]) -> dict[DatasetDict, Union[dict[str, DataLoader], DataLoader]]:
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
        
        if not isinstance(dataset, (DatasetDict, HFDataset)):
            raise TypeError("Expected dataset to be a DatasetDict or Dataset")
        if not hasattr(config, 'batch_size') or not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise ValueError("config.batch_size must be a positive integer")
        if not hasattr(config, 'num_workers') or not isinstance(config.num_workers, int) or config.num_workers < 0:
            raise ValueError("config.num_workers must be a non-negative integer")
        
        dataset = self._ensure_validation_split(dataset)

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
                
        required_columns = ["input_ids", "attention_mask", "labels"]
        for split in dataset.keys():
            missing_columns = [col for col in required_columns if col not in dataset[split].column_names]
            if missing_columns:
                raise ValueError(f"Missing required columns {missing_columns} in {split} split")
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
                    drop_last=False
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
        # DETERMINISTIC RESULTS
        if self.config.get("seed", None) is not None:
            deterministic(self.config.seed, strict=self.strict_determinism)
            fabric.seed_everything(self.config.seed)

        # MONITORING
        self.monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=self.config.get("log_iter_interval", 100))

        # OUTPUT DIR AND SYNC
        if fabric.global_rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
        fabric.barrier()

        # Independent, explicitly seeded sampling in strict mode. Validation and
        # model initialization cannot consume the training sampler's RNG stream.
        if self.strict_determinism:
            for split, loader in list(self.dataloaders.items()):
                sampler = DistributedSampler(
                    loader.dataset, num_replicas=fabric.world_size,
                    rank=fabric.global_rank, shuffle=(split == "train"),
                    seed=self.config.seed, drop_last=False,
                )
                generator = torch.Generator().manual_seed(
                    self.config.seed + fabric.global_rank + (0 if split == "train" else 1)
                )
                self.dataloaders[split] = DataLoader(
                    loader.dataset, batch_size=self.config.batch_size,
                    sampler=sampler, num_workers=self.config.num_workers,
                    pin_memory=True, drop_last=False,
                    generator=generator, worker_init_fn=seed_worker,
                )

        # FABRIC DATALOADERS SETUP
        self.dataloaders = {k: fabric.setup_dataloaders(v, use_distributed_sampler=not self.strict_determinism) for k, v in self.dataloaders.items()}

        # MODEL: instantiate within the fabric.init_module() context
        t0 = time.perf_counter()
        with fabric.init_module():
            # Instantiate the model that inheriths from LightningModule
            self.model = self._instantiate_model()
        

            
        if self.strict_determinism:
            write_manifest(self.config, fabric, self.model, self.datasets)

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
        scheduler = select_scheduler(
            optimizer, 
            self.config.lr_scheduler, 
            self.config.number_epochs, 
            fabric.world_size, 
            self.config.batch_size, 
            self.datasets['train'],
            self.config.warmup_proportion, 
            self.config.gradient_accumulation_steps
        )        
        
        # STATE
        self.state = {
            "model": self.model, 
            "optimizer": optimizer, 
            "hparams": self.hparams, 
            "iter_num": 0, 
            "step_count": 0, 
            "current_epoch": 0,
            "scheduler": scheduler
        }
        # LOAD INITIAL WEIGHTS (for continual training learning)
        self._load_initial_weights(fabric)
        
        # RESUME (for continuing training)
        self._load_from_checkpoint(fabric)
        
        # TRAINING
        train_time = time.perf_counter()
        self._train(fabric)
        self.cli_logger.info(f"Training time: {(time.perf_counter() - train_time):.2f}s")
        if fabric.device.type == "cuda":
            self.cli_logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
