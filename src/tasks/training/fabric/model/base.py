from typing import Tuple
import lightning as L
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger
from src.tasks.training.fabric.model.utils import AVAILABLE_MODELS
import logging

class BaseModel(L.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cli_logger = get_logger(__name__, kwargs.get("verbose_level", VerboseLevel.DEBUG))
        self.args = kwargs
        self.model_name = kwargs["model_name"]
        if kwargs['precision'] == 'bf16-true':
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32


    def _batch_validation(self, batch: Tuple[torch.Tensor, ...], step_type: str, *args) -> dict:
        """
        Validate batch structure for validation step.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing required tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key '{key}' in batch for {step_type} step")

    def _model_validation(self) -> None:
        """
        Validate model structure for validation step.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing required tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """
        if self.model is None:
            raise ValueError("The model class is not initialized")

        if not isinstance(self.model, AVAILABLE_MODELS):
            raise ValueError(f"The model class selected is not supported, currently supported models are: {AVAILABLE_MODELS}")
    
    def on_train_start(self):
        """
        Log training start information.

        This hook is executed at the beginning of training. It logs essential details about the
        training environment, such as world size, global rank, device, precision, and strategy,
        which are useful for debugging in distributed training scenarios.
        """
        # Access Fabric and its attributes
        self.cli_logger.info("-"*20)
        self.cli_logger.info("Training Start Statistics:")
        self.cli_logger.info("World Size: %s", self.fabric.world_size)
        self.cli_logger.info("Global Rank: %s", self.fabric.global_rank)
        self.cli_logger.info("Local Rank: %s", self.fabric.local_rank)
        self.cli_logger.info("Device: %s", self.fabric.device)
        self.cli_logger.info("Precision: %s", self.fabric.precision)
        self.cli_logger.info("Strategy: %s", self.fabric.strategy)
        self.cli_logger.info("-"*20)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one training step.

        This method is called for each batch of data during training. It processes the input batch,
        computes the forward pass through the model, and calculates the loss. The loss is then
        returned for backpropagation and optimization.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch of data containing input tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """

        # Validate batch and model (only on debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._batch_validation(batch, "train")
            self._model_validation()
        
        # Ensure correct dtypes for transformer models (indices must be long)
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].long()
        labels = batch['labels'].long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Log training metrics periodically
        if hasattr(self, 'global_step') and self.global_step % 5 == 0:
            self.cli_logger.debug(f"Training step {self.global_step}: loss = {outputs.loss.item():.4f}")
        
        return {
            "loss": outputs.loss,
            "outputs": outputs,
        }

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one validation step.

        This method is called for each batch of data during validation. It processes the input batch,
        computes the forward pass through the model, and calculates the loss. The loss is then
        returned for backpropagation and optimization.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch of data containing input tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """

        # Validate batch and model (only on debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._batch_validation(batch, "validation")
            self._model_validation()

        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Log validation metrics periodically
        if hasattr(self, 'global_step') and self.global_step % 5 == 0:
            self.cli_logger.debug(f"Validation step {self.global_step}: loss = {outputs.loss.item():.4f}")
        
        return {
            "loss": outputs.loss,
            "outputs": outputs,
        }

    def test_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one test step.

        This method is called for each batch of data during testing. It processes the input batch,
        computes the forward pass through the model, and calculates the loss. The loss is then
        returned for backpropagation and optimization.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch of data containing input tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """
        # Validate batch and model (only on debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._batch_validation(batch, "test")
            self._model_validation()


        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Log test metrics periodically
        if hasattr(self, 'global_step') and self.global_step % 5 == 0:
            self.cli_logger.debug(f"Test step {self.global_step}: loss = {outputs.loss.item():.4f}")
        
        return {
            "loss": outputs.loss,
            "outputs": outputs,
        }
    