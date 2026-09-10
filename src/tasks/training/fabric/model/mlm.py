from typing import Tuple, Dict, Any
import lightning as L
from transformers import AutoModelForMaskedLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger
from src.tasks.training.fabric.model.base import BaseModel
import logging

"""
The LightningModules used for masked language model training.
This module defines MLM models that extend LightningModule for training,
validation, and testing using a pre-trained masked language model with pre-tokenized
datasets that include proper MLM masking.
"""


# Base class for Masked Language Model training with Fabric
class FabricMLM(BaseModel):
    """
    PyTorch Lightning Module for Masked Language Model Training with Fabric.

    This class wraps a Hugging Face pre-trained masked language model and integrates
    it into a LightningModule for MLM training. It handles pre-tokenized datasets
    where tokens are randomly masked for the model to predict. Unlike causal LM,
    MLM can see the entire sequence context when predicting masked tokens.
    It leverages Fabric's distributed environment attributes for enhanced logging during training.
    """    

    def __init__(self, **kwargs) -> None:
        """
        Initialize the FabricMLM model.

        Loads a pre-trained AutoModelForMaskedLM model using the specified parameters.
        Determines the torch data type based on the 'precision' key in kwargs,
        configures logging, and stores additional arguments for future use.
        Optimized for masked language model training with pre-tokenized datasets.

        Args:
            **kwargs: Arbitrary keyword arguments that include:
                - model_name (str): Identifier for the pre-trained model.
                - precision (str): Model precision setting ('bf16-true' selects bfloat16, otherwise uses float32).
                - verbose_level (Optional): Logging verbosity level.
                - zero_stage (Optional): DeepSpeed ZeRO stage level.
                - mlm_probability (Optional): Probability of masking tokens (default: 0.15).
                - ignore_index (Optional): Token ID to ignore in loss calculation (default: -100).
        """
        super().__init__(**kwargs)        
        self.mlm_probability = kwargs.get("mlm_probability", 0.15)
        self.ignore_index = kwargs.get("ignore_index", -100)       
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
        )
        
        self.cli_logger.info(f"Initialized MLM model: {self.model_name}")
        self.cli_logger.info(f"MLM probability: {self.mlm_probability}")
        self.cli_logger.info(f"Ignore index for loss calculation: {self.ignore_index}")
        self.cli_logger.info(f"Model dtype: {self.torch_dtype}")
       
    def _log_mlm_metrics(self, batch, loss, step_type="step"):
        masked_tokens = (batch['labels'] != self.ignore_index).sum().item()
        total_tokens = batch['labels'].numel()
        masking_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
        self.cli_logger.debug(
            f"MLM {step_type}: loss = {loss.item():.4f}, "
            f"masked_tokens = {masked_tokens}, masking_ratio = {masking_ratio:.3f}"
        )

    def training_step(self, batch, *args):
        # Use base class validation and forward pass
        out = super().training_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="train")
        return out

    def validation_step(self, batch, *args):
        out = super().validation_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="validation")
        return out

    def test_step(self, batch, *args):
        out = super().test_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="test")
        return out