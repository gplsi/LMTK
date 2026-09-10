from typing import Tuple
import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger
from src.tasks.training.fabric.model.base import BaseModel


"""
The LightningModules used for instruction tuning training.
This module defines instruction tuning models that extend LightningModule for training,
validation, and testing using a pre-trained causal language model with pre-tokenized
instruction datasets that include proper label masking.
"""


# Base class for Instruction Tuning models with Fabric
class FabricInstruction(BaseModel):
    """
    PyTorch Lightning Module for Instruction Tuning with Fabric.

    This class wraps a Hugging Face pre-trained causal language model and integrates
    it into a LightningModule for instruction tuning. It handles pre-tokenized datasets
    where labels are properly masked to only train on response tokens, not instruction tokens.
    It leverages Fabric's distributed environment attributes for enhanced logging during training.
    """    

    def __init__(self, **kwargs) -> None:
        """
        Initialize the FabricInstruction model.

        Loads a pre-trained AutoModelForCausalLM model using the specified parameters.
        Determines the torch data type based on the 'precision' key in kwargs,
        configures logging, and stores additional arguments for future use.
        Optimized for instruction tuning with pre-tokenized datasets.

        Args:
            **kwargs: Arbitrary keyword arguments that include:
                - model_name (str): Identifier for the pre-trained model.
                - precision (str): Model precision setting ('bf16-true' selects bfloat16, otherwise uses float32).
                - verbose_level (Optional): Logging verbosity level.
                - zero_stage (Optional): DeepSpeed ZeRO stage level.
                - ignore_index (Optional): Token ID to ignore in loss calculation (default: -100).
        """
        super().__init__(**kwargs)
        self.cli_logger = get_logger(__name__, kwargs.get("verbose_level", VerboseLevel.DEBUG))
        self.args = kwargs
        model_name = kwargs["model_name"]
        
        # Store ignore_index for loss calculation (standard for instruction tuning)
        self.ignore_index = kwargs.get("ignore_index", -100)
                
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            use_cache=False
        )
        
