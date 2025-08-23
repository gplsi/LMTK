from typing import Tuple
import lightning as L
from transformers import AutoModelForCausalLM, AutoConfig
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger
from src.tasks.training.fabric.model.base import BaseModel


"""
The LightningModules used for each case should be specified in this script.
This module defines generative models that extend LightningModule for training,
validation, and testing using a pre-trained causal language model.
"""


# Base class for Generative models with Fabric
class FabricCLM(BaseModel):
    """
    PyTorch Lightning Module for Generative Models with Fabric.

    This class wraps a Hugging Face pre-trained causal language model and integrates
    it into a LightningModule for streamlined training, validation, and testing. It also
    leverages Fabric's distributed environment attributes for enhanced logging during training.
    """    

    def __init__(self, **kwargs) -> None:
        """
        Initialize the FabricGeneration model.

        Loads a pre-trained AutoModelForCausalLM model using the specified parameters.
        Determines the torch data type based on the 'precision' key in kwargs,
        configures logging, and stores additional arguments for future use.

        Args:
            **kwargs: Arbitrary keyword arguments that include:
                - model_name (str): Identifier for the pre-trained model.
                - precision (str): Model precision setting ('bf16-true' selects bfloat16, otherwise uses float32).
                - verbose_level (Optional): Logging verbosity level.
                - zero_stage (Optional): DeepSpeed ZeRO stage level.
        """
        super().__init__(**kwargs)
        
        from_scratch: bool = bool(kwargs.get("from_scratch", False))
        if from_scratch:
            # Initialize model weights from scratch using the base config of model_name
            cfg = AutoConfig.from_pretrained(self.model_name)
            cfg.use_cache = False
            self.model = AutoModelForCausalLM.from_config(cfg)
        else:
            # Default behavior: load pretrained weights
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                use_cache=False
            )