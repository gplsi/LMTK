from typing import Tuple, Dict, Any
import lightning as L
from transformers import AutoModelForMaskedLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger


"""
The LightningModules used for masked language model training.
This module defines MLM models that extend LightningModule for training,
validation, and testing using a pre-trained masked language model with pre-tokenized
datasets that include proper MLM masking.
"""


# Base class for Masked Language Model training with Fabric
class FabricMLM(L.LightningModule):
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
        super().__init__()
        self.cli_logger = get_logger(__name__, kwargs.get("verbose_level", VerboseLevel.DEBUG))
        self.args = kwargs
        model_name = kwargs["model_name"]
        
        # Store MLM-specific parameters
        self.mlm_probability = kwargs.get("mlm_probability", 0.15)
        self.ignore_index = kwargs.get("ignore_index", -100)
        
        if kwargs['precision'] == 'bf16-true':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        
        # Log MLM-specific configuration
        self.cli_logger.info(f"Initialized MLM model: {model_name}")
        self.cli_logger.info(f"MLM probability: {self.mlm_probability}")
        self.cli_logger.info(f"Ignore index for loss calculation: {self.ignore_index}")
        self.cli_logger.info(f"Model dtype: {torch_dtype}")
       
    def on_train_start(self):
        """
        Log training start information for MLM training.

        This hook is executed at the beginning of training. It logs essential details about the
        training environment, such as world size, global rank, device, precision, and strategy,
        which are useful for debugging in distributed training scenarios.
        """
        # Access Fabric and its attributes
        self.cli_logger.debug("-"*20)
        self.cli_logger.debug("MLM Training Start Statistics:")
        self.cli_logger.debug("World Size: %s", self.fabric.world_size)
        self.cli_logger.debug("Global Rank: %s", self.fabric.global_rank)
        self.cli_logger.debug("Local Rank: %s", self.fabric.local_rank)
        self.cli_logger.debug("Device: %s", self.fabric.device)
        self.cli_logger.debug("Precision: %s", self.fabric.precision)
        self.cli_logger.debug("Strategy: %s", self.fabric.strategy)
        self.cli_logger.debug("MLM Probability: %s", self.mlm_probability)
        self.cli_logger.debug("-"*20)

    def training_step(self, batch: Dict[str, torch.Tensor], *args) -> dict:
        """
        Execute one training step for MLM training.

        Processes a single batch of pre-tokenized MLM data, computes the forward pass 
        through the model, and calculates the loss. The labels contain the original tokens
        for masked positions and ignore_index (-100) for unmasked positions.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing the inputs required for the model.
                Expected keys: 'input_ids', 'attention_mask', 'labels'
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """
        # Validate batch structure
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key '{key}' in batch for MLM training")
        
        # Forward pass with pre-tokenized and properly masked data
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Log training metrics periodically
        if hasattr(self, 'global_step') and self.global_step % 100 == 0:
            # Calculate number of masked tokens for logging
            masked_tokens = (batch['labels'] != self.ignore_index).sum().item()
            total_tokens = batch['labels'].numel()
            masking_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
            
            self.cli_logger.debug(f"Training step {self.global_step}: loss = {outputs.loss.item():.4f}, "
                                f"masked_tokens = {masked_tokens}, masking_ratio = {masking_ratio:.3f}")
        
        return {
            "loss": outputs.loss,
            "outputs": outputs,
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], *args) -> dict:
        """
        Execute one validation step for MLM training.

        Runs a forward pass for a batch of pre-tokenized MLM validation data, calculates the 
        validation loss on masked tokens, and logs it for performance monitoring.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing required tensors.
                Expected keys: 'input_ids', 'attention_mask', 'labels'
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the validation 'loss' and model 'outputs'.
        """
        # Validate batch structure
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key '{key}' in validation batch for MLM training")
        
        # Forward pass with pre-tokenized and properly masked data
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        
        # Calculate validation metrics
        masked_tokens = (batch['labels'] != self.ignore_index).sum().item()
        total_tokens = batch['labels'].numel()
        masking_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
        
        # Log the validation loss and metrics for progress monitoring
        self.cli_logger.debug(f"Validation loss: {loss.item():.4f}, "
                            f"masked_tokens: {masked_tokens}, masking_ratio: {masking_ratio:.3f}")
        
        return {
            "loss": loss,
            "outputs": outputs,
        }

    def test_step(self, batch: Dict[str, torch.Tensor], *args) -> dict:
        """
        Execute one test step for MLM training.

        Processes a test batch with pre-tokenized MLM data, computes the forward pass 
        of the model along with the loss on masked tokens, and logs the result. 
        This step is used to assess the performance of the MLM model on test data.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing the input data.
                Expected keys: 'input_ids', 'attention_mask', 'labels'
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the test 'loss' and model 'outputs'.
        """
        # Validate batch structure
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key '{key}' in test batch for MLM training")
        
        # Forward pass with pre-tokenized and properly masked data
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        
        # Calculate test metrics
        masked_tokens = (batch['labels'] != self.ignore_index).sum().item()
        total_tokens = batch['labels'].numel()
        masking_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
        
        # Log the test loss and metrics for debugging purposes
        self.cli_logger.debug(f"Test loss: {loss.item():.4f}, "
                            f"masked_tokens: {masked_tokens}, masking_ratio: {masking_ratio:.3f}")
        
        return {
            "loss": loss,
            "outputs": outputs,
        }








