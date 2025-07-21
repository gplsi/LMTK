from typing import Tuple
import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger


"""
The LightningModules used for instruction tuning training.
This module defines instruction tuning models that extend LightningModule for training,
validation, and testing using a pre-trained causal language model with pre-tokenized
instruction datasets that include proper label masking.
"""


# Base class for Instruction Tuning models with Fabric
class FabricInstruction(L.LightningModule):
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
        super().__init__()
        self.cli_logger = get_logger(__name__, kwargs.get("verbose_level", VerboseLevel.DEBUG))
        self.args = kwargs
        model_name = kwargs["model_name"]
        
        # Store ignore_index for loss calculation (standard for instruction tuning)
        self.ignore_index = kwargs.get("ignore_index", -100)
        
        if kwargs['precision'] == 'bf16-true':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_cache=False
        )
        
        # Log instruction tuning specific configuration
        self.cli_logger.info(f"Initialized instruction tuning model: {model_name}")
        self.cli_logger.info(f"Ignore index for loss calculation: {self.ignore_index}")
        self.cli_logger.info(f"Model dtype: {torch_dtype}")
       
    def on_train_start(self):
        """
        Log training start information.

        This hook is executed at the beginning of training. It logs essential details about the
        training environment, such as world size, global rank, device, precision, and strategy,
        which are useful for debugging in distributed training scenarios.
        """
        # Access Fabric and its attributes
        self.cli_logger.debug("-"*20)
        self.cli_logger.debug("Training Start Statistics:")
        self.cli_logger.debug("World Size: %s", self.fabric.world_size)
        self.cli_logger.debug("Global Rank: %s", self.fabric.global_rank)
        self.cli_logger.debug("Local Rank: %s", self.fabric.local_rank)
        self.cli_logger.debug("Device: %s", self.fabric.device)
        self.cli_logger.debug("Precision: %s", self.fabric.precision)
        self.cli_logger.debug("Strategy: %s", self.fabric.strategy)
        self.cli_logger.debug("-"*20)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one training step for instruction tuning.

        Processes a single batch of pre-tokenized instruction data, computes the forward pass 
        through the model, and calculates the loss. The labels are already properly masked
        to only compute loss on response tokens (instruction tokens are masked with ignore_index).

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
                raise ValueError(f"Missing required key '{key}' in batch for instruction tuning")
        
        # Forward pass with pre-tokenized and properly masked labels
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Log training metrics periodically
        if hasattr(self, 'global_step') and self.global_step % 100 == 0:
            self.cli_logger.debug(f"Training step {self.global_step}: loss = {outputs.loss.item():.4f}")
        
        return {
            "loss": outputs.loss,
            "outputs": outputs,
        }

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one validation step for instruction tuning.

        Runs a forward pass for a batch of pre-tokenized validation data, calculates the 
        validation loss on properly masked labels, and logs it for performance monitoring.

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
                raise ValueError(f"Missing required key '{key}' in validation batch for instruction tuning")
        
        # Forward pass with pre-tokenized and properly masked labels
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        # Log the validation loss for progress monitoring
        self.cli_logger.debug(f"Validation loss: {loss.item():.4f}")
        
        return {
            "loss": loss,
            "outputs": outputs,
        }

    def test_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one test step for instruction tuning.

        Processes a test batch with pre-tokenized instruction data, computes the forward pass 
        of the model along with the loss on properly masked labels, and logs the result. 
        This step is used to assess the performance of the instruction-tuned model on test data.

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
                raise ValueError(f"Missing required key '{key}' in test batch for instruction tuning")
        
        # Forward pass with pre-tokenized and properly masked labels
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        # Log the test loss for debugging purposes
        self.cli_logger.debug(f"Test loss: {loss.item():.4f}")
        
        return {
            "loss": loss,
            "outputs": outputs,
        }


class FabricInstructionTuning(FabricInstruction):
    """
    PyTorch Lightning Module for Instruction Tuning with Fabric.

    This class extends FabricInstruction to provide specialized functionality for instruction tuning.
    It is designed to work with pre-tokenized instruction datasets where labels are properly masked
    to only compute loss on response tokens. This ensures the model learns to generate appropriate
    responses to instructions without being trained on the instruction tokens themselves.
    
    Key features:
    - Handles pre-tokenized datasets with proper label masking
    - Optimized for instruction-response pairs
    - Enhanced logging for instruction tuning metrics
    - Validation of batch structure for instruction tuning requirements
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize the FabricInstructionTuning model.
        
        Args:
            **kwargs: Keyword arguments passed to parent FabricInstruction class.
                Additional instruction tuning specific parameters can be added here.
        """
        super().__init__(**kwargs)
        
        # Log initialization of instruction tuning model
        self.cli_logger.info("Initialized FabricInstructionTuning model for instruction tuning")
        self.cli_logger.info("Model expects pre-tokenized datasets with proper label masking")
    
    def on_train_start(self):
        """
        Log training start information specific to instruction tuning.
        
        Extends the parent method with instruction tuning specific logging.
        """
        super().on_train_start()
        self.cli_logger.info("Starting instruction tuning training")
        self.cli_logger.info("Training only on response tokens (instruction tokens are masked)")





