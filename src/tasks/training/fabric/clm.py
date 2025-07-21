from typing import Tuple
import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger


"""
The LightningModules used for each case should be specified in this script.
This module defines generative models that extend LightningModule for training,
validation, and testing using a pre-trained causal language model.
"""


# Base class for Generative models with Fabric
class FabricCLM(L.LightningModule):
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
        super().__init__()
        self.cli_logger = get_logger(__name__, kwargs.get("verbose_level", VerboseLevel.DEBUG))
        self.args = kwargs
        model_name = kwargs["model_name"]
        
        if kwargs['precision'] == 'bf16-true':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_cache=False
        )
       
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
        Execute one training step.

        Processes a single batch of training data, computes the forward pass through the model,
        calculates the loss, and logs it for monitoring. The batch is expected to include tensors
        for 'input_ids', 'attention_mask', and 'labels'.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing the inputs required for the model.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the computed 'loss' and model 'outputs'.
        """
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return {
            "loss":outputs.loss,
            "outputs":outputs,
        }

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one validation step.

        Runs a forward pass for a batch of validation data, calculates the validation loss,
        and logs it for performance monitoring.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing required tensors.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the validation 'loss' and model 'outputs'.
        """
        
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        # Log the validation loss for progress monitoring.
        self.cli_logger.debug(f"Validation_loss: {loss.item()}")
        return {
            "loss": loss,
            "outputs": outputs,
        }

    def test_step(self, batch: Tuple[torch.Tensor, ...], *args) -> dict:
        """
        Execute one test step.

        Processes a test batch, computes the forward pass of the model along with the loss,
        and logs the result. This step is used to assess the performance of the model on test data.

        Args:
            batch (Tuple[torch.Tensor, ...]): A batch containing the input data.
            *args: Additional arguments (if any).

        Returns:
            dict: Contains the test 'loss' and model 'outputs'.
        """
        
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        # Log the test loss for debugging purposes.
        self.cli_logger.debug(f"Test_loss: {loss.item()}")
        return {
            "loss": loss,
            "outputs": outputs,
        }