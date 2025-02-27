import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

from utils.logging import VerboseLevel, get_logger


"""
The LightningModules used for each case should be specified in this script
"""

# Base class for Generative models with Fabric
class FabricGeneration(L.LightningModule):
    def __init__(self, **kwargs):
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

    def training_step(self, batch, *args):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return {
            "loss":outputs.loss,
            "outputs":outputs,
        }

    def validation_step(self, batch, *args):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return {
            "loss":outputs.loss,
            "outputs":outputs,
        }

    def test_step(self, batch, *args):
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.cli_logger.debug("test_loss", loss.loss, prog_bar=True)
        return {
            "loss":outputs.loss,
            "outputs":outputs,
        }


# Base class for Classification models with Fabric
# TODO: Add classification model





