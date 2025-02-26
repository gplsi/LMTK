import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW


# Base class for Generative models with Fabric
class FabricGeneration(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args["model_name"]
        if args['precision'] == 'bf16-true':
            torch_dtype = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, 
        use_cache=False, attn_implementation="flash_attention_2")

    def on_train_start(self):
        # Access Fabric and its attributes
        print(self.fabric.world_size)

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        return outputs, loss

    def validation_step(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs
        return outputs

    def test_step(self, batch):
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("test_loss", loss.loss, prog_bar=True)
        return outputs
