import lightning as L
from transformers import AutoModelForCausalLM
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

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

    # def configure_optimizers(self): 
    #     """Prepare optimizer and schedule (linear warmup and decay)"""
    #     model = self.model
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]