from src.tasks.training.fabric.wrappers.models.gpt2.classes import *
from transformers import GPT2Config
set_global_gpt2_config(GPT2Config.from_pretrained("gpt2"))

