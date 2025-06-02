from logging import getLogger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



class UploadHuggingface:
    def __init__(self, base_model, model, tokenizer, repo_id):
        self.base_model = base_model
        self.model = model
        self.tokenizer = tokenizer
        self.repo_id = repo_id
        self.logger = getLogger(__name__)

    def _upload_tokenizer(self, message: str):
        self.tokenizer.push_to_hub(
            self.repo_id,
            commit_message=message
        )

    def _upload_model(self, message: str, max_shard_size:str ="5GB", safe_serialization:bool =True, create_pr:bool =False):
        self.model.push_to_hub(
            self.repo_id,
            commit_message=message,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            create_pr=create_pr
        )

    def _validate_upload(self):
        test_model = AutoModelForCausalLM.from_pretrained(self.repo_id)
        test_tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        
        # Test functionality
        test_input = test_tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            output = test_model(**test_input)
        
        self.logger.info("✅ FSDP model validation successful!")
        self.logger.info(f"✅ Model has {test_model.num_parameters():,} parameters")
        self.logger.info(f"Model available at: https://huggingface.co/{self.repo_id}")
        
        return True

    def execute(self, message: str="Add model and tokenizer", max_shard_size:str ="5GB", safe_serialization:bool =True, create_pr:bool =False):
        self._upload_model(
            message=message,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            create_pr=create_pr
        )
        
        self._upload_tokenizer(message=message)
        self._validate_upload()
