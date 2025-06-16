"""
Publish Orchestrator Module

This module provides the PublishOrchestrator class, which handles the orchestration 
of the model publishing process. The workflow includes validating configuration, 
formatting the model from FSDP checkpoint to HuggingFace format, and uploading 
the model to a designated repository.
"""

from box import Box
from transformers import AutoTokenizer
from src.utils.logging import get_logger, set_logger_level
from src.tasks.publish.upload.huggingface import UploadHuggingface
from src.tasks.publish.format.utils import FORMAT_HANDLERS
from src.utils.orchestrator import BaseOrchestrator


class PublishOrchestrator(BaseOrchestrator):
    """
    Orchestrates the publish workflow.
    """
    def __init__(self, config: Box):
        super().__init__(config)
        self.logger = get_logger(__name__, self.verbose_level)
        # Only load tokenizer if publish config and base_model exist
        if hasattr(self.config, 'publish') and self.config.publish.get('base_model'):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.publish.get('base_model'))
                self.logger.debug(f"Loaded tokenizer for base model: {self.config.publish.get('base_model')}")
            except Exception as e:
                self.logger.warning(f"Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.logger.warning("No base_model provided; tokenizer will be None.")
            self.tokenizer = None
    
    def validate_config(self):
        """
        Validate the publish configuration.
        """
        if not hasattr(self.config, 'publish') or not self.config.publish:
            raise ValueError("Publish configuration must be provided")
        if not self.config.publish.get('host'):
            raise ValueError("Publish host must be provided")
        if not self.config.publish.get('base_model'):
            raise ValueError("Publish base_model must be provided")
        if not self.config.publish.get('repo_id'):
            raise ValueError("Publish repo_id must be provided")
        if not self.config.publish.get('checkpoint_path'):
            raise ValueError("Publish checkpoint_path must be provided")
        if not self.config.publish.get('format'):
            raise ValueError("Publish format must be provided")

    def format_model(self):
        format_handler = FORMAT_HANDLERS[self.config.publish.get('format')](
            self.config.publish.get('host'), 
            self.config.publish.get('base_model'), 
            self.config.publish.get('checkpoint_path')
        )
        model = format_handler.execute()
        return model
    
    def upload_model(self, model):
        upload_handler = UploadHuggingface(
            base_model=self.config.publish.get('base_model'),
            model=model,
            tokenizer=self.tokenizer,
            repo_id=self.config.publish.get('repo_id')
        )
        
        # Get configuration values with defaults from schema
        base_model = self.config.publish.get('base_model')
        message = self.config.publish.get('commit_message', f"Add {base_model} model and tokenizer")
        max_shard_size = self.config.publish.get('max_shard_size', "5GB")
        safe_serialization = self.config.publish.get('safe_serialization', True)
        create_pr = self.config.publish.get('create_pr', False)
        
        self.logger.info(f"Uploading model to {self.config.publish.get('repo_id')} with shard size {max_shard_size}")
        
        return upload_handler.execute(
            message=message,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            create_pr=create_pr
        )
    
    def execute(self):
        try:
            self.logger.info("Starting publish workflow")
            self.validate_config()
            model = self.format_model()
            self.upload_model(model)
            self.logger.info("✅ Publish workflow completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error in publishing workflow: {e}")
            raise
        