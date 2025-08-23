"""
Convert Orchestrator Module

This module provides the ConvertOrchestrator class, which handles the orchestration 
of the model conversion process. The workflow includes validating configuration and 
formatting the model from the checkpoint format to the platform format it will be uploaded.
"""

from box import Box
from transformers import AutoTokenizer
from src.utils.logging import get_logger, set_logger_level
from src.tasks.convert.formats.utils import FORMAT_HANDLERS
from src.utils.orchestrator import BaseOrchestrator

class ConvertOrchestrator(BaseOrchestrator):
    """
    Orchestrates the conversion workflow.
    """

    def __init__(self, config: Box):
        super().__init__(config)
        self.logger = get_logger(__name__, self.verbose_level)
        # Only load tokenizer if convert config and base_model exist
        if hasattr(self.config, 'convert') and self.config.convert.get('base_model'):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.convert.get('base_model'))
                self.logger.debug(f"Loaded tokenizer for base model: {self.config.convert.get('base_model')}")
            except Exception as e:
                self.logger.warning(f"Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.logger.warning("No base_model provided; tokenizer will be None.")
            self.tokenizer = None
    
    def _validate_config(self):
        """
        Validate the convert configuration.
        """
        if not hasattr(self.config, 'convert') or not self.config.convert:
            raise ValueError("Convert configuration must be provided")
        if not self.config.convert.get('base_model'):
            raise ValueError("Convert base_model must be provided")
        if not self.config.convert.get('checkpoint_path'):
            raise ValueError("Convert checkpoint_path must be provided")
        if not self.config.convert.get('initial_format'):
            raise ValueError("Convert initial format must be provided")
        if not self.config.convert.get('final_format'):
            raise ValueError("Convert final format must be provided")
        if not self.config.convert.get('output_dir'):
            raise ValueError("Convert output_dir must be provided")
    
    def _convert(self):
        initial_format = self.config.convert.get('initial_format')
        final_format = self.config.convert.get('final_format')
        key = f"{initial_format}_{final_format}"
        task_type = self.config.convert.get('task_type')  # optional: "mlm" | "clm" | "instruction"
        format_handler = FORMAT_HANDLERS[key](
            self.config.convert.get('base_model'), 
            self.config.convert.get('checkpoint_path'),
            task_type=task_type
        )
        summary = format_handler.execute(output_dir=self.config.convert.get('output_dir'))
        return summary
    
    def execute(self):
        try:
            self.logger.info("Starting convert workflow")
            self._validate_config()
            summary = self._convert()
            self.logger.info("✅ Convert workflow completed")
            return summary
        except Exception as e:
            self.logger.error(f"❌ Error in convert workflow: {e}")
            raise
        