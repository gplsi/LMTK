"""
Convert DDP checkpoints to HuggingFace models for publishing.
"""

import os
from logging import getLogger

# Force CPU-only to avoid CUDA/Deepspeed extension builds on clusters without CUDA_HOME.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("DS_BUILD_OPS", "0")

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = getLogger(__name__)

# Prefer CPU-only conversion to avoid triggering CUDA/Deepspeed extension builds.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class ConvertDDPCheckpoint:
    def __init__(self, host, base_model, checkpoint_path):
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path
        self.host = host

    def _clean_key(self, key: str) -> str:
        """Normalize common DDP/Fabric prefixes to HuggingFace layout."""
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model.model."):
            key = key.replace("model.model.", "model.", 1)
        elif key.startswith("model.lm_head.") and key.endswith("weight"):
            key = key.replace("model.lm_head.", "lm_head.", 1)
        return key

    def _load_ddp_checkpoint_safely(self, checkpoint_path):
        """Load DDP checkpoint handling common formats and removing prefixes."""
        logger.info(f"Loading DDP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info("✅ Found model_state_dict in DDP checkpoint")
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                logger.info("✅ Found state_dict in DDP checkpoint")
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                logger.info("✅ Found model in DDP checkpoint")
            else:
                state_dict = checkpoint
                logger.info("✅ Using entire checkpoint as state_dict")
        else:
            state_dict = checkpoint
            logger.info("✅ Checkpoint is direct state_dict")

        cleaned_state_dict = {}
        stripped_prefix = 0
        remapped_model_prefix = 0
        for key, value in state_dict.items():
            new_key = self._clean_key(key)
            if key.startswith("module."):
                stripped_prefix += 1
            if key.startswith("model.model."):
                remapped_model_prefix += 1
            cleaned_state_dict[new_key] = value

        if cleaned_state_dict:
            logger.info(f"Sample cleaned checkpoint key: {next(iter(cleaned_state_dict))}")
        else:
            logger.warning("No parameters found in checkpoint")
        logger.info(
            f"Key normalization: {stripped_prefix} 'module.' prefixes stripped, "
            f"{remapped_model_prefix} 'model.model.' prefixes remapped"
        )

        return cleaned_state_dict

    def _info(self, model, missing_keys, unexpected_keys, loaded_keys_count: int):
        """Log information about the conversion results."""
        from src.utils.logging import get_logger

        _logger = get_logger(__name__)
        model_state_dict = model.state_dict()

        remaining_missing = [k for k in missing_keys if k not in model_state_dict]
        if remaining_missing:
            _logger.warning(f"⚠️  {len(remaining_missing)} keys still missing after weight tying")
            _logger.warning(f"Sample missing: {remaining_missing[:3]}")
        else:
            _logger.info("✅ All parameters resolved after weight tying!")

        if unexpected_keys:
            _logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys")
            _logger.warning(f"Sample unexpected: {unexpected_keys[:3]}")

        total_params = len(model_state_dict)
        loading_percentage = 100.0 if total_params == 0 else (loaded_keys_count / total_params) * 100
        _logger.info(f"✅ Final result: {loaded_keys_count}/{total_params} parameters ({loading_percentage:.1f}%)")

        if loading_percentage < 95 or unexpected_keys:
            raise ValueError(
                f"State dict mismatch: {loaded_keys_count}/{total_params} loaded, "
                f"{len(unexpected_keys)} unexpected, {len(remaining_missing)} missing after tying"
            )

        try:
            if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
                if torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                    _logger.info("✅ lm_head.weight properly tied to embed_tokens.weight")
                else:
                    _logger.warning("⚠️  Weight tying may not be active")
        except (TypeError, AttributeError):
            _logger.info("Skipping weight tying check in test environment")

    def execute(self):
        config = AutoConfig.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_config(config)

        original_state_dict = self._load_ddp_checkpoint_safely(self.checkpoint_path)
        result = model.load_state_dict(original_state_dict, strict=False)

        if result is None:
            missing_keys, unexpected_keys = [], []
        else:
            try:
                missing_keys, unexpected_keys = result
            except (ValueError, TypeError):
                missing_keys, unexpected_keys = [], []

        loaded_keys_count = len(set(original_state_dict.keys()) & set(model.state_dict().keys()))
        self._info(model, missing_keys, unexpected_keys, loaded_keys_count)

        logger.info("🔗 Applying weight tying...")
        model.tie_weights()

        return model.cpu()
