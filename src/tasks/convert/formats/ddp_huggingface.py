"""
Conversion utilities to move DDP checkpoints into HuggingFace format.

The implementation mirrors the FSDP converter behavior (directory scanning,
summary reporting, weight tying) but avoids FSDP-specific key rewriting.
"""

import os
from logging import getLogger

# Force CPU-only to avoid CUDA/Deepspeed extension builds on clusters without CUDA_HOME.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("DS_BUILD_OPS", "0")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = getLogger(__name__)

# Avoid accidental GPU/deepspeed initialisation when converting checkpoints.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class DDPtoHuggingFace:
    def __init__(self, base_model, checkpoint_path):
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path

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
        """Load a DDP checkpoint handling common structures and stripping prefixes."""
        logger.info(f"Loading DDP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle typical checkpoint layouts
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
            sample_key = next(iter(cleaned_state_dict))
            logger.info(f"Sample cleaned checkpoint key: {sample_key}")
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

    def _convert_checkpoint(self, checkpoint_file):
        """Convert a single checkpoint file to HuggingFace format."""
        config = AutoConfig.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_config(config)

        original_state_dict = self._load_ddp_checkpoint_safely(checkpoint_file)
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
        return model

    def execute(self, output_dir):
        checkpoint_files = []
        if os.path.isfile(self.checkpoint_path) and self.checkpoint_path.endswith(".pth"):
            checkpoint_files = [self.checkpoint_path]
            logger.info(f"Found single checkpoint file: {self.checkpoint_path}")
        elif os.path.isdir(self.checkpoint_path):
            logger.info(f"Scanning directory for checkpoint files: {self.checkpoint_path}")
            for root, _, files in os.walk(self.checkpoint_path):
                for file in files:
                    if file.endswith(".pth"):
                        full_path = os.path.join(root, file)
                        checkpoint_files.append(full_path)
                        logger.info(f"Discovered checkpoint: {full_path}")
            if not checkpoint_files:
                logger.warning(f"No .pth checkpoint files found in directory: {self.checkpoint_path}")
            else:
                checkpoint_files.sort()
                logger.info(f"Total checkpoints discovered: {len(checkpoint_files)}")
        else:
            raise ValueError(f"Checkpoint path does not exist or is not a file/directory: {self.checkpoint_path}")

        summary = []
        total = len(checkpoint_files)
        if total == 0:
            logger.warning("No checkpoint files to convert")
            return summary

        results = []
        for idx, checkpoint_file in enumerate(checkpoint_files, 1):
            logger.info(f"[Progress] Converting checkpoint {idx}/{total}: {checkpoint_file}")
            try:
                model = self._convert_checkpoint(checkpoint_file)
                success, error = True, None
            except Exception as e:
                logger.error(f"Failed to convert {checkpoint_file}: {e}")
                model, success, error = None, False, str(e)
            results.append(model)
            summary.append({"checkpoint": checkpoint_file, "success": success, "error": error})

        logger.info(f"Saving converted models to output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        successful_conversions = 0
        failed_conversions = 0

        for idx, (checkpoint_file, model) in enumerate(zip(checkpoint_files, results), 1):
            if model is not None:
                relative_path = os.path.relpath(
                    checkpoint_file,
                    self.checkpoint_path if os.path.isdir(self.checkpoint_path) else os.path.dirname(self.checkpoint_path),
                )
                base_name = os.path.splitext(relative_path)[0].replace(os.sep, "_")
                final_dir = os.path.join(output_dir, base_name)

                logger.info(f"[Progress] Saving HuggingFace model {idx}/{total}: {os.path.basename(checkpoint_file)} -> {final_dir}")
                try:
                    os.makedirs(final_dir, exist_ok=True)
                    # stay on CPU to avoid DS/extension build; saving is CPU friendly
                    model_cpu = model.cpu()
                    model_cpu.save_pretrained(final_dir)
                    tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                    tokenizer.save_pretrained(final_dir)

                    for entry in summary:
                        if entry["checkpoint"] == checkpoint_file:
                            entry["output_dir"] = final_dir

                    successful_conversions += 1
                    logger.info(f"✅ Successfully saved model to {final_dir}")
                except Exception as e:
                    logger.error(f"❌ Failed to save model for {checkpoint_file}: {e}")
                    for entry in summary:
                        if entry["checkpoint"] == checkpoint_file:
                            entry["success"] = False
                            entry["error"] = str(e)
                    failed_conversions += 1
            else:
                logger.warning(f"⏭️  Skipping saving for {checkpoint_file} due to previous conversion errors.")
                failed_conversions += 1

        logger.info("=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total checkpoints found: {total}")
        logger.info(f"Successfully converted: {successful_conversions}")
        logger.info(f"Failed conversions: {failed_conversions}")
        logger.info(f"Output directory: {output_dir}")

        if successful_conversions > 0:
            logger.info("✅ Conversion completed with at least some successes")
        else:
            logger.error("❌ All conversions failed")

        logger.info("=" * 60)
        return summary
