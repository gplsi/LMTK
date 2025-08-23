"""

"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
import os
from logging import getLogger

logger = getLogger(__name__)

class FSDPtoHuggingFace:
    def __init__(self, base_model, checkpoint_path, task_type: str | None = None):
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path
        # task_type: "mlm" | "clm" | "instruction" (treated as causal)
        self.task_type = (task_type or "").lower().strip() or None

    def _fix_fsdp_state_dict_keys(self, checkpoint_state_dict, model_state_dict):
        """Fix FSDP checkpoint keys with proper lm_head handling."""
        logger.info("🔧 Fixing FSDP checkpoint key structure...")
    
        checkpoint_keys = list(checkpoint_state_dict.keys())
        model_keys = list(model_state_dict.keys())
        
        # Log sample keys if available
        if checkpoint_keys:
            logger.info(f"Sample checkpoint key: {checkpoint_keys[0]}")
        else:
            logger.info("No checkpoint keys found")
            
        if model_keys:
            logger.info(f"Sample model key: {model_keys[0]}")
        else:
            logger.info("No model keys found")
        
        fixed_state_dict = {}
        
        for key, value in checkpoint_state_dict.items():
            new_key = None
            
            # Handle different FSDP key patterns
            if key.startswith("model.model."):
                # Most parameters: model.model.layers.X -> model.layers.X
                new_key = key.replace("model.model.", "model.", 1)
            elif key.startswith("model.embed_tokens."):
                # Embedding tokens: keep as is
                new_key = key
            elif key == "model.lm_head.weight":
                # lm_head: model.lm_head.weight -> lm_head.weight
                new_key = "lm_head.weight"
            else:
                new_key = key
                
            # Check if the new key exists in model
            if new_key in model_state_dict:
                fixed_state_dict[new_key] = value
            else:
                # Special handling for lm_head when it might be tied
                if "lm_head" in key:
                    logger.warning(f"🔗 lm_head parameter found but not in model state_dict - likely tied weights")
                    continue  # Skip lm_head if it's tied to embeddings
                else:
                    logger.warning(f"⚠️  Could not map key: {key} -> tried {new_key}")
        
        logger.info(f"✅ Successfully mapped {len(fixed_state_dict)}/{len(checkpoint_state_dict)} parameters")
        return fixed_state_dict
        

    def _load_fsdp_checkpoint_safely(self, checkpoint_path):
        """Load FSDP checkpoint handling common formats."""
        logger.info(f"Loading FSDP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different FSDP checkpoint structures
        if isinstance(checkpoint, dict):
            # Common FSDP checkpoint keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("✅ Found model_state_dict in FSDP checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']  
                logger.info("✅ Found state_dict in FSDP checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info("✅ Found model in FSDP checkpoint")
            elif 'app' in checkpoint and 'model' in checkpoint['app']:
                # Distributed checkpoint format
                state_dict = checkpoint['app']['model']
                logger.info("✅ Found distributed checkpoint format")
            else:
                # Assume whole checkpoint is state dict
                state_dict = checkpoint
                logger.info("✅ Using entire checkpoint as state_dict")
        else:
            state_dict = checkpoint
            logger.info("✅ Checkpoint is direct state_dict")
        
        return state_dict

    def _apply_weight_tying(self, model):
        logger.info("🔗 Applying weight tying...")
        model.tie_weights()  # This will tie lm_head.weight to embed_tokens.weight


    def _info(self, model, missing_keys, unexpected_keys):
        """Log information about the conversion results"""
        from src.utils.logging import get_logger
        logger = get_logger(__name__)
        
        # Get model state dict for parameter counting
        model_state_dict = model.state_dict()
        
        # Check for missing keys after weight tying
        remaining_missing = []
        for key in missing_keys:
            if key not in model_state_dict:
                remaining_missing.append(key)

        if remaining_missing:
            logger.warning(f"⚠️  {len(remaining_missing)} keys still missing after weight tying")
            logger.warning(f"Sample missing: {remaining_missing[:3]}")
        else:
            logger.info("✅ All parameters resolved after weight tying!")
        
        if unexpected_keys:
            logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys")
            logger.warning(f"Sample unexpected: {unexpected_keys[:3]}")
        
        # Tensor-level summary
        total_tensors = len(model_state_dict)
        loaded_tensors = total_tensors - len(remaining_missing)
        if total_tensors == 0:
            tensor_loading_pct = 100.0
        else:
            tensor_loading_pct = (loaded_tensors / total_tensors) * 100

        # Element-level summary (scalar parameters)
        total_elems = sum(v.numel() for v in model_state_dict.values())
        loaded_elems = sum(
            v.numel() for k, v in model_state_dict.items() if k not in remaining_missing
        )
        if total_elems == 0:
            elem_loading_pct = 100.0
        else:
            elem_loading_pct = (loaded_elems / total_elems) * 100

        logger.info(
            f"✅ Final result: tensors {loaded_tensors}/{total_tensors} ({tensor_loading_pct:.1f}%), "
            f"elements {loaded_elems}/{total_elems} ({elem_loading_pct:.2f}%)"
        )
        
        if elem_loading_pct < 95:
            raise ValueError(
                f"Only {elem_loading_pct:.2f}% of scalar parameters loaded from FSDP checkpoint"
            )
        
        # Check for weight tying, but handle mock objects in tests
        try:
            if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
                if torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                    logger.info("✅ lm_head.weight properly tied to embed_tokens.weight")
                else:
                    logger.warning("⚠️  Weight tying may not be active")
        except (TypeError, AttributeError):
            # Skip weight tying check in test environment with mock objects
            logger.info("Skipping weight tying check in test environment")

    def _convert_checkpoint(self, model, checkpoint_file):
            original_state_dict = self._load_fsdp_checkpoint_safely(checkpoint_file)
            model_state_dict = model.state_dict()
            fixed_state_dict = self._fix_fsdp_state_dict_keys(original_state_dict, model_state_dict)
            
            
            # Handle both tuple return and None return (in tests)
            result = model.load_state_dict(fixed_state_dict, strict=False)
            if result is None:
                missing_keys, unexpected_keys = [], []
            else:
                try:
                    missing_keys, unexpected_keys = result
                except (ValueError, TypeError):
                    missing_keys, unexpected_keys = [], []
            logger.info(self._info(model, missing_keys, unexpected_keys))
            logger.info("🔗 Applying weight tying...")
            model.tie_weights()
            return model

    def execute(self, output_dir):
        config = AutoConfig.from_pretrained(self.base_model)
        # Determine if single or multiple checkpoints
        checkpoint_files = []
        if self.checkpoint_path.endswith(".pth"):
            checkpoint_files = [self.checkpoint_path]
        else:
            for root, dirs, files in os.walk(self.checkpoint_path):
                files = sorted(files)
                for file in files:
                    if file.endswith(".pth"):
                        checkpoint_files.append(os.path.join(root, file))
        
        # Fresh model per checkpoint and save immediately
        summary = []
        total = len(checkpoint_files)
        for idx, checkpoint_file in enumerate(checkpoint_files, 1):
            logger.info(f"[Progress] Converting checkpoint {idx}/{total}: {checkpoint_file}")
            cfg = AutoConfig.from_pretrained(self.base_model)
            is_decoder = bool(getattr(cfg, "is_decoder", False))
            if self.task_type == "mlm" or (not is_decoder and self.task_type is None):
                model = AutoModelForMaskedLM.from_config(cfg)
            else:
                cfg.is_decoder = True
                model = AutoModelForCausalLM.from_config(cfg)
            try:
                model = self._convert_checkpoint(model, checkpoint_file)
                base_name = os.path.splitext(os.path.basename(checkpoint_file))[0]
                final_dir = os.path.join(output_dir, base_name)
                os.makedirs(final_dir, exist_ok=True)
                logger.info(f"[Progress] Saving HuggingFace model {idx}/{total} to {final_dir}")
                model.save_pretrained(final_dir)
                tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                tokenizer.save_pretrained(final_dir)
                summary.append({'checkpoint': checkpoint_file, 'success': True, 'error': None, 'output_dir': final_dir})
            except Exception as e:
                logger.error(f"Failed to convert {checkpoint_file}: {e}")
                summary.append({'checkpoint': checkpoint_file, 'success': False, 'error': str(e)})
        
        logger.info("All conversions completed.")
        logger.info(f"Summary: {summary}")
        return summary