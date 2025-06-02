"""

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
import os
from logging import getLogger

logger = getLogger(__name__)

class ConvertFSDPCheckpoint:
    def __init__(self, host, base_model, checkpoint_path):
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path
        self.host = host

    def _fix_fsdp_state_dict_keys(self, checkpoint_state_dict, model_state_dict):
        """Fix FSDP checkpoint keys with proper lm_head handling."""
        logger.info("🔧 Fixing FSDP checkpoint key structure...")
    
        checkpoint_keys = list(checkpoint_state_dict.keys())
        model_keys = list(model_state_dict.keys())
        
        logger.info(f"Sample checkpoint key: {checkpoint_keys[0]}")
        logger.info(f"Sample model key: {model_keys[0]}")
        
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
        remaining_missing = []
        for key in missing_keys:
            if key not in model.state_dict():
                remaining_missing.append(key)

        if remaining_missing:
            logger.warning(f"⚠️  {len(remaining_missing)} keys still missing after weight tying")
            logger.warning(f"Sample missing: {remaining_missing[:3]}")
        else:
            logger.info("✅ All parameters resolved after weight tying!")
        
        if unexpected_keys:
            logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys")
            logger.warning(f"Sample unexpected: {unexpected_keys[:3]}")
        
        total_params = len(model_state_dict)
        loaded_params = total_params - len(remaining_missing)
        loading_percentage = (loaded_params / total_params) * 100
        
        logger.info(f"✅ Final result: {loaded_params}/{total_params} parameters ({loading_percentage:.1f}%)")
        
        if loading_percentage < 95:
            raise ValueError(f"Only {loading_percentage:.1f}% of parameters loaded from FSDP checkpoint")
        
        if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
            if torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                logger.info("✅ lm_head.weight properly tied to embed_tokens.weight")
            else:
                logger.warning("⚠️  Weight tying may not be active")



    def execute(self):
        config = AutoConfig.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_config(config)
        
        original_state_dict = self._load_fsdp_checkpoint_safely(self.checkpoint_path)
        model_state_dict = model.state_dict()
        fixed_state_dict = self._fix_fsdp_state_dict_keys(original_state_dict, model_state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        self._info(model, missing_keys, unexpected_keys)

        logger.info("🔗 Applying weight tying...")
        model.tie_weights()  # This will tie lm_head.weight to embed_tokens.weight
        
        return model
        

def main():
    try:
        # 1. Load config & create model
        print(f"Loading configuration from {BASE_MODEL}...")
        config = AutoConfig.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_config(config)
        
        # 2. Load FSDP checkpoint
        original_state_dict = load_fsdp_checkpoint_safely(CHECKPOINT_PATH)
        
        # 3. Fix FSDP key mapping
        model_state_dict = model.state_dict()
        fixed_state_dict = fix_fsdp_state_dict_keys(original_state_dict, model_state_dict)
        
        # 4. Load weights
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        
        # 5. CRITICAL: Apply weight tying for lm_head
        print("🔗 Applying weight tying...")
        model.tie_weights()  # This will tie lm_head.weight to embed_tokens.weight
        
        # 6. Re-check missing keys after weight tying
        remaining_missing = []
        for key in missing_keys:
            if key not in model.state_dict():
                remaining_missing.append(key)
        
        if remaining_missing:
            print(f"⚠️  {len(remaining_missing)} keys still missing after weight tying")
            print(f"Sample missing: {remaining_missing[:3]}")
        else:
            print("✅ All parameters resolved after weight tying!")
        
        if unexpected_keys:
            print(f"⚠️  {len(unexpected_keys)} unexpected keys")
            print(f"Sample unexpected: {unexpected_keys[:3]}")
        
        # Calculate final loading success
        total_params = len(model_state_dict)
        loaded_params = total_params - len(remaining_missing)
        loading_percentage = (loaded_params / total_params) * 100
        
        print(f"✅ Final result: {loaded_params}/{total_params} parameters ({loading_percentage:.1f}%)")
        
        if loading_percentage < 95:
            raise ValueError(f"Only {loading_percentage:.1f}% of parameters loaded from FSDP checkpoint")
        
        # 7. Verify lm_head is properly tied
        print("🔍 Verifying weight tying...")
        if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
            if torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                print("✅ lm_head.weight properly tied to embed_tokens.weight")
            else:
                print("⚠️  Weight tying may not be active")
        
        # 8. Load tokenizer
        print(f"Loading tokenizer from {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # 9. Upload to Hub
        print("Uploading FSDP-trained model to Hub...")
        model.push_to_hub(
            REPO_ID,
            commit_message="Add FSDP continually pre-trained Cecilia weights (epoch 2) with proper weight tying",
            max_shard_size="5GB",
            safe_serialization=True,
            create_pr=False
        )
        
        tokenizer.push_to_hub(
            REPO_ID,
            commit_message="Add Salamandra-2b compatible tokenizer"
        )
        
        print("🚀 FSDP model uploaded successfully!")
        
        # 10. Validate upload
        print("\n🧪 Validating uploaded model...")
        test_model = AutoModelForCausalLM.from_pretrained(REPO_ID)
        test_tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        
        # Test functionality
        test_input = test_tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            output = test_model(**test_input)
        
        print("✅ FSDP model validation successful!")
        print(f"✅ Model has {test_model.num_parameters():,} parameters")
        print(f"Model available at: https://huggingface.co/{REPO_ID}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing FSDP checkpoint: {e}")
        raise

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)