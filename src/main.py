# Add NVML patching at the earliest possible point
import os
import sys

# Set critical environment variables
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Import and apply our PyTorch patches
from src.utils.torch_patches import apply_all_patches
apply_all_patches()

# Continue with the rest of the imports and code
import argparse
from box import Box
from src.config.config_loader import ConfigLoader
import yaml

def execute_task(config_path: str):
    # Load the YAML file (to extract the task value)
    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f)
    raw_config = Box(raw_data, box_dots=True)

    # Ensure the 'task' key exists.
    task = raw_config.get("task")
    if not task:
        raise ValueError("Missing 'task' key in configuration.")

    # Validate using the task-specific schema (which, if desired, may include the base fields)
    validator = ConfigValidator()
    config = validator.validate(config_path, task)

    # Dynamically import and dispatch to the task handler.
    task_module = __import__(f"tasks.{task}", fromlist=[""])
    task_module.execute(config)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to experiment config')
    args = parser.parse_args()
    execute_task(args.config)
    #execute_task("config/experiments/test_tokenizer_local.yaml")
    
    
    # FOR CURRENT GPT-2 TESTING
    #execute_task("config/experiments/test_tokenizer_local.yaml")
    # execute_task("config/experiments/test_continual.yaml")

    # For deepspeed testing
    execute_task("config/experiments/continual_gpt-2_deepspeed.yaml")