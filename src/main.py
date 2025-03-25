# Disable NVML initialization BEFORE importing any other modules
import os
# Set critical environment variables to disable NVML functions
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NVML_SKIP_INIT"] = "1"  # Skip NVML initialization completely
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["DS_SKIP_NVML_INIT"] = "1"  # DeepSpeed-specific NVML skip

# Continue with imports
import sys
import argparse
from box import Box
from dotenv import load_dotenv
import wandb
import yaml
from src.config.config_loader import ConfigValidator

# Import our patching utility after setting environment variables
from src.utils.torch_patches import apply_all_patches


load_dotenv()  # This loads the variables from .env
wandb.login(key=os.getenv('WANDB_API_KEY'))


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
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to experiment config')
    args = parser.parse_args()
    execute_task(args.config)
