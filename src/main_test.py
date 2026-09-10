# Set environment variables for HuggingFace cache directories to use workspace
import os
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/datasets"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/transformers"

from box import Box
import yaml
import sys
from src.config.config_loader import ConfigValidator
from src.utils.version import display_version_info


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
    from pathlib import Path
    config = validator.validate(Path(config_path), task)

    # Map task types to their corresponding modules
    task_module_map = {
        'clm_training': 'training',
        'mlm_training': 'training', 
        'instruction': 'training',
        'tokenization': 'tokenization',
        'publish': 'publish'
    }
    
    # Get the module name for this task type
    module_name = task_module_map.get(task, task)
    
    # Dynamically import and dispatch to the task handler
    task_module = __import__(f"tasks.{module_name}", fromlist=[""])
    task_module.execute(config)
    
if __name__ == '__main__': 
    # Execute the task
    try:
        execute_task(config_path="/workspace/config/examples/tokenization_instruction_v5_example.yaml")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
