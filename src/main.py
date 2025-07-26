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
    import argparse
    parser = argparse.ArgumentParser(description='Continual Pretraining Framework')
    parser.add_argument('--config', '-c', help='Path to experiment config')
    parser.add_argument('--version', '-v', action='store_true', help='Display version information')
    parser.add_argument('--validate', action='store_true', help='Validate configuration without execution')
    args = parser.parse_args()
    
    # Handle version command
    if args.version:
        display_version_info()
        sys.exit(0)
        
    # Ensure a config is provided when needed
    if not args.config:
        parser.print_help()
        sys.exit(1)
        
    # Handle validation only
    if args.validate:
        print(f"Validating configuration: {args.config}")
        validator = ConfigValidator()
        with open(args.config, "r") as f:
            raw_data = yaml.safe_load(f)
        raw_config = Box(raw_data, box_dots=True)
        task = raw_config.get("task")
        if not task:
            print("ERROR: Missing 'task' key in configuration.")
            sys.exit(1)
        try:
            validator.validate(args.config, task)
            print("Configuration is valid!")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: Invalid configuration: {e}")
            sys.exit(1)
    
    # Execute the task
    try:
        execute_task(args.config)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
