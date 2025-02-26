from box import Box
import yaml
from src.config.config_loader import ConfigValidator


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
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('config', help='Path to experiment config')
    #args = parser.parse_args()
    #execute_task(args.config)
    #execute_task("config/experiments/test_tokenizer_local.yaml")
    
    
    # FOR CURRENT GPT-2 TESTING
    #execute_task("config/experiments/test_tokenizer_local.yaml")
    execute_task("config/experiments/test_continual.yaml")