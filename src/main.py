from box import Box
import yaml
import sys
import os
import logging
from pathlib import Path
from src.config.config_loader import ConfigValidator
from src.utils.version import display_version_info
from src.utils.logging import get_logger, setup_logging, VerboseLevel


logger = get_logger(__name__)


def execute_task(config_path: str):
    """Execute a task based on the provided configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Raises:
        ValueError: If the task is missing or invalid
        ImportError: If the task module cannot be imported
        Exception: For any other errors during task execution
    """
    # Load the YAML file (to extract the task value)
    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f)
    raw_config = Box(raw_data, box_dots=True)

    # Ensure the 'task' key exists
    task = raw_config.get("task")
    if not task:
        raise ValueError("Missing 'task' key in configuration.")
        
    # Set up logging based on configuration
    verbose_level = raw_config.get("verbose_level", VerboseLevel.INFO)
    setup_logging(verbose_level)
    
    # Log the task being executed
    logger.info(f"Executing task: {task}")
    logger.info(f"Using configuration file: {config_path}")

    # Log framework and strategy if available (for training tasks)
    framework = raw_config.get("framework")
    strategy = raw_config.get("strategy")
    
    # Only log framework and strategy for training tasks that have them
    if task in ["pretraining", "instruction"]:
        if framework:
            logger.info(f"Using framework: {framework}")
        if strategy:
            logger.info(f"Using strategy: {strategy}")

    # Determine the appropriate schema name based on task type
    if task in ["pretraining", "instruction"]:
        # For training tasks, use the base training schema
        schema_name = "training"
    else:
        # For non-training tasks, use the task name directly
        schema_name = task

    # Validate using the appropriate schema
    validator = ConfigValidator()
    try:
        config = validator.validate(config_path, schema_name)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = config.get("output_dir", f"outputs/{config.experiment_name}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Dynamically import and dispatch to the task handler
    try:
        task_module = __import__(f"tasks.{task}", fromlist=[""])
        logger.info(f"Successfully imported task module: tasks.{task}")
        task_module.execute(config)
    except ImportError as e:
        logger.error(f"Failed to import task module: tasks.{task}")
        raise ImportError(f"Task '{task}' is not implemented or could not be imported: {e}")
    except Exception as e:
        logger.error(f"Error executing task '{task}': {e}")
        raise


def list_available_tasks():
    """List all available tasks in the framework.
    
    Returns:
        List of available task names
    """
    tasks_dir = Path(__file__).parent / "tasks"
    tasks = []
    
    for item in tasks_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__") and (item / "__init__.py").exists():
            tasks.append(item.name)
    
    return sorted(tasks)


def validate_config(config_path: str, verbose: bool = False):
    """Validate a configuration file without executing the task.
    
    Args:
        config_path: Path to the configuration file
        verbose: Whether to print verbose output
        
    Returns:
        True if the configuration is valid, False otherwise
    """
    try:
        # Load the configuration
        with open(config_path, "r") as f:
            raw_data = yaml.safe_load(f)
        raw_config = Box(raw_data, box_dots=True)
        
        # Check for task key
        task = raw_config.get("task")
        if not task:
            logger.error("Missing 'task' key in configuration.")
            return False
        
        # Log framework and strategy if available (for training tasks)
        framework = raw_config.get("framework")
        strategy = raw_config.get("strategy")
        
        # Extract task, framework, and strategy for logging
        task = raw_config.get("task")
        framework = raw_config.get("framework")
        strategy = raw_config.get("strategy")
        
        if not task:
            logger.error("Task not specified in configuration file.")
            sys.exit(1)
        
        logger.info(f"Executing task: {task}")
        if framework:
            logger.info(f"Using framework: {framework}")
        if strategy:
            logger.info(f"Using strategy: {strategy}")
        
        # For training tasks, use the base training schema
        if task in ["pretraining", "instruction"]:
            schema_name = "training"
        else:
            # For non-training tasks, use the task name directly
            schema_name = task
        
        # Validate the configuration
        validator = ConfigValidator()
        try:
            validator.validate(config_path, schema_name)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
        
        if verbose:
            logger.info(f"Configuration is valid: {config_path}")
            logger.info(f"Task: {task}")
            if framework:
                logger.info(f"Framework: {framework}")
            if strategy:
                logger.info(f"Strategy: {strategy}")
            logger.info(f"Experiment name: {raw_config.get('experiment_name', 'Not specified')}")
        
        return True
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Continual Pretraining Framework')
    parser.add_argument('--config', '-c', help='Path to experiment config')
    parser.add_argument('--version', '-v', action='store_true', help='Display version information')
    parser.add_argument('--validate', action='store_true', help='Validate configuration without execution')
    parser.add_argument('--list-tasks', '-l', action='store_true', help='List available tasks')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Set up basic logging
    setup_logging(VerboseLevel.INFO if args.verbose else VerboseLevel.WARN)
    
    # Handle version command
    if args.version:
        display_version_info()
        sys.exit(0)
    
    # Handle list tasks command
    if args.list_tasks:
        tasks = list_available_tasks()
        print("Available tasks:")
        for task in tasks:
            print(f"  - {task}")
        sys.exit(0)
        
    # Ensure a config is provided when needed
    if not args.config and not args.list_tasks and not args.version:
        parser.print_help()
        sys.exit(1)
        
    # Handle validation only
    if args.validate:
        logger.info(f"Validating configuration: {args.config}")
        if validate_config(args.config, args.verbose):
            logger.info("Configuration is valid!")
            sys.exit(0)
        else:
            logger.error("Configuration validation failed.")
            sys.exit(1)
    
    # Execute the task
    if args.config:
        try:
            execute_task(args.config)
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except ImportError as e:
            logger.error(f"Task import error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Execution error: {e}")
            sys.exit(1)
