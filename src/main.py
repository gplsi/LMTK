from utils.config_loader import ConfigValidator

def execute_task(config_path: str):
    validator = ConfigValidator()
    
    # Load base schema first
    base_config = validator.validate(config_path, 'base')
    
    # Load task-specific schema
    full_config = validator.validate(config_path, base_config.task)
    
    # Dispatch to appropriate task handler
    task_module = __import__(f'tasks.{base_config.task}', fromlist=[''])
    task_module.execute(full_config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to experiment config')
    args = parser.parse_args()
    execute_task(args.config)
