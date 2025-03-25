# Developer Guide

## Adding New Tasks

This guide explains how to implement new tasks in the LM Continual Pretraining Framework.

## Task Implementation Pattern

Every task in the framework follows this structure:

```
src/tasks/your_task/
├── __init__.py          # Task entry point
├── orchestrator.py      # Task orchestration logic
└── components/          # Task-specific components
    ├── __init__.py
    └── your_components.py
```

### 1. Create Task Structure

```bash
mkdir -p src/tasks/your_task/components
touch src/tasks/your_task/__init__.py
touch src/tasks/your_task/orchestrator.py
```

### 2. Implement Task Entry Point

In `src/tasks/your_task/__init__.py`:

```python
from box import Box
from src.tasks.your_task.orchestrator import YourTaskOrchestrator

def execute(config: Box):
    orchestrator = YourTaskOrchestrator(config)
    return orchestrator.execute()
```

### 3. Create Task Orchestrator

The orchestrator follows the base pattern in `src/tasks/your_task/orchestrator.py`:

```python
from utils import inherit_init_params
from utils.orchestrator import BaseOrchestrator

@inherit_init_params
class YourTaskOrchestrator(BaseOrchestrator):
    """
    Orchestrates your task workflow.
    """
    def validate_config(self) -> None:
        """Validate task-specific configuration"""
        if not self.config.required_field:
            raise ValueError("required_field must be provided")
    
    def execute(self) -> None:
        """Execute the complete task workflow"""
        self.logger.info("Starting task workflow")
        try:
            # 1. Validate configuration
            self.validate_config()
            
            # 2. Load required resources
            resource = self.load_resource()
            
            # 3. Execute task logic
            result = self.process_resource(resource)
            
            # 4. Save outputs
            self.save_results(result)
            
            self.logger.info("Task completed successfully")
        except Exception as e:
            self.logger.error(f"Task failed: {str(e)}")
            raise
```

## Configuration System

### 1. Define Schema

Create a schema in `config/schemas/your_task.schema.yaml`:

```yaml
type: object
required:
  - task
  - output_dir
  - task_specific_field
properties:
  task:
    type: string
    const: your_task
  output_dir:
    type: string
    description: Output directory path
  task_specific_field:
    type: object
    description: Your task configuration
    properties:
      # Define task-specific properties
```

### 2. Register Schema

In `src/config/config_loader.py`, add your schema:

```python
TASK_SCHEMAS = {
    'your_task': 'your_task.schema.yaml',
    # ... existing schemas
}
```

### 3. Create Example Config

In `config/experiments/your_task.yaml`:

```yaml
task: your_task
output_dir: outputs/your_task
task_specific_field:
  # Task-specific configuration
```

## Testing New Tasks

1. Create Unit Tests:
```python
# tests/unit/test_your_task.py
@pytest.mark.unit
class TestYourTask:
    def test_task_validation(self):
        orchestrator = YourTaskOrchestrator(invalid_config)
        with pytest.raises(ValueError):
            orchestrator.validate_config()
```

2. Create Integration Test:
```python
# tests/integration/test_your_task_pipeline.py
@pytest.mark.integration
class TestYourTaskPipeline:
    def test_end_to_end(self):
        # Test complete task workflow
```

3. Add Test Grid Configuration:
```yaml
# config/test_grids/minimal_test_grid.yaml
your_task:
  - name: basic_test
    # Test configuration
```

## Best Practices

1. **Configuration Validation**
   - Always implement validate_config()
   - Check for required fields
   - Validate field types and values

2. **Error Handling**
   - Use try/except in execute()
   - Log errors before re-raising
   - Provide meaningful error messages

3. **Logging**
   - Use self.logger for consistency
   - Log start/end of major operations
   - Include relevant metrics/stats

4. **Resource Management**
   - Clean up resources in finally blocks
   - Use context managers when possible
   - Handle partial failures gracefully

5. **Testing**
   - Write unit tests for components
   - Include integration tests
   - Add to test grid system