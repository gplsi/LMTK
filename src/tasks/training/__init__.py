from box import Box

# src/tasks/training.py
def execute(config: Box):
    from src.tasks.training.orchestrator import ContinualOrchestrator
    orchestrator = ContinualOrchestrator(config)
    return orchestrator.execute()