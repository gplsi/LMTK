from box import Box
from src.tasks.continual.orchestrator import ContinualOrchestrator

# src/tasks/continual.py
def execute(config: Box):
    orchestrator = ContinualOrchestrator(config)
    return orchestrator.execute()