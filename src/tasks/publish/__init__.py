from box import Box
from src.tasks.publish.orchestrator import PublishOrchestrator


# src/tasks/publish.py
def execute(config: Box):
    orchestrator = PublishOrchestrator(config)
    return orchestrator.execute()