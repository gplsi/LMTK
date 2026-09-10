from box import Box
from src.tasks.anonymization.orchestrator import AnonymizationOrchestrator


# src/tasks/anonimize.py
def execute(config: Box):
    orchestrator = AnonymizationOrchestrator(config)
    return orchestrator.execute()