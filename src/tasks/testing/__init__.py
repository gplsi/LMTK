from box import Box
from src.tasks.testing.orchestrator import TestingOrchestrator


def execute(config: Box):
    orchestrator = TestingOrchestrator(config)
    return orchestrator.execute()
