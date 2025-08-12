from box import Box
from src.tasks.tokenization.orchestrator import TokenizationOrchestrator


# src/tasks/tokenization.py
def execute(config: Box):
    orchestrator = TokenizationOrchestrator(config)
    return orchestrator.execute()