from box import Box
import importlib

# Ensure we're using the latest version of the orchestrator
importlib.reload(importlib.import_module('src.tasks.pretraining.orchestrator'))
from src.tasks.pretraining.orchestrator import ContinualOrchestrator

# src/tasks/continual.py
def execute(config: Box):
    # Initialize the orchestrator with the configuration
    orchestrator = ContinualOrchestrator(config)
    # Execute the orchestrator's main method
    return orchestrator.execute()