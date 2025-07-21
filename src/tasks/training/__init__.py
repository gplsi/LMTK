from box import Box

# src/tasks/continual.py
def execute(config: Box):
    from src.tasks.clm_training.orchestrator import ContinualOrchestrator
    orchestrator = ContinualOrchestrator(config)
    return orchestrator.execute()