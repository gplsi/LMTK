from box import Box
from src.tasks.dataset_merge.orchestrator import DatasetMergeOrchestrator


def execute(config: Box):
    orchestrator = DatasetMergeOrchestrator(config)
    return orchestrator.execute()
