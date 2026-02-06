from box import Box

from src.tasks.dataset_analysis.orchestrator import DatasetAnalysisOrchestrator


def execute(config: Box):
    orchestrator = DatasetAnalysisOrchestrator(config)
    return orchestrator.execute()
