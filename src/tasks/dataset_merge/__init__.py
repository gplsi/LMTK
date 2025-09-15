"""
Clean Dataset Merge Module

Provides a modular, efficient approach to merging datasets with proper
separation of concerns across compatibility checking, sampling, and merging.
"""

from box import Box
from src.tasks.dataset_merge.orchestrator import DatasetMergeOrchestrator
from src.tasks.dataset_merge.compatibility import DatasetCompatibilityChecker
from src.tasks.dataset_merge.sampling import DatasetSampler
from src.tasks.dataset_merge.merging import DatasetMerger


def execute(config: Box):
    """Main entry point for dataset merge operations."""
    orchestrator = DatasetMergeOrchestrator(config)
    return orchestrator.execute()


__all__ = [
    'execute',
    'DatasetMergeOrchestrator',
    'DatasetCompatibilityChecker', 
    'DatasetSampler',
    'DatasetMerger'
]
