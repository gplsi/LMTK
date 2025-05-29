"""
Abstract training task module.

This module provides the foundation for all training tasks,
including pretraining and instruction fine-tuning.
"""

from src.abstract_tasks.training.orchestrator import TrainingOrchestrator
from src.abstract_tasks.training.trainer import TrainerBase

__all__ = ["TrainingOrchestrator", "TrainerBase"]
