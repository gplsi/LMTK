"""
CSV logging utility for instruction fine-tuning with Lightning Fabric.

This module provides a CSV logger for tracking training metrics during
instruction fine-tuning with Lightning Fabric.
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, TextIO
import lightning as L
from lightning.fabric.loggers import CSVLogger


def step_csv_logger(
    output_dir: str,
    model_name: str,
    flush_logs_every_n_steps: int = 10,
) -> CSVLogger:
    """
    Create a CSV logger for instruction fine-tuning metrics.
    
    Args:
        output_dir: Directory to save logs
        model_name: Name of the model being trained
        flush_logs_every_n_steps: How often to flush logs to disk
        
    Returns:
        A configured CSVLogger instance
    """
    # Create a sanitized model name for the log file
    model_name_safe = model_name.replace("/", "_").replace(":", "_")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = CSVLogger(
        root_dir=output_dir,
        name=f"instruction_tuning_{model_name_safe}",
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )
    
    return logger


class InstructionMetricsLogger:
    """
    Custom metrics logger for instruction fine-tuning.
    
    This class provides additional metrics tracking specifically for
    instruction fine-tuning, such as per-task performance metrics.
    """
    
    def __init__(self, log_dir: str, model_name: str) -> None:
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save logs
            model_name: Name of the model being trained
        """
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task-specific log file
        self.task_log_path = self.log_dir / f"{model_name}_task_metrics.csv"
        self.task_log_file: Optional[TextIO] = None
        self.task_csv_writer = None
        
        # Initialize task log file
        self._init_task_log()
    
    def _init_task_log(self) -> None:
        """Initialize the task-specific log file with headers."""
        is_new_file = not self.task_log_path.exists()
        
        self.task_log_file = open(self.task_log_path, "a", newline="")
        self.task_csv_writer = csv.writer(self.task_log_file)
        
        if is_new_file:
            self.task_csv_writer.writerow([
                "timestamp", "step", "task_type", "accuracy", "loss", "samples"
            ])
    
    def log_task_metrics(
        self, 
        step: int, 
        task_type: str, 
        metrics: Dict[str, float], 
        samples: int
    ) -> None:
        """
        Log metrics for a specific task type.
        
        Args:
            step: Current training step
            task_type: Type of instruction task (e.g., "summarize", "translate")
            metrics: Dictionary of metrics for this task
            samples: Number of samples for this task
        """
        if self.task_csv_writer is None:
            self._init_task_log()
        
        timestamp = time.time()
        accuracy = metrics.get("accuracy", 0.0)
        loss = metrics.get("loss", 0.0)
        
        self.task_csv_writer.writerow([
            timestamp, step, task_type, accuracy, loss, samples
        ])
        self.task_log_file.flush()
    
    def close(self) -> None:
        """Close all open log files."""
        if self.task_log_file is not None:
            self.task_log_file.close()
            self.task_log_file = None
            self.task_csv_writer = None
