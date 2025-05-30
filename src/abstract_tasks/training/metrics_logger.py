"""
Metrics logger for training tasks.

This module provides a unified interface for logging training metrics
across different backends (console, CSV, WandB, TensorBoard).
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import torch.nn as nn


class MetricsLogger:
    """Base class for logging training metrics."""
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        log_to_console: bool = True,
        log_to_csv: bool = True,
        log_to_wandb: bool = False,
        log_to_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_tags: Optional[List[str]] = None,
        tensorboard_log_dir: Optional[str] = None,
    ) -> None:
        """Initialize the metrics logger.
        
        Args:
            output_dir: Directory to save outputs
            experiment_name: Name of the experiment
            log_to_console: Whether to log to console
            log_to_csv: Whether to log to CSV
            log_to_wandb: Whether to log to Weights & Biases
            log_to_tensorboard: Whether to log to TensorBoard
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            wandb_config: Weights & Biases configuration
            wandb_tags: Weights & Biases tags
            tensorboard_log_dir: TensorBoard log directory
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_to_console = log_to_console
        self.log_to_csv = log_to_csv
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        
        self.csv_file = None
        self.csv_writer = None
        if self.log_to_csv:
            self._setup_csv_logger()
        
        self.wandb = None
        if self.log_to_wandb:
            self._setup_wandb_logger(
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_config=wandb_config,
                wandb_tags=wandb_tags,
            )
        
        self.tensorboard_writer = None
        if self.log_to_tensorboard:
            self._setup_tensorboard_logger(tensorboard_log_dir)
    
    def _setup_csv_logger(self) -> None:
        """
        Set up the CSV logger.
        """
        csv_path = os.path.join(self.output_dir, f"{self.experiment_name}_metrics.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_header_written = False
    
    def _setup_wandb_logger(
        self,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_tags: Optional[List[str]] = None,
    ) -> None:
        """
        Set up the Weights & Biases logger.
        
        Args:
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            wandb_config: Weights & Biases configuration
            wandb_tags: Weights & Biases tags
        """
        try:
            import wandb
            self.wandb = wandb
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config,
                tags=wandb_tags,
                name=self.experiment_name,
                dir=self.output_dir,
            )
        except ImportError:
            logger.warning("WandB not installed. Disabling WandB logging.")
            self.log_to_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}. Disabling WandB logging.")
            self.log_to_wandb = False
    
    def _setup_tensorboard_logger(self, tensorboard_log_dir: Optional[str] = None) -> None:
        """
        Set up the TensorBoard logger.
        
        Args:
            tensorboard_log_dir: TensorBoard log directory
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tensorboard_log_dir = tensorboard_log_dir or os.path.join(self.output_dir, "tensorboard")
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(tensorboard_log_dir, self.experiment_name)
            )
        except ImportError:
            logger.warning("TensorBoard not installed. Disabling TensorBoard logging.")
            self.log_to_tensorboard = False
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}. Disabling TensorBoard logging.")
            self.log_to_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step
        """
        if self.log_to_console:
            metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            logger.info(f"Step {step}: {metrics_str}")
        
        if self.log_to_csv and self.csv_writer is not None:
            if not self.csv_header_written:
                self.csv_writer.writerow(["step"] + list(metrics.keys()))
                self.csv_header_written = True
            
            self.csv_writer.writerow([step] + list(metrics.values()))
            self.csv_file.flush()
        
        if self.log_to_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)
        
        if self.log_to_tensorboard and self.tensorboard_writer is not None:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
    
    def log_model(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Log model to WandB.
        
        Args:
            model: Model to log
            optimizer: Optimizer to log
        """
        if self.log_to_wandb and self.wandb is not None:
            try:
                self.wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                logger.warning(f"Failed to log model to WandB: {e}")
    
    def close(self) -> None:
        """
        Close all loggers.
        """
        
        if self.csv_file is not None:
            self.csv_file.close()
        
        if self.log_to_wandb and self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to close WandB: {e}")
        
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()


def create_metrics_logger(
    config: Any,
    output_dir: str,
    experiment_name: Optional[str] = None,
) -> MetricsLogger:
    """
    Create a metrics logger from configuration.
    
    Args:
        config: Configuration object or dictionary
        output_dir: Directory to save outputs
        experiment_name: Name of the experiment
        
    Returns:
        A configured metrics logger
    """
    logging_config = getattr(config, "logging", {})
    
    return MetricsLogger(
        output_dir=output_dir,
        experiment_name=experiment_name or getattr(config, "experiment_name", None),
        log_to_console=logging_config.get("console", True),
        log_to_csv=logging_config.get("csv", True),
        log_to_wandb=logging_config.get("wandb", {}).get("enabled", False),
        log_to_tensorboard=logging_config.get("tensorboard", {}).get("enabled", False),
        wandb_project=logging_config.get("wandb", {}).get("project", None),
        wandb_entity=logging_config.get("wandb", {}).get("entity", None),
        wandb_config=config,
        wandb_tags=logging_config.get("wandb", {}).get("tags", None),
        tensorboard_log_dir=logging_config.get("tensorboard", {}).get("log_dir", None),
    )
