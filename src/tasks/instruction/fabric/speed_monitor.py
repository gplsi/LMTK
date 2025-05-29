"""
Speed monitoring utility for instruction fine-tuning with Lightning Fabric.

This module provides the SpeedMonitorFabric class that tracks and reports
training speed metrics such as samples per second, tokens per second,
and estimated time remaining.
"""

import time
from typing import Optional, Dict, Any
import lightning as L


class SpeedMonitorFabric:
    """
    Monitor and log training speed metrics for instruction fine-tuning.
    
    This class tracks training progress and calculates performance metrics
    such as samples per second, tokens per second, and estimated time remaining.
    It integrates with Lightning Fabric for distributed training scenarios.
    """
    
    def __init__(self, fabric: L.Fabric, log_interval: int = 1) -> None:
        """
        Initialize the speed monitor.
        
        Args:
            fabric: Lightning Fabric instance for distributed training
            log_interval: How often to log metrics (in steps)
        """
        self.fabric = fabric
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_logged_time = self.start_time
        self.last_logged_step = 0
        self.total_samples = 0
        self.total_tokens = 0
    
    def on_train_batch_end(
        self,
        samples: int,
        elapsed_time: float,
        world_size: int,
        step: int,
        lengths: Optional[int] = None,
        train_loss: Optional[float] = None,
    ) -> None:
        """
        Update and log speed metrics at the end of a training batch.
        
        Args:
            samples: Number of samples processed so far
            elapsed_time: Total elapsed training time
            world_size: Number of devices/processes
            step: Current training step
            lengths: Number of tokens in the current batch
            train_loss: Training loss for the current batch
        """
        # Update counters
        self.total_samples = samples
        if lengths is not None:
            self.total_tokens += lengths * world_size
        
        # Check if it's time to log
        if step % self.log_interval != 0:
            return
        
        current_time = time.time()
        
        # Calculate time elapsed since last log
        time_since_last = current_time - self.last_logged_time
        steps_since_last = step - self.last_logged_step
        
        # Only log if we have meaningful time difference
        if time_since_last < 0.001 or steps_since_last == 0:
            return
        
        # Calculate metrics
        samples_per_sec = samples / elapsed_time
        steps_per_sec = steps_since_last / time_since_last
        
        # Prepare metrics dict
        metrics: Dict[str, Any] = {
            "train/samples_per_sec": samples_per_sec,
            "train/steps_per_sec": steps_per_sec,
            "train/elapsed_time": elapsed_time,
        }
        
        # Add tokens per second if available
        if self.total_tokens > 0:
            tokens_per_sec = self.total_tokens / elapsed_time
            metrics["train/tokens_per_sec"] = tokens_per_sec
        
        # Add loss if available
        if train_loss is not None:
            metrics["train/loss"] = train_loss
        
        # Log metrics
        self.fabric.log_dict(metrics)
        
        # Update last logged time and step
        self.last_logged_time = current_time
        self.last_logged_step = step
