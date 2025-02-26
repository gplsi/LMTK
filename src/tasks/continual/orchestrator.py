# src/tasks/tokenization.py
from box import Box
from datasets import Dataset
import torch
from src.utils.logging import get_logger
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from src.tasks.continual.fabric.distributed import FSDP, DeepSpeed, DistributedDataParallel, DataParallel
from utils import inherit_init_params
from utils.orchestrator import BaseOrchestrator

class ContinualOrchestrator(BaseOrchestrator):
    """Orchestrates the continual pretraining workflow."""

    def __init__(self, config: Box):
        super().__init__(config)
        
        # get all devices available in torch so we can set them to torch modules
        if (torch.cuda.is_available()):
            self.devices = torch.cuda.device_count()
            self.logger.info(f"Found {self.devices} CUDA devices available for training")
            if self.devices > 0:
                return
            
        self.logger.warning("No CUDA devices available for training. Training will be done on CPU")
        self.devices = "cpu"

    def validate_config(self) -> None:
        """Validate continual configuration."""
        # TODO: Implement general configuration validation
        pass

    def fsdp(self, dataset):
        """Execute fsdp continual pretraining task."""
        self.logger.info("Starting FSDP continual pretraining task")
        
        # TODO: Validate specific configuration
        
        trainer = FSDP(
            devices=self.devices,
            config=self.config,
            checkpoint=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("FSDP training finished")
        
    def ddp(self, dataset):
        """Execute ddp continual pretraining task."""
        self.logger.info("Starting DDP continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DistributedDataParallel(
            devices=self.devices,
            config=self.config,
            checkpoint=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("DDP training finished")
        
    def dp(self, dataset):
        """Execute ddp continual pretraining task."""
        self.logger.info("Starting DP continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DataParallel(
            devices=self.devices,
            config=self.config,
            checkpoint=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("DP training finished")

    def deep_speed(self, dataset):
        """Execute deep speed continual pretraining task."""
        self.logger.info("Starting Deep Speed continual pretraining task")
        # TODO: Validate specific configuration
        
        trainer = DeepSpeed(
            devices=self.devices,
            config=self.config,
            checkpoint=self.config.get("checkpoint", None)
        )
        
        trainer.setup()
        self.logger.info("Deep Speed training finished")

    def execute(self) -> None:
        """Execute the complete tokenization workflow."""
        self.logger.info("Starting tokenization workflow")
        try:
            # 1. Validate configuration
            self.validate_config()
            
            # 2. Load dataset
            dataset = self.load_dataset()

            # Select specific continual method
            strategy = self.config.get("strategy", "fsdp")
            if strategy == "fsdp":
                self.fsdp(dataset)
            elif strategy == "ddp":
                self.ddp(dataset)
            elif strategy == "deep_speed":
                self.deep_speed(dataset)
            elif strategy == "dp":
                self.dp(dataset)
            else:
                raise ValueError(f"Invalid parallelization strategy: {strategy}")
            
            self.logger.info("Continual Pretraining completed successfully")

        except Exception as e:
            self.logger.error(f"Tokenization workflow failed: {str(e)}")
            raise
