from box import Box
from src.utils.logging import get_logger
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from abc import ABC
from datasets import Dataset

class BaseOrchestrator(ABC):
    """Base Orchestrator Class"""

    def __init__(self, config: Box):
        self.config = config
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )
        self.logger = get_logger(__name__, self.verbose_level)
        self.storage = DatasetStorage(self.verbose_level)

    def _validate__dataset_config(self) -> None:
        """Validate tokenization configuration."""
        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")

    def load_dataset(self) -> Dataset:
        """Load dataset based on configuration."""
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            )
        )
        
        self._validate__dataset_config()
        

        if self.config.dataset.source == "local":
            if self.config.dataset.format == "dataset":
                self.logger.info(f"Loading dataset from path '{self.config.dataset.nameOrPath}'")
                dataset = dataset_handler.load_from_disk(self.config.dataset.nameOrPath)
                
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            
            elif self.config.dataset.format == "files":
                self.logger.info(f"Loading dataset from files at dir '{self.config.dataset.nameOrPath}'")
                dataset= dataset_handler.process_files(
                    self.config.dataset.nameOrPath,
                    extension=self.config.dataset.file_config.format,
                )
                if self.config.test_size:
                    self.logger.info(f"Splitting dataset with test size: {self.config.test_size}")
                dataset = dataset_handler.split(dataset, split_ratio=self.config.test_size)
                
                return dataset
            raise ValueError(f"Invalid dataset format: {self.config.dataset.format}")
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")
