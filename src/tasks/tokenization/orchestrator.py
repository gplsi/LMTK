# src/tasks/tokenization.py
from box import Box
from datasets import Dataset
from src.utils.logging import get_logger, set_logger_level
from src.tasks.tokenization.tokenizer import CausalLMTokenizer
from src.utils.logging import VerboseLevel
from src.utils.dataset import DatasetStorage
from src.utils.orchestrator import BaseOrchestrator
from src.tasks.tokenization.tokenizer.config import TokenizerConfig
from utils import inherit_init_params

@inherit_init_params
class TokenizationOrchestrator(BaseOrchestrator):
    """Orchestrates the tokenization workflow."""

    def validate_config(self) -> None:
        """Validate tokenization configuration."""
        if not self.config.tokenizer:
            raise ValueError("Tokenizer configuration must be provided")
        if not self.config.tokenizer.name:
            raise ValueError("Tokenizer name must be provided")
        if not self.config.dataset:
            raise ValueError("Dataset configuration must be provided")
        if not self.config.output or not self.config.output.path:
            raise ValueError("Output path must be provided")

    def load_dataset(self) -> Dataset:
        """Load dataset based on configuration."""
        dataset_handler = DatasetStorage(
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            )
        )

        if self.config.dataset.source == "local":
            if self.config.dataset.format == "dataset":
                return dataset_handler.load_from_disk(self.config.dataset.nameOrPath)
            elif self.config.dataset.format == "files":
                return dataset_handler.process_files(
                    self.config.dataset.nameOrPath,
                    extension=self.config.dataset.file_config.format,
                )
            raise ValueError(f"Invalid dataset format: {self.config.dataset.format}")
        elif self.config.dataset.source == "huggingface":
            raise NotImplementedError("HuggingFace dataset loading not implemented yet")
        raise ValueError(f"Invalid dataset source: {self.config.dataset.source}")

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Execute tokenization task."""
        tokenizer_config = TokenizerConfig(
            context_length=self.config.tokenizer.context_length,
            overlap=self.config.tokenizer.get("overlap"),
            tokenizer_name=self.config.tokenizer.name,
            verbose_level=VerboseLevel(
                self.config.get("verbose_level", VerboseLevel.INFO)
            ),
        )

        tokenizer = CausalLMTokenizer(tokenizer_config)
        return tokenizer.tokenize(dataset)

    def execute(self) -> None:
        """Execute the complete tokenization workflow."""
        self.logger.info("Starting tokenization workflow")
        try:
            # 1. Validate configuration
            self.validate_config()

            # 2. Load dataset
            dataset = self.load_dataset()

            # 3. Tokenize dataset
            tokenized_dataset = self.tokenize_dataset(dataset)

            # 4. Save results
            self.storage.save_to_disk(tokenized_dataset, self.config.output.path)

            self.logger.info("Tokenization workflow completed successfully")

        except Exception as e:
            self.logger.error(f"Tokenization workflow failed: {str(e)}")
            raise
