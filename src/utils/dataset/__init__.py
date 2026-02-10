try:
    from src.utils.dataset.storage import DatasetStorage
except ModuleNotFoundError:  # pragma: no cover - lightweight test envs may not include datasets
    DatasetStorage = None  # type: ignore[assignment]
from src.utils.dataset.metadata import (
    TOKENIZATION_METADATA_FILENAME,
    TOKENIZATION_METADATA_SCHEMA_VERSION,
    build_tokenization_metadata,
    extract_eos_token_id,
    read_tokenization_metadata,
    tokenization_metadata_path,
    write_tokenization_metadata,
)
