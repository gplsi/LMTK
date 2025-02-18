import pprint
from box import Box
from src.utils._dataset_helpers import DatasetHelper, VerboseLevel


# src/tasks/tokenization.py
def execute(config: Box):
    print("Tokenizing with the following config:")
    pprint.pprint(config.items)

    if config.dataset is None:
        raise ValueError("Dataset configuration must be provided")

    # Handle dataset input
    if config.dataset.source == "local":
        if config.dataset.nameOrPath is None:
            raise ValueError("Dataset name or path must be provided for local datasets")

        dataset_handler = DatasetHelper(
            verbose_level=(
                VerboseLevel(config.verbose_level)
                if config.verbose_level is not None
                else VerboseLevel.INFO
            )
        )
        
        if config.dataset.format == "dataset":
            # TODO: needs testing
            dataset = dataset_handler.load_from_disk(config.dataset.nameOrPath)
        elif config.dataset.format == "files":
            dataset = dataset_handler.process_files(config.dataset.nameOrPath)
        else:
            raise ValueError("Invalid dataset format")

    elif config.dataset.source == "huggingface":
        # TODO: we need to implement and test this functionality
        print("It is from Hugging Face!")
    else:
        raise ValueError("Invalid dataset source")

    # Handle tokenizer input
    if config.tokenizer is None:
        raise ValueError("Tokenizer configuration must be provided")
    
    

    # workflow
    # if dataset from files and not transformer dataset
    # - load from files
    # - create dataset
    # - save to disk (if specified)

    # then for all datasets
    # - tokenize
    # - save to disk (if specified)
