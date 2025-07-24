import yaml
from pathlib import Path
from box import Box
from jsonschema import Draft7Validator, RefResolver

"""
Module for configuration validation using JSON Schema.

This module defines the ConfigValidator class that loads JSON schemas (in YAML format)
from a designated directory and validates configuration YAML files against these schemas.
"""



class ConfigValidator:
    """
    A utility class for validating configuration files using JSON Schema.

    The class loads JSON schema definitions from a given directory and stores them in a
    dictionary, keyed by their corresponding file URI. This enables proper resolution of
    schema references when validating configuration data.

    """
    
    def __init__(self, schema_dir="config/schemas") -> None:
        """
        Initialize the ConfigValidator instance and load all available schemas.

        Parameters:
            schema_dir (str): The directory path that contains schema files in YAML format.
                Defaults to "config/schemas".

        The schema directory is resolved to an absolute path to ensure consistency, and
        all schemas found in the directory (including subdirectories) are loaded into the
        internal schema_store for later reference resolution during validation.

        """
        self.schema_dir = Path(schema_dir).resolve()  # Ensure absolute path
        self.schema_store = {}
        self._load_all_schemas()

    def _load_all_schemas(self) -> None:
        """
        Recursively load all JSON schema files from the schema directory.

        This private method searches for files ending with the ".schema.yaml" extension, loads
        each schema using yaml.safe_load, and stores them in the schema_store dictionary. The key
        for each schema is its file URI, ensuring that references using $ref resolve correctly.

        """
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
                
                # Store with full resolved path
                schema_id = f"file://{schema_path.resolve()}"
                self.schema_store[schema_id] = schema
                
                # Also store with relative path from schema_dir for easier resolution
                relative_path = schema_path.relative_to(self.schema_dir)
                relative_id = f"file://{self.schema_dir.resolve()}/{relative_path}"
                self.schema_store[relative_id] = schema
                
                # Store with just the relative path for direct $ref resolution
                self.schema_store[str(relative_path)] = schema

    def _find_schema_file(self, config_data: dict, task_name: str) -> Path:
        """
        Auto-discover the correct schema file based on config structure and task.
        
        Args:
            config_data (dict): The loaded configuration data
            task_name (str): The main task name (e.g., 'tokenization', 'training')
            
        Returns:
            Path: The path to the appropriate schema file
            
        Raises:
            FileNotFoundError: If no matching schema file is found
        """
        # First, try the exact task name in root directory
        root_schema = self.schema_dir / f"{task_name}.schema.yaml"
        if root_schema.exists():
            return root_schema
            
        # Determine if this is a training or tokenization task based on config structure
        is_training_task = self._is_training_config(config_data)
        is_tokenization_task = self._is_tokenization_config(config_data)
        
        # For training tasks (clm_training, mlm_training, instruction)
        if is_training_task:
            # Try direct training schema match
            training_schema = self.schema_dir / "training" / f"{task_name}.schema.yaml"
            if training_schema.exists():
                return training_schema
                
            # Fallback to general training schema in subfolder
            general_schema = self.schema_dir / "training" / "training.schema.yaml"
            if general_schema.exists():
                return general_schema
            
        # For tokenization tasks, look for specific subtask schemas
        elif task_name == "tokenization" or is_tokenization_task:
            # Check if tokenizer.task exists in config
            tokenizer_task = None
            if "tokenizer" in config_data and "task" in config_data["tokenizer"]:
                tokenizer_task = config_data["tokenizer"]["task"]
            
            # Try tokenization-specific schemas
            if tokenizer_task:
                specific_schema = self.schema_dir / "tokenization" / f"tokenization.{tokenizer_task}.schema.yaml"
                if specific_schema.exists():
                    return specific_schema
                    
        # Search subdirectories with priority: training first, then tokenization
        search_order = ["training", "tokenization"] if is_training_task else ["tokenization", "training"]
        
        for subdir in search_order:
            subdir_path = self.schema_dir / subdir
            if subdir_path.exists():
                for schema_path in subdir_path.rglob("*.schema.yaml"):
                    # Check if the schema filename contains the task name
                    if task_name in schema_path.stem:
                        return schema_path
        
        # Final fallback: search all subdirectories
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            # Check if the schema filename contains the task name
            if task_name in schema_path.stem:
                return schema_path
                
        # If no schema found, raise an error with helpful information
        available_schemas = [str(p.relative_to(self.schema_dir)) for p in self.schema_dir.rglob("*.schema.yaml")]
        raise FileNotFoundError(
            f"No schema file found for task '{task_name}'. "
            f"Available schemas: {', '.join(available_schemas)}"
        )
    
    def _is_training_config(self, config_data: dict) -> bool:
        """
        Determine if a configuration is for training based on its structure.
        
        Args:
            config_data (dict): The loaded configuration data
            
        Returns:
            bool: True if this appears to be a training configuration
        """
        # Training configs typically have these keys
        training_indicators = [
            'model_name',           # Model to train
            'number_epochs',        # Training epochs
            'batch_size',          # Training batch size
            'lr',                  # Learning rate
            'parallelization_strategy',  # Distributed training
            'logging_config',      # Training logging
            'gradient_accumulation',  # Training optimization
            'validate_after_epoch',   # Training validation
        ]
        
        # Count how many training indicators are present
        training_score = sum(1 for key in training_indicators if key in config_data)
        
        # Also check for dataset structure typical of training
        has_training_dataset = (
            'dataset' in config_data and 
            isinstance(config_data['dataset'], dict) and
            ('source' in config_data['dataset'] or 'nameOrPath' in config_data['dataset'])
        )
        
        # Consider it a training config if it has multiple training indicators
        return training_score >= 3 or has_training_dataset
    
    def _is_tokenization_config(self, config_data: dict) -> bool:
        """
        Determine if a configuration is for tokenization based on its structure.
        
        Args:
            config_data (dict): The loaded configuration data
            
        Returns:
            bool: True if this appears to be a tokenization configuration
        """
        # Tokenization configs typically have these keys
        tokenization_indicators = [
            'tokenizer',           # Tokenizer configuration
            'output',             # Output directory for tokenized data
            'max_length',         # Token sequence length
            'stride',             # Tokenization stride
            'overlap',            # Token overlap
        ]
        
        # Count how many tokenization indicators are present
        tokenization_score = sum(1 for key in tokenization_indicators if key in config_data)
        
        # Check for tokenizer-specific nested structure
        has_tokenizer_config = (
            'tokenizer' in config_data and 
            isinstance(config_data['tokenizer'], dict) and
            'task' in config_data['tokenizer']
        )
        
        # Consider it a tokenization config if it has tokenizer indicators
        return tokenization_score >= 2 or has_tokenizer_config

    def validate(self, config_path: Path, schema_name: str) -> Box:
        """
        Validate a configuration file against a JSON schema with auto-discovery.

        :param config_path: The file path to the configuration YAML file to be validated.
        :type config_path: Path
        :param schema_name: The base name of the task for schema discovery.
        :type schema_name: str
        :return: A Box object containing the configuration data, enabling dot notation for attribute access.
        :rtype: Box
        :raises ValueError: If the configuration fails validation.
        :raises FileNotFoundError: If no appropriate schema file is found.

        The method executes the following steps:

        - Loads configuration data from the specified YAML file.
        - Auto-discovers the appropriate schema file based on config structure and task.
        - Constructs a RefResolver with the preloaded schemas to handle JSON Schema references.
        - Validates the configuration data using the Draft7Validator.
        - If validation errors are found, aggregates them into a detailed error message.
        """
        # Load config data
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Auto-discover the appropriate schema file
        task_schema_path = self._find_schema_file(config_data, schema_name)
        
        # Load the discovered schema
        with open(task_schema_path, 'r') as f:
            task_schema = yaml.safe_load(f)

        # Create resolver with preloaded schema store
        # Use the directory containing the discovered schema as base URI for relative references
        # This allows schemas to use local relative paths (e.g., components/model.schema.yaml)
        schema_base_dir = task_schema_path.parent
        
        # Create a schema store that contains the exact keys RefResolver will look for
        # when combining base_uri with relative references
        resolver_store = dict(self.schema_store)  # Start with global store
        
        # Add schemas with keys that match base_uri + relative_ref combinations
        base_uri_str = f"file://{schema_base_dir.resolve()}/"
        
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            
            # For schemas under schema_base_dir, add them with the exact key RefResolver will use
            try:
                relative_to_base = schema_path.relative_to(schema_base_dir)
                resolver_key = base_uri_str + str(relative_to_base)
                resolver_store[resolver_key] = schema
                
                # Also add with just the relative path for direct resolution
                resolver_store[str(relative_to_base)] = schema
            except ValueError:
                # Schema not under schema_base_dir, handle parent references
                if schema_path.name == "base.schema.yaml" and schema_path.parent == self.schema_dir:
                    # Handle ../base.schema.yaml reference
                    parent_ref_key = base_uri_str + "../base.schema.yaml"
                    resolver_store[parent_ref_key] = schema
                    resolver_store["../base.schema.yaml"] = schema
        
        resolver = RefResolver(
            base_uri=base_uri_str,
            referrer=task_schema,
            store=resolver_store
        )
        
        # Debug: Test resolution of all component references
        component_refs = [
            "../base.schema.yaml",
            "components/model.schema.yaml",
            "components/data.schema.yaml", 
            "components/training_args.schema.yaml",
            "components/optimizer.schema.yaml",
            "components/scheduler.schema.yaml"
        ]

        # Validate with error formatting
        validator = Draft7Validator(task_schema, resolver=resolver)
        errors = list(validator.iter_errors(config_data))
        
        if errors:
            error_messages = []
            for error in errors:
                path = ".".join(map(str, error.absolute_path))
                error_messages.append(f"[{path}] {error.message}")
            raise ValueError(
                f"Configuration validation failed using schema '{task_schema_path.relative_to(self.schema_dir)}':\n" + 
                "\n".join(error_messages)
            )

        return Box(config_data, box_dots=True)
