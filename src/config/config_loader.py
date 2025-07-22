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
                schema_id = f"file://{schema_path.resolve()}"  # Match actual file URI
                self.schema_store[schema_id] = schema

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
            
        # For tokenization tasks, look for specific subtask schemas
        if task_name == "tokenization":
            # Check if tokenizer.task exists in config
            tokenizer_task = None
            if "tokenizer" in config_data and "task" in config_data["tokenizer"]:
                tokenizer_task = config_data["tokenizer"]["task"]
            
            # Try tokenization-specific schemas
            if tokenizer_task:
                specific_schema = self.schema_dir / "tokenization" / f"tokenization.{tokenizer_task}.schema.yaml"
                if specific_schema.exists():
                    return specific_schema
                
        # For training tasks, look for specific training type schemas
        elif task_name == "training":
            # Check if task_type exists in config
            task_type = None
            if "task_type" in config_data:
                task_type = config_data["task_type"]
            elif "model" in config_data and "task_type" in config_data["model"]:
                task_type = config_data["model"]["task_type"]
                
            # Try training-specific schemas
            if task_type:
                specific_schema = self.schema_dir / "training" / f"{task_type}.schema.yaml"
                if specific_schema.exists():
                    return specific_schema
                    
            # Fallback to general training schema in subfolder
            general_schema = self.schema_dir / "training" / "training.schema.yaml"
            if general_schema.exists():
                return general_schema
                
        # Search all subdirectories for a matching schema
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
        # while keeping the full schema store for cross-directory references
        schema_base_dir = task_schema_path.parent
        resolver = RefResolver(
            base_uri=f"file://{schema_base_dir}/",
            referrer=task_schema,
            store=self.schema_store
        )

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
