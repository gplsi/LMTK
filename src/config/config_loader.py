import yaml
from pathlib import Path
from box import Box
from jsonschema import Draft7Validator, RefResolver, validators

"""
Module for configuration validation using JSON Schema.

This module defines the ConfigValidator class that loads JSON schemas (in YAML format)
from a designated directory and validates configuration YAML files against these schemas.
"""


class FilenameRefResolver(RefResolver):
    """
    Custom RefResolver that handles references by schema filename.
    
    This resolver looks up schemas by their filename (e.g., "base.schema.yaml"),
    enabling simple $ref resolution regardless of directory structure.
    """
    
    def resolve_remote(self, uri):
        """
        Override resolve_remote to handle filename-based references.
        
        Looks up the schema by filename in the store.
        """
        # Try direct lookup by filename
        if uri in self.store:
            return self.store[uri]
        
        # Also try extracting just the filename from the URI if it contains a path
        from pathlib import Path
        filename = Path(uri).name
        if filename in self.store:
            return self.store[filename]
        
        # Fall back to default behavior
        return super().resolve_remote(uri)


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
        schema_dir_path = Path(schema_dir)
        if not schema_dir_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            schema_dir_path = (project_root / schema_dir_path).resolve()
        else:
            schema_dir_path = schema_dir_path.resolve()

        self.schema_dir = schema_dir_path  # Ensure absolute path to schemas
        self.project_root = Path(__file__).resolve().parents[2]  # Store project root
        self.schema_store = {}
        self._load_all_schemas()

    def _load_schema_file(self, schema_path: Path) -> dict:
        """Load a schema file and normalize its $id to an absolute file URI."""
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f) or {}

        if not isinstance(schema, dict):
            raise ValueError(f"Schema at {schema_path} must define a mapping")

        schema_id_value = schema.get("$id")
        if not schema_id_value or not str(schema_id_value).startswith("file://"):
            schema["$id"] = schema_path.resolve().as_uri()

        return schema

    def _load_all_schemas(self) -> None:
        """
        Recursively load all JSON schema files from the schema directory.

        This private method searches for files ending with the ".schema.yaml" extension, loads
        each schema, and stores them in the schema_store dictionary using the filename as key.
        
        This enables simple $ref resolution by filename (e.g., "base.schema.yaml") regardless
        of the actual directory structure.
        """
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            schema = self._load_schema_file(schema_path)
            
            # Store by filename as the primary key for simple $ref resolution
            filename = schema_path.name
            self.schema_store[filename] = schema

    def _find_schema_file(self, config_data: dict, task_name: str) -> Path:
        """
        Find the schema file matching the task name.
        
        Task names should directly correspond to schema filenames:
        - "clm_training" → "clm_training.schema.yaml" (searches all subdirs)
        - "publish" → "publish.schema.yaml"
        - "tokenization" → "tokenization.schema.yaml"
        
        Args:
            config_data (dict): The loaded configuration data
            task_name (str): The task name matching the schema filename
            
        Returns:
            Path: The path to the schema file
            
        Raises:
            FileNotFoundError: If no matching schema file is found
        """
        # Search for schema file matching task name recursively
        schema_filename = f"{task_name}.schema.yaml"
        matching_schemas = list(self.schema_dir.rglob(schema_filename))
        
        if matching_schemas:
            # Return the first match (there should only be one per task name)
            return matching_schemas[0]
        
        # If no exact match found, provide helpful error
        available_schemas = [str(p.relative_to(self.schema_dir)) for p in self.schema_dir.rglob("*.schema.yaml")]
        raise FileNotFoundError(
            f"No schema file found for task '{task_name}' (looking for '{schema_filename}'). "
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
        task_schema = self._load_schema_file(task_schema_path)

        # Create custom resolver that handles filename-based references
        # All $refs should use just the schema filename (e.g., "base.schema.yaml")
        resolver = FilenameRefResolver(
            base_uri="",
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
