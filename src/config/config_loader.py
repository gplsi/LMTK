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
        - schema_dir (str): The directory path that contains schema files in YAML format.
                            Defaults to "config/schemas".
        
        The schema directory is resolved to an absolute path to ensure consistency, and
        all schemas found in the directory (including subdirectories) are loaded into the
        internal schema_store for later reference resolution during validation.
        """
        self.schema_dir = Path(schema_dir).resolve()  # Convert the schema directory to an absolute path.
        self.schema_store = {}  # Dictionary to store loaded schemas with their file URIs as keys.
        self._load_all_schemas()  # Load all schema files from the schema directory.

    def _load_all_schemas(self) -> None:
        """
        Recursively load all JSON schema files from the schema directory.

        This private method searches for files ending with the ".schema.yaml" extension, loads
        each schema using yaml.safe_load, and stores them in the schema_store dictionary. The key
        for each schema is its file URI, ensuring that references using $ref resolve correctly.
        """
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            # Open the schema file in read-only mode and load its content.
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)  # Parse the YAML file into a Python dictionary.
                # Generate a file URI from the absolute path of the schema.
                schema_id = f"file://{schema_path.resolve()}"
                self.schema_store[schema_id] = schema  # Store the schema in the schema_store.

    def validate(self, config_path: Path, schema_name: str) -> Box:
        """
        Validate a configuration file against a specified JSON schema.

        The method executes the following steps:
        - Loads configuration data from the specified YAML file.
        - Loads the task-specific schema identified by schema_name.
        - Constructs a RefResolver with the preloaded schemas to handle JSON Schema references.
        - Validates the configuration data using the Draft7Validator.
        - If validation errors are found, aggregates them into a detailed error message.

        Parameters:
        - config_path (Path): The file path to the configuration YAML file to be validated.
        - schema_name (str): The base name (without extension) of the schema file to use for validation.

        Returns:
        - Box: A Box object containing the configuration data, enabling dot notation for attribute access.

        Raises:
        - ValueError: If the configuration fails validation. The exception message will contain
                      detailed error messages for all validation issues encountered.
        """
        # Load the configuration data from the provided config_path.
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)  # Parse YAML into a Python dictionary.

        # Build the file path to the task-specific schema based on the provided schema_name.
        task_schema_path = self.schema_dir / f"{schema_name}.schema.yaml"
        # Load the task-specific JSON schema.
        with open(task_schema_path, 'r') as f:
            task_schema = yaml.safe_load(f)

        # Create a RefResolver instance with the base URI set to the schema directory.
        # This resolver uses the preloaded schema_store to resolve any $ref pointers in the schemas.
        resolver = RefResolver(
            base_uri=f"file://{self.schema_dir}/",  # The base URI for resolving relative references.
            referrer=task_schema,  # The main schema that may contain $ref references.
            store=self.schema_store  # Preloaded schemas to resolve references.
        )

        # Initialize the schema validator using the task schema and the resolver.
        validator = Draft7Validator(task_schema, resolver=resolver)
        # Gather all validation errors.
        errors = list(validator.iter_errors(config_data))
        
        if errors:
            error_messages = []
            # Process each validation error to create a diagnostic message.
            for error in errors:
                # Generate a dot-separated string representing the path in the configuration where the error occurred.
                path = ".".join(map(str, error.absolute_path))
                error_messages.append(f"[{path}] {error.message}")
            # Raise an exception with all the collected error messages if any errors are found.
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))
        
        # Return the validated configuration data wrapped in a Box for dot notation access.
        return Box(config_data, box_dots=True)
