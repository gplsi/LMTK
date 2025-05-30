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

    def _get_schema_file(self, schema_name: str, config_data: dict = None) -> str:
        """
        Determine the appropriate schema file based on the schema name and configuration data.
        
        For training tasks, this builds the schema name based on task, framework, and strategy.
        For non-training tasks, it uses the original schema name.
        
        Parameters:
            schema_name (str): The base name of the schema file.
            config_data (dict, optional): The configuration data, used to extract framework and strategy.
                Defaults to None.
                
        Returns:
            str: The name of the schema file to use for validation.
        """
        # No strategy mapping needed - use strategy names directly
        
        # For training tasks, build the schema name based on task, framework, and strategy
        if schema_name == "training" and config_data:
            # Start with the base training schema
            schema_file = schema_name
            
            # Add task if present
            task = config_data.get("task")
            if task:
                schema_file = f"{schema_file}.{task}"
                
                # Add framework if present (only for training tasks)
                framework = config_data.get("framework")
                if framework:
                    schema_file = f"{schema_file}.{framework}"
                    
                    # Add strategy if present (only for training tasks with framework)
                    strategy = config_data.get("strategy")
                    if strategy:
                        # For DataParallel strategy, map dp to dataparallel
                        if strategy == "dp":
                            schema_file = f"{schema_file}.dataparallel"
                        else:
                            schema_file = f"{schema_file}.{strategy}"
            
            return f"{schema_file}.schema.yaml"
        
        # For non-training tasks, use the original schema name
        return f"{schema_name}.schema.yaml"

    def _resolve_schema_refs(self, schema, resolver):
        """
        Recursively resolve all $ref references in a schema.
        
        :param schema: The schema to resolve references in
        :param resolver: The RefResolver to use for resolving references
        :return: A new schema with all references resolved
        """
        if not isinstance(schema, dict):
            return schema
        
        # Create a new schema to avoid modifying the original
        resolved_schema = {}
        
        # Handle $ref if present
        if "$ref" in schema:
            ref = schema["$ref"]
            # Resolve the reference
            with resolver.resolving(ref) as resolved:
                # Merge the resolved schema with the current schema
                resolved_schema.update(self._resolve_schema_refs(resolved, resolver))
                # Copy all other properties from the current schema
                for k, v in schema.items():
                    if k != "$ref":
                        resolved_schema[k] = self._resolve_schema_refs(v, resolver)
                return resolved_schema
        
        # Handle allOf if present - merge all schemas in the allOf list
        if "allOf" in schema:
            # Start with an empty schema
            merged_schema = {}
            # Process each schema in allOf
            for subschema in schema["allOf"]:
                # Resolve the subschema
                resolved_subschema = self._resolve_schema_refs(subschema, resolver)
                # Merge with the current merged schema
                self._merge_schemas(merged_schema, resolved_subschema)
            
            # Copy all other properties from the current schema
            for k, v in schema.items():
                if k != "allOf":
                    if k in merged_schema and isinstance(merged_schema[k], dict) and isinstance(v, dict):
                        self._merge_schemas(merged_schema[k], self._resolve_schema_refs(v, resolver))
                    else:
                        merged_schema[k] = self._resolve_schema_refs(v, resolver)
            
            return merged_schema
        
        # Process all other properties
        for k, v in schema.items():
            resolved_schema[k] = self._resolve_schema_refs(v, resolver)
        
        return resolved_schema
    
    def _merge_schemas(self, target, source):
        """
        Merge two schemas, with source taking precedence for conflicts.
        
        :param target: The target schema to merge into
        :param source: The source schema to merge from
        """
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                self._merge_schemas(target[k], v)
            else:
                target[k] = v

    def validate(self, config_path: Path, schema_name: str) -> Box:
        """
        Validate a configuration file against a specified JSON schema.

        :param config_path: The file path to the configuration YAML file to be validated.
        :type config_path: Path
        :param schema_name: The base name (without extension) of the schema file to use for validation.
        :type schema_name: str
        :return: A Box object containing the configuration data, enabling dot notation for attribute access.
        :rtype: Box
        :raises ValueError: If the configuration fails validation. The exception message will contain detailed error messages for all validation issues encountered.

        The method executes the following steps:

        - Loads configuration data from the specified YAML file.
        - Loads the task-specific schema identified by schema_name.
        - Constructs a RefResolver with the preloaded schemas to handle JSON Schema references.
        - Resolves all schema references to create a complete schema.
        - Validates the configuration data using the Draft7Validator.
        - If validation errors are found, aggregates them into a detailed error message.
        """
        # Load config data
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Determine the appropriate schema file
        schema_file = self._get_schema_file(schema_name, config_data)
        task_schema_path = self.schema_dir / schema_file

        # Load task schema
        with open(task_schema_path, 'r') as f:
            task_schema = yaml.safe_load(f)

        # Create resolver with preloaded schema store
        resolver = RefResolver(
            base_uri=f"file://{self.schema_dir}/",
            referrer=task_schema,
            store=self.schema_store
        )
        
        # Resolve all schema references to create a complete schema
        resolved_schema = self._resolve_schema_refs(task_schema, resolver)
        
        # Validate with error formatting
        validator = Draft7Validator(resolved_schema)
        errors = list(validator.iter_errors(config_data))
        
        if errors:
            error_messages = []
            for error in errors:
                path = ".".join(map(str, error.absolute_path))
                error_messages.append(f"[{path}] {error.message}")
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))

        return Box(config_data, box_dots=True)
