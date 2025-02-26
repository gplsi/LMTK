import yaml
from pathlib import Path
from box import Box
from jsonschema import Draft7Validator, RefResolver

class ConfigValidator:
    def __init__(self, schema_dir="config/schemas"):
        self.schema_dir = Path(schema_dir).resolve()  # Ensure absolute path
        self.schema_store = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        """Load schemas with proper URI identifiers"""
        for schema_path in self.schema_dir.rglob("*.schema.yaml"):
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
                schema_id = f"file://{schema_path.resolve()}"  # Match actual file URI
                self.schema_store[schema_id] = schema

    def validate(self, config_path: Path, schema_name: str) -> Box:
        """
        Validate config against a composed schema (base + task-specific)
        using JSON Schema's $ref and allOf capabilities
        """
        # Load config data
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Load task schema
        task_schema_path = self.schema_dir / f"{schema_name}.schema.yaml"
        with open(task_schema_path, 'r') as f:
            task_schema = yaml.safe_load(f)

        # Create resolver with preloaded schema store
        resolver = RefResolver(
            base_uri=f"file://{self.schema_dir}/",
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
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))

        return Box(config_data, box_dots=True)
