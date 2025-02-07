import yaml
import json
import os
from pathlib import Path
from jsonschema import validate, ValidationError, RefResolver

class ConfigValidator:
    def __init__(self, schema_dir="config/schemas"):
        self.schema_dir = Path(schema_dir)
        self.schema_cache = {}
        
    def _load_schema(self, schema_path):
        """Cache schemas to handle references"""
        if schema_path not in self.schema_cache:
            with open(schema_path) as f:
                self.schema_cache[schema_path] = yaml.safe_load(f)
        return self.schema_cache[schema_path]
    
    def validate_config(self, config_path):
        """Validate configuration against its declared schema"""
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if "$schema" not in config:
            raise ValidationError("Missing $schema declaration in config")
        
        schema_path = self.schema_dir / config["$schema"]
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema {schema_path} not found")
            
        schema = self._load_schema(schema_path)
        resolver = RefResolver(
            base_uri=f"file://{self.schema_dir}/",
            referrer=schema,
            store=self.schema_cache
        )
        
        try:
            validate(
                instance=config,
                schema=schema,
                resolver=resolver
            )
            return True, None
        except ValidationError as e:
            return False, self._format_error(e)
            
    def _format_error(self, error):
        return {
            "message": error.message,
            "path": list(error.absolute_path),
            "validator": error.validator,
            "schema_path": list(error.relative_schema_path)
        }
