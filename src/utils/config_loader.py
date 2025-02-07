from pathlib import Path
import yamale
from box import Box

class ConfigValidator:
    def __init__(self, schema_dir='config'):
        self.schema_dir = Path(schema_dir)
        self.validator = yamale.validator()
        
    def _load_schema(self, schema_name):
        schema_path = self.schema_dir / f'{schema_name}.schema.yaml'
        return yamale.make_schema(schema_path, parser='ruamel')

    def validate(self, config_path: Path, schema_name: str) -> Box:
        """Validate and return configuration as dot-accessible Box"""
        schema = self._load_schema(schema_name)
        data = yamale.make_data(config_path, parser='ruamel')
        
        try:
            self.validator.validate(schema, data)
            return Box(yamale.util.merge(data[0][0]), box_dots=True)
        except yamale.YamaleError as e:
            raise ValueError(f"Config validation failed: {str(e)}")
