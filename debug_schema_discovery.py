#!/usr/bin/env python3
"""
Debug script for testing schema auto-discovery logic.
Run this in VS Code debugger to step through the schema discovery process.
"""

import sys
import os
from pathlib import Path
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.config_loader import ConfigValidator

def debug_schema_discovery():
    """Debug the schema auto-discovery process step by step."""
    
    # Configuration
    config_path = Path("config/examples/tokenization_clm_training_testing_example.yaml")
    schema_dir = Path("config/schemas")
    
    print("=" * 60)
    print("SCHEMA AUTO-DISCOVERY DEBUG SESSION")
    print("=" * 60)
    
    # Check if files exist
    print(f"1. Checking if config file exists: {config_path}")
    if not config_path.exists():
        print(f"   ERROR: Config file not found at {config_path.absolute()}")
        return
    print(f"   ✓ Config file found")
    
    print(f"\n2. Checking if schema directory exists: {schema_dir}")
    if not schema_dir.exists():
        print(f"   ERROR: Schema directory not found at {schema_dir.absolute()}")
        return
    print(f"   ✓ Schema directory found")
    
    # Load config data
    print(f"\n3. Loading config data from {config_path}")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print(f"   Config keys: {list(config_data.keys())}")
    print(f"   Task: {config_data.get('task', 'NOT FOUND')}")
    
    if 'tokenizer' in config_data:
        print(f"   Tokenizer keys: {list(config_data['tokenizer'].keys())}")
        print(f"   Tokenizer task: {config_data['tokenizer'].get('task', 'NOT FOUND')}")
    
    # List available schemas
    print(f"\n4. Available schema files:")
    schema_files = list(schema_dir.rglob("*.schema.yaml"))
    for schema_file in sorted(schema_files):
        relative_path = schema_file.relative_to(schema_dir)
        print(f"   - {relative_path}")
    
    # Initialize validator
    print(f"\n5. Initializing ConfigValidator")
    validator = ConfigValidator(schema_dir=str(schema_dir))
    print(f"   Schema store contains {len(validator.schema_store)} schemas")
    
    # Test schema discovery
    print(f"\n6. Testing schema auto-discovery")
    task_name = config_data.get('task', 'unknown')
    
    try:
        print(f"   Looking for schema for task: '{task_name}'")
        discovered_schema = validator._find_schema_file(config_data, task_name)
        print(f"   ✓ DISCOVERED SCHEMA: {discovered_schema}")
        print(f"   ✓ Relative path: {discovered_schema.relative_to(schema_dir)}")
        
        # Verify the schema file exists and is readable
        print(f"\n7. Verifying discovered schema")
        if discovered_schema.exists():
            print(f"   ✓ Schema file exists")
            with open(discovered_schema, 'r') as f:
                schema_content = yaml.safe_load(f)
            print(f"   ✓ Schema is valid YAML")
            print(f"   Schema title: {schema_content.get('title', 'No title')}")
            print(f"   Schema description: {schema_content.get('description', 'No description')[:100]}...")
        else:
            print(f"   ✗ ERROR: Schema file does not exist!")
            
    except Exception as e:
        print(f"   ✗ ERROR in schema discovery: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    # Test full validation
    print(f"\n8. Testing full validation process")
    try:
        validated_config = validator.validate(config_path, task_name)
        print(f"   ✓ Validation successful!")
        print(f"   Config type: {type(validated_config)}")
        print(f"   Config keys: {list(validated_config.keys())}")
    except Exception as e:
        print(f"   ✗ ERROR in validation: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEBUG SESSION COMPLETE")
    print("=" * 60)

def test_specific_scenarios():
    """Test specific schema discovery scenarios."""
    
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC SCENARIOS")
    print("=" * 60)
    
    schema_dir = Path("config/schemas")
    validator = ConfigValidator(schema_dir=str(schema_dir))
    
    # Test scenarios
    scenarios = [
        {
            "name": "Tokenization with CLM task",
            "config": {
                "task": "tokenization",
                "tokenizer": {"task": "clm_training"}
            },
            "expected": "tokenization/tokenization.clm_training.schema.yaml"
        },
        {
            "name": "Tokenization with instruction task",
            "config": {
                "task": "tokenization",
                "tokenizer": {"task": "instruction"}
            },
            "expected": "tokenization/tokenization.instruction.schema.yaml"
        },
        {
            "name": "Tokenization with MLM task",
            "config": {
                "task": "tokenization",
                "tokenizer": {"task": "mlm_training"}
            },
            "expected": "tokenization/tokenization.mlm_training.schema.yaml"
        },
        {
            "name": "Training with CLM task",
            "config": {
                "task": "training",
                "task_type": "clm_training"
            },
            "expected": "training/clm_training.schema.yaml"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"   Config: {scenario['config']}")
        print(f"   Expected: {scenario['expected']}")
        
        try:
            discovered = validator._find_schema_file(scenario['config'], scenario['config']['task'])
            relative_path = discovered.relative_to(schema_dir)
            print(f"   Discovered: {relative_path}")
            
            if str(relative_path) == scenario['expected']:
                print(f"   ✓ MATCH!")
            else:
                print(f"   ✗ MISMATCH!")
                
        except Exception as e:
            print(f"   ✗ ERROR: {e}")

if __name__ == "__main__":
    # Set working directory to the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Run debug functions
    debug_schema_discovery()
    test_specific_scenarios()
    
    # Breakpoint for VS Code debugger
    print("\n🔍 Set breakpoint here to inspect variables in debugger")
    breakpoint()  # This will pause execution in VS Code debugger
