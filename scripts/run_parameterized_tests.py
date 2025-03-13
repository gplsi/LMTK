#!/usr/bin/env python
"""
Script to run parameterized tests with different configurations.
This allows testing with various combinations of parameters to ensure robustness.
"""
import argparse
import itertools
import os
import subprocess
import tempfile
from pathlib import Path
import yaml
import sys


def create_test_config(output_path, **kwargs):
    """Create a test configuration file with the specified parameters."""
    # Load the base test configuration
    base_config_path = Path("config/experiments/test_config.yaml")
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update configuration with provided parameters
    for key, value in kwargs.items():
        if "." in key:
            # Handle nested keys (e.g., "dataset.source")
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    # Write the configuration to the output file
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    
    return output_path


def run_tests_with_config(config_path, test_type="unit", test_pattern=None):
    """Run tests using the specified configuration."""
    # Set environment variable so tests can access the config
    os.environ["TEST_CONFIG_PATH"] = str(config_path)
    
    # Build the pytest command
    cmd = ["pytest"]
    
    if test_type == "unit":
        cmd.append("tests/unit")
    elif test_type == "integration":
        cmd.append("tests/integration")
    elif test_type == "all":
        cmd.append("tests")
    
    if test_pattern:
        cmd.append(f"-k {test_pattern}")
    
    cmd.append("-v")
    
    # Run the tests
    print(f"Running tests with configuration: {config_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:", result.stderr)
    
    return result.returncode == 0


def run_parameterized_tests(param_grid, test_type="unit", test_pattern=None):
    """Run tests with all combinations of parameters from the parameter grid."""
    # Generate all combinations of parameters
    keys = param_grid.keys()
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # Track test results
    results = []
    
    # Create a temporary directory for test configurations
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, combination in enumerate(combinations):
            # Create a dictionary of parameters for this combination
            params = {key: value for key, value in zip(keys, combination)}
            
            # Create the test configuration file
            config_path = os.path.join(temp_dir, f"test_config_{i}.yaml")
            create_test_config(config_path, **params)
            
            # Run the tests
            success = run_tests_with_config(config_path, test_type, test_pattern)
            results.append((params, success))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    failures = 0
    for params, success in results:
        status = "PASSED" if success else "FAILED"
        if not success:
            failures += 1
        
        print(f"{status}: {params}")
    
    print("\n" + "=" * 50)
    print(f"Total combinations: {len(results)}")
    print(f"Passed: {len(results) - failures}")
    print(f"Failed: {failures}")
    
    # Return success only if all tests passed
    return failures == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterized tests with different configurations")
    parser.add_argument("--test-type", choices=["unit", "integration", "all"], default="unit",
                        help="Type of tests to run")
    parser.add_argument("--test-pattern", type=str, default=None, 
                        help="Pattern to select specific tests")
    parser.add_argument("--param-file", type=str, default=None,
                        help="YAML file containing parameter grid")
    args = parser.parse_args()
    
    # Define default parameter grid for comprehensive testing
    default_param_grid = {
        "model_name": ["gpt2"],
        "precision": ["16-mixed", "bf16-mixed"],
        "parallelization_strategy": ["none", "fsdp"],
        "gradient_checkpointing": [True, False],
    }
    
    # Load parameter grid from file if provided
    param_grid = default_param_grid
    if args.param_file:
        with open(args.param_file, "r") as f:
            param_grid = yaml.safe_load(f)
    
    # Run the tests
    success = run_parameterized_tests(param_grid, args.test_type, args.test_pattern)
    sys.exit(0 if success else 1)