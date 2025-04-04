"""Version utilities for the Continual Pretraining Framework.

This module provides utilities for retrieving and displaying
version information from within the codebase.
"""

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

__all__ = ["get_version", "get_version_info", "display_version_info"]


def get_version() -> str:
    """Get the current version of the package.

    Returns:
        str: The version string.
    """
    try:
        # First try to get version from installed package metadata
        return importlib.metadata.version("continual-pretrain")
    except importlib.metadata.PackageNotFoundError:
        # Fall back to reading from pyproject.toml
        try:
            import tomli
            # Look for pyproject.toml in parent directories
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                pyproject_path = current_dir / "pyproject.toml"
                if pyproject_path.exists():
                    with open(pyproject_path, "rb") as f:
                        pyproject_data = tomli.load(f)
                        if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
                            return pyproject_data["tool"]["poetry"]["version"]
                current_dir = current_dir.parent
            
            # If we get here, we couldn't find pyproject.toml
            return "unknown"
        except (ImportError, KeyError):
            return "unknown"


def get_version_info() -> Dict[str, str]:
    """Get detailed version information.

    Returns:
        Dict[str, str]: Dictionary with version details.
    """
    import platform
    import torch
    import transformers

    info = {
        "version": get_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    
    # Add CUDA info if available
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    
    return info


def display_version_info(file=None) -> None:
    """Display version information in a formatted way.

    Args:
        file: File-like object to write to (defaults to sys.stdout).
    """
    if file is None:
        file = sys.stdout
        
    info = get_version_info()
    
    max_key_length = max(len(key) for key in info.keys())
    
    print("Continual Pretraining Framework", file=file)
    print("=" * 35, file=file)
    
    for key, value in info.items():
        print(f"{key.ljust(max_key_length)} : {value}", file=file)