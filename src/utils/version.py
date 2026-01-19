"""Version information for the ML Training Framework."""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO

__version__ = "0.1.0"


def get_version() -> str:
    """Return the current version of the framework."""
    try:
        from importlib.metadata import version
        return version("continual-pretrain")
    except Exception:
        pass

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomli
        except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.11+
            import tomllib as tomli
        with pyproject_path.open("rb") as handle:
            data = tomli.load(handle)
        return data.get("tool", {}).get("poetry", {}).get("version", __version__)

    return __version__


def get_system_info() -> Dict[str, Any]:
    """Get information about the current system."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_version_info() -> Dict[str, Any]:
    """Collect version metadata for the framework and key dependencies."""
    info: Dict[str, Any] = {
        "version": get_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        info["torch"] = "not installed"

    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "not installed"

    return info


def display_version_info(file: TextIO | None = None) -> None:
    """Display detailed version information about the framework."""
    if file is None:
        file = sys.stdout

    info = get_version_info()
    print("Continual Pretraining Framework", file=file)
    for key in ("version", "python", "platform", "torch", "cuda", "gpu", "transformers"):
        if key in info:
            print(f"{key:<10}: {info[key]}", file=file)
