"""Version information for the ML Training Framework."""

import platform
import sys
from datetime import datetime
from typing import Dict, Any

__version__ = "0.1.0"

def get_version() -> str:
    """Return the current version of the framework.
    
    Returns:
        str: The current version number.
    """
    return __version__

def get_system_info() -> Dict[str, Any]:
    """Get information about the current system.
    
    Returns:
        Dict[str, Any]: Dictionary containing system information.
    """
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def display_version_info() -> None:
    """Display detailed version information about the framework."""
    system_info = get_system_info()
    
    print(f"ML Training Framework v{get_version()}")
    print(f"Python Version: {system_info['python_version'].split()[0]}")
    print(f"Platform: {system_info['platform']}")
    print(f"Time: {system_info['timestamp']}")
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"CUDA Devices: {torch.cuda.device_count()}")
            print(f"Current CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")