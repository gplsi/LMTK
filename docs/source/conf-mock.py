"""
Mock module configuration for documentation build.
This file is automatically imported by conf.py.
"""
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Monkeypatch numpy.float_ for NumPy >=2.0 (removed in 2.0)
try:
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64  # Patch for Sphinx/autodoc compatibility
except ImportError:
    pass

# Mock modules that cause documentation build failures
MOCK_MODULES = [
    'wandb', 'wandb.sdk', 'wandb.sdk.data_types', 
    'numpy.float_', 'torch.cuda', 'torch.distributed',
    'torch.nn.parallel', 'torch.nn.parallel.distributed',
    'deepspeed'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
