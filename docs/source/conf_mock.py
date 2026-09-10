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

# Mock modules that cause documentation build failures
MOCK_MODULES = [
    'wandb', 'wandb.sdk', 'wandb.sdk.data_types', 
    'numpy.float_', 'torch.cuda', 'torch.distributed',
    'torch.nn.parallel', 'torch.nn.parallel.distributed',
    'deepspeed'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
