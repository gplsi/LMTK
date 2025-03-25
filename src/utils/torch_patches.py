"""
Utility module for patching PyTorch CUDA functions to handle NVML compatibility issues.

This module provides comprehensive patching to handle NVML compatibility issues 
that occur during distributed training with DeepSpeed and PyTorch.
"""

import os
import sys
import logging
from functools import wraps

# Set up a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Apply critical environment variables immediately to prevent NVML issues
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Optimize DeepSpeed performance and disable NVML features
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
#os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

# Disable NVML initialization in DeepSpeed
os.environ["NVML_SKIP_INIT"] = "1"

# Import torch only once to avoid circular imports
try:
    import torch
except ImportError:
    logger.warning("PyTorch not found, patches will not be applied")
    torch = None

def patch_torch_cuda():
    """
    Patch PyTorch's CUDA functions to handle NVML-related errors.
    """
    if torch is None:
        logger.warning("Cannot patch torch, module not imported")
        return

    # Patch tensor creation functions
    for func_name in ['empty', 'zeros', 'ones', 'full', 'rand', 'randn']:
        if not hasattr(torch, func_name):
            continue
            
        original_func = getattr(torch, func_name)
        
        def make_safe_func(orig_f, fname):
            @wraps(orig_f)
            def safe_tensor_func(*args, **kwargs):
                try:
                    return orig_f(*args, **kwargs)
                except RuntimeError as e:
                    if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
                        logger.warning(f"NVML error in {fname} tensor creation, using fallback path")
                        # Create on CPU first, then move to the original device if specified
                        if 'device' in kwargs and kwargs['device'] is not None and str(kwargs['device']).startswith('cuda'):
                            original_device = kwargs['device']
                            kwargs['device'] = 'cpu'
                            tensor = orig_f(*args, **kwargs)
                            return tensor.to(original_device)
                    raise
            return safe_tensor_func
            
        setattr(torch, func_name, make_safe_func(original_func, func_name))
        logger.info(f"Patched torch.{func_name} successfully")

    # Patch distributed module if available
    if hasattr(torch, 'distributed') and hasattr(torch.distributed, 'barrier'):
        original_barrier = torch.distributed.barrier
        
        @wraps(original_barrier)
        def safe_barrier(*args, **kwargs):
            try:
                return original_barrier(*args, **kwargs)
            except RuntimeError as e:
                if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
                    logger.warning("NVML barrier issue detected, using time-based synchronization")
                    import time
                    time.sleep(2)
                    return None
                raise
        
        torch.distributed.barrier = safe_barrier
        logger.info("Patched torch.distributed.barrier successfully")

def apply_all_patches():
    """
    Apply all patches necessary to fix NVML-related issues.
    This function must be called explicitly to apply the patches.
    """
    if torch is not None and torch.cuda.is_available():
        logger.info("CUDA is available - applying patches")
        patch_torch_cuda()
    else:
        logger.info("CUDA not available, no patches applied")

# Apply patches automatically on import
if torch is not None:
    apply_all_patches()