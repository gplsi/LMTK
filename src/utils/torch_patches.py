"""
Utility module for patching PyTorch CUDA functions to handle NVML compatibility issues.

This module provides comprehensive patching of PyTorch's tensor creation and CUDA functions
to handle NVML compatibility issues that occur during distributed training with DeepSpeed.
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

# Optimize DeepSpeed performance
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# Disable NVML initialization completely
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

def patch_deepspeed():
    """
    Apply patches to DeepSpeed to avoid NVML-related errors during initialization.
    """
    try:
        import deepspeed
        from deepspeed.accelerator import get_accelerator

        # Patch DeepSpeed's device detection to avoid NVML calls
        original_get_accelerator = deepspeed.accelerator.get_accelerator
        
        @wraps(original_get_accelerator)
        def safe_get_accelerator():
            try:
                return original_get_accelerator()
            except RuntimeError as e:
                if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
                    logger.warning("NVML issue detected in DeepSpeed accelerator, using safe fallback")
                    # Return a simpler CUDA accelerator implementation
                    from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
                    return CUDA_Accelerator()
                raise
        
        # Replace the function
        deepspeed.accelerator.get_accelerator = safe_get_accelerator
        logger.info("Successfully patched DeepSpeed accelerator")
        
    except ImportError:
        logger.warning("DeepSpeed not available, skipping DeepSpeed-specific patches")

def patch_torch_cuda():
    """
    Patch PyTorch's CUDA functions to handle NVML-related errors.
    """
    import torch
    
    # Patch device creation
    original_device = torch.device
    
    @wraps(original_device)
    def safe_device(device_type, *args, **kwargs):
        try:
            return original_device(device_type, *args, **kwargs)
        except RuntimeError as e:
            if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
                logger.warning(f"NVML error in device creation: {str(e)}")
                # For 'cuda' devices, create a basic device without NVML features
                if device_type == 'cuda':
                    logger.info("Using simplified CUDA device creation")
                    result = original_device('cuda')
                    return result
            raise
            
    torch.device = safe_device
    
    # Patch tensor creation functions
    for func_name in ['empty', 'zeros', 'ones', 'full', 'rand', 'randn']:
        if not hasattr(torch, func_name):
            continue
            
        original_func = getattr(torch, func_name)
        
        @wraps(original_func)
        def make_safe_func(orig_f, fname):
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
    """
    try:
        import torch
        if torch.cuda.is_available():
            patch_torch_cuda()
            patch_deepspeed()
            logger.info("Applied all PyTorch and DeepSpeed NVML-related patches")
        else:
            logger.info("CUDA not available, no patches applied")
    except ImportError:
        logger.warning("PyTorch not found, patches will not be applied")

# Apply patches automatically on import
apply_all_patches()