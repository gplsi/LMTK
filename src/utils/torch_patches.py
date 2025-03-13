"""
Utility module for patching PyTorch CUDA functions to handle NVML compatibility issues.

This module provides patched versions of PyTorch CUDA functions that might
cause errors due to missing NVML library functions or version mismatches.
"""

import os
import sys
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def patch_torch_distributed():
    """
    Apply patches to torch.distributed to avoid NVML-related errors.
    
    This function patches the torch.distributed.barrier function to handle
    NVML-related errors gracefully, allowing distributed training to work
    even with NVIDIA driver/NVML version mismatches.
    """
    try:
        import torch.distributed

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
                else:
                    raise
        
        # Replace the original barrier with our safe version
        torch.distributed.barrier = safe_barrier
        logger.info("Successfully patched torch.distributed.barrier")
        
    except ImportError:
        logger.warning("torch.distributed not available, skipping patch")
        pass

def patch_torch_tensor_creation():
    """
    Apply patches to torch.empty and related functions to avoid NVML errors during tensor creation.
    
    This is particularly important for DeepSpeed which uses these functions during model initialization.
    """
    try:
        import torch
        
        original_empty = torch.empty
        
        @wraps(original_empty)
        def safe_empty(*args, **kwargs):
            try:
                return original_empty(*args, **kwargs)
            except RuntimeError as e:
                if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
                    logger.warning("NVML error detected during tensor creation, retrying with different device order")
                    # Try to use a different device allocation strategy
                    if 'device' in kwargs and kwargs['device'].type == 'cuda':
                        # Keep track of original device
                        original_device = kwargs['device']
                        # First create on CPU
                        kwargs['device'] = 'cpu'
                        tensor = original_empty(*args, **kwargs)
                        # Then move to original device
                        return tensor.to(original_device)
                    else:
                        raise
                else:
                    raise
        
        # Replace the original tensor creation functions
        torch.empty = safe_empty
        logger.info("Successfully patched torch.empty")
        
    except ImportError:
        logger.warning("torch not available, skipping tensor creation patch")
        pass

def apply_all_patches():
    """Apply all available patches to handle NVML-related issues."""
    # Set critical environment variables
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # Apply function patches
    patch_torch_distributed()
    patch_torch_tensor_creation()
    
    logger.info("Applied all PyTorch NVML compatibility patches")

# Apply patches automatically when the module is imported
if __name__ != "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_all_patches()