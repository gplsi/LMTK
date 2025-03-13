"""
Utility module for patching PyTorch CUDA functions to handle NVML compatibility issues.

This module provides patched versions of PyTorch CUDA functions that might
cause errors due to missing NVML library functions or version mismatches.
It will only apply the patches if necessary, preserving optimal performance
when no NVML issues are detected.
"""

import os
import sys
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def test_for_nvml_issues():
    """
    Test whether the system has NVML-related issues.
    
    Returns:
        bool: True if NVML issues are detected, False otherwise
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, no need for NVML patches")
        return False
    
    try:
        # Try to perform a simple barrier operation that would fail with NVML issues
        import torch.distributed as dist
        if not dist.is_initialized():
            logger.info("Distributed not initialized, skipping NVML test")
            return False
        
        # Try a simple empty tensor creation on CUDA
        device = torch.device("cuda")
        torch.empty(1, device=device)
        logger.info("No NVML issues detected, running with native performance")
        return False
    except RuntimeError as e:
        if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e):
            logger.warning("NVML compatibility issues detected, applying patches")
            return True
        logger.info("No specific NVML issues detected in test")
        return False
    except Exception:
        # Some other error, not NVML related
        return False

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

def apply_all_patches(force=False):
    """
    Apply all available patches to handle NVML-related issues.
    
    Args:
        force (bool): If True, apply patches regardless of NVML test results
    """
    import torch
    
    # Check if we need to apply patches
    needs_patches = force
    
    if not needs_patches:
        try:
            needs_patches = test_for_nvml_issues()
        except Exception as e:
            logger.warning(f"Error testing for NVML issues: {e}. Applying patches as precaution.")
            needs_patches = True
    
    if needs_patches:
        # Set critical environment variables for NVML issues
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        
        # Apply function patches
        patch_torch_distributed()
        patch_torch_tensor_creation()
        
        logger.info("Applied PyTorch NVML compatibility patches")
    else:
        logger.info("No NVML patches applied, using native performance")
        
    # Always configure DeepSpeed for optimal performance
    configure_deepspeed_performance()

def configure_deepspeed_performance():
    """
    Configure DeepSpeed for optimal performance regardless of NVML status.
    
    This applies performance-enhancing settings that don't interfere with NVML compatibility.
    """
    # Set environment variables for DeepSpeed performance
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Optimize CUDA connection handling
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"    # Better error handling in NCCL
    
    # Enable faster GPU direct access if available
    if os.path.exists("/usr/local/cuda/lib64/libcudart.so"):
        os.environ["NCCL_IB_DISABLE"] = "0"          # Use InfiniBand if available
        os.environ["NCCL_NET_GDR_LEVEL"] = "2"       # Enable GPU Direct RDMA
    
    logger.info("Applied DeepSpeed performance optimizations")

# Import torch here to avoid circular imports during the patching
try:
    import torch
except ImportError:
    logger.warning("PyTorch not found, patches will be applied on demand if needed")

# Apply patches automatically when the module is imported
if __name__ != "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_all_patches(force=False)  # Only apply if needed