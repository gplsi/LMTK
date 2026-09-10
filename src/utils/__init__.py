import inspect
from functools import wraps
import multiprocessing

import psutil

def inherit_init_params(cls):
    """
    Decorator that keeps the subclass' __init__ implementation while exposing the
    signature of the parent __init__. This prevents accidentally discarding the
    subclass initialisation logic while still providing the expected signature for
    external tooling (e.g. CLI autocompletion).
    """
    if not cls.__bases__:
        return cls

    base_init = getattr(cls.__bases__[0], "__init__", None)
    original_init = cls.__init__

    if base_init is not None:
        try:
            signature = inspect.signature(base_init)
        except (TypeError, ValueError):
            signature = inspect.Signature()
    else:
        signature = inspect.Signature()

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        return original_init(self, *args, **kwargs)

    wrapped_init.__signature__ = signature
    cls.__init__ = wrapped_init
    return cls

def get_optimal_thread_count():
    """Get optimal thread count for Rayon based on system configuration"""
    logical_cores = multiprocessing.cpu_count()
    
    try:
        # Try to get physical core count (more accurate for CPU-bound tasks)
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores:
            return physical_cores
    except:
        pass
    
    # Fallback to logical cores
    return logical_cores
