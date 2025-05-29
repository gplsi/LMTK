import inspect
from functools import wraps
import multiprocessing

import psutil

def inherit_init_params(cls):
    base_init = cls.__bases__[0].__init__
    sig = inspect.signature(base_init)
    
    @wraps(base_init)
    def new_init(self, *args, **kwargs):
        return base_init(self, *args, **kwargs)
    
    new_init.__signature__ = sig
    cls.__init__ = new_init
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
