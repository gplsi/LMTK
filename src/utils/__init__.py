import inspect
from functools import wraps

def inherit_init_params(cls):
    base_init = cls.__bases__[0].__init__
    sig = inspect.signature(base_init)
    
    @wraps(base_init)
    def new_init(self, *args, **kwargs):
        return base_init(self, *args, **kwargs)
    
    new_init.__signature__ = sig
    cls.__init__ = new_init
    return cls
