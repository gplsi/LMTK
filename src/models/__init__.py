"""
Reusable model components and architecture modification utilities.

This package provides attention and feed-forward network (FFN) replacement
helpers referenced by training configs. The default implementations provided
here are safe no-ops that keep the original model unchanged while logging what
would have been modified. They are intended to make experiments runnable even
when specialized modules are not available.

If you plan to experiment with custom attention/FFN blocks (e.g., KAN-based),
extend the functions in subpackages to actually modify the Hugging Face model
structure and return the updated model.
"""

__all__ = [
    "attention",
    "ffn",
]


