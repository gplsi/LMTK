from __future__ import annotations

from typing import Optional

import torch.nn as nn


def replace_roberta_ffn_with_kan(model: nn.Module, order: int = 3) -> nn.Module:
    """
    Placeholder FFN replacement for RoBERTa models.

    This function is a safe no-op: it logs intent (if the model exposes a logger)
    and returns the model unchanged. It exists so experiments that reference
    KAN-based FFN replacements can run without the full TraceableFormer modules.

    Args:
        model: A Hugging Face RoBERTaForMaskedLM or compatible model instance.
        order: Polynomial order parameter accepted for compatibility.

    Returns:
        The unmodified model.
    """
    # If the model carries a logger attribute (e.g., via Lightning wrappers), log the intent
    logger: Optional[object] = getattr(model, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        try:
            logger.info(
                f"[models.ffn] Requested KAN FFN replacement (order={order}). Using no-op placeholder."
            )
        except Exception:
            pass
    return model


