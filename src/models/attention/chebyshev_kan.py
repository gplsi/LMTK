from __future__ import annotations

from typing import Optional

import torch.nn as nn


def replace_roberta_attention_with_chebyshev(model: nn.Module, order: int = 5) -> nn.Module:
    """
    Placeholder attention replacement for RoBERTa models using a Chebyshev-based kernel.

    This default implementation is a no-op that logs intent (if possible) and returns the
    model unchanged. It exists so that experiments depending on this hook will still run
    without requiring the full custom attention implementation.

    Args:
        model: A Hugging Face RoBERTa model (e.g., RobertaForMaskedLM).
        order: Chebyshev polynomial order to use (ignored here).

    Returns:
        The unmodified model.
    """
    logger: Optional[object] = getattr(model, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        try:
            logger.info(
                f"[models.attention] Requested Chebyshev attention (order={order}). Using no-op placeholder."
            )
        except Exception:
            pass
    return model


