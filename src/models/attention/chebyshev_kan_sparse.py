from __future__ import annotations

from typing import Optional

import torch.nn as nn


def replace_roberta_attention_with_chebyshev_sparse(
    model: nn.Module, order: int = 5, sparsity: str = "topk", topk: int = 64, window: int = 128
) -> nn.Module:
    """
    Placeholder sparse Chebyshev attention replacement for RoBERTa models.

    This is a no-op that logs the request (if a logger is available) and returns the model unchanged.
    It allows experiments to run even without the custom implementation present.
    """
    logger: Optional[object] = getattr(model, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        try:
            logger.info(
                (
                    "[models.attention] Requested Sparse Chebyshev attention "
                    f"(order={order}, sparsity={sparsity}, topk={topk}, window={window}). Using no-op placeholder."
                )
            )
        except Exception:
            pass
    return model


