from __future__ import annotations

from typing import Optional

import torch.nn as nn


def replace_roberta_all_layers_with_basis_sparse(
    model: nn.Module,
    max_order: int = 5,
    basis_name: str = "dct",
    basis_rank: int = 512,
    sparsity: str = "topk",
    topk: int = 64,
    window: int = 128,
) -> nn.Module:
    """
    Placeholder sparse orthogonal basis attention replacement for all layers.

    No-op implementation that logs the request parameters and returns the model unchanged.
    """
    logger: Optional[object] = getattr(model, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        try:
            logger.info(
                (
                    "[models.attention] Requested sparse orthogonal basis attention (all layers) "
                    f"(max_order={max_order}, basis={basis_name}, rank={basis_rank}, "
                    f"sparsity={sparsity}, topk={topk}, window={window}). Using no-op placeholder."
                )
            )
        except Exception:
            pass
    return model


