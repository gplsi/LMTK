from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


class ChebyshevSelfAttention(nn.Module):
    """
    Drop-in replacement for HuggingFace RobertaSelfAttention that adds a Chebyshev
    positional bias per head. The core projections (q/k/v/out) are identical to the
    baseline, and we introduce a small number of extra learnable parameters:
      - alphas: [num_heads, order+1]
    Bias term B_h(i,j) = sum_{k=0..order} alphas[h,k] * T_k(x_{i,j}) where
    x_{i,j} is the normalized relative distance in [-1, 1] between token i and j.
    """

    def __init__(self, config, orig_self_attn: nn.Module, order: int = 5) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Reuse original projections to keep weights compatible
        self.query = orig_self_attn.query
        self.key = orig_self_attn.key
        self.value = orig_self_attn.value
        self.dropout = orig_self_attn.dropout
        # Output projection is handled by RobertaSelfOutput (attention.output.dense),
        # so ChebyshevSelfAttention must return the context without projecting here.

        self.order = int(order)
        # Small learnable parameter set
        self.alphas = nn.Parameter(torch.zeros(self.num_attention_heads, self.order + 1))
        nn.init.normal_(self.alphas, mean=0.0, std=0.02)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        return (
            x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    @staticmethod
    def _chebyshev_polys(x: torch.Tensor, max_order: int) -> torch.Tensor:
        """
        Compute T_k(x) for k=0..max_order for a tensor x in [-1,1].
        Returns shape [..., max_order+1].
        """
        t0 = torch.ones_like(x)
        if max_order == 0:
            return t0.unsqueeze(-1)
        t1 = x
        polys = [t0, t1]
        for k in range(2, max_order + 1):
            tk = 2.0 * x * polys[-1] - polys[-2]
            polys.append(tk)
        return torch.stack(polys, dim=-1)

    def _chebyshev_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Relative distances i-j normalized to [-1, 1]
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]
        denom = max(seq_len - 1, 1)
        x = (rel / denom).clamp(-1, 1)
        # Compute T_k(x) => [L, L, K]
        tk = self._chebyshev_polys(x, self.order)
        # Combine with per-head alphas: bias[h, i, j] = sum_k alpha[h,k] * T_k(x_{i,j})
        # alphas: [H, K], tk: [L, L, K] => bias: [H, L, L]
        # Combine per-head alphas with Chebyshev basis across (i,j)
        bias = torch.einsum("hk,ijk->hij", self.alphas, tk)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Only self-attention path is supported here
        is_cross = encoder_hidden_states is not None
        if is_cross:
            raise NotImplementedError("ChebyshevSelfAttention currently supports self-attention only")

        bsz, tgt_len, _ = hidden_states.size()
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self._shape(query, bsz, tgt_len)
        key = self._shape(key, bsz, tgt_len)
        value = self._shape(value, bsz, tgt_len)

        # Scaled dot-product attention logits
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.attention_head_size)

        # Add Chebyshev positional bias per head: shape [H, L, L] -> [B, H, L, L]
        bias = self._chebyshev_bias(tgt_len, attn_weights.device, attn_weights.dtype)
        attn_weights = attn_weights + bias.unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # broadcasted

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value)  # [B, H, L, D]
        context = context.permute(0, 2, 1, 3).contiguous()
        attn_output = context.view(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if output_attentions:
            outputs = outputs + (attn_probs,)
        # No caching currently
        return outputs  # type: ignore[return-value]


def replace_roberta_attention_with_chebyshev(model: nn.Module, order: int = 5) -> nn.Module:
    """
    Replace each RobertaSelfAttention with ChebyshevSelfAttention (self-attention only).
    Copies over projection weights to preserve initialization and adds small
    learnable alphas per head and polynomial degree.
    """
    if not hasattr(model, "roberta"):
        return model
    encoder = getattr(model.roberta, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        return model

    for layer in encoder.layer:
        attn = layer.attention
        orig_self = getattr(attn, "self", None)
        if orig_self is None:
            continue
        # Build replacement
        cheb_self = ChebyshevSelfAttention(model.config, orig_self, order=order)
        # Swap
        layer.attention.self = cheb_self
    return model


