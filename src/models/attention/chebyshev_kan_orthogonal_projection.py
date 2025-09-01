from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


def _relative_position_matrix(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]
    denom = max(seq_len - 1, 1)
    x = (rel / denom).clamp(-1, 1)
    return x


def _orthogonal_basis_map(seq_len: int, max_order: int, basis_name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build an orthogonal basis over relative positions for pairs (i, j):
    - "dct": T_k(i,j) = cos(pi * k * (i-j) / (L-1)), k=0..max_order
    - "legendre": T_k(i,j) = P_k( (i-j)/(L-1) ) using recurrence

    Returns tensor of shape [L, L, K] where K = max_order+1.
    """
    x = _relative_position_matrix(seq_len, device, dtype)
    if basis_name.lower() == "dct":
        # DCT-I like basis on relative positions
        k = torch.arange(0, max_order + 1, device=device, dtype=dtype)  # [K]
        # x in [-1,1] -> map to [0,1] to stabilize cos argument
        # Using direct (i-j)/(L-1) already in [-1,1], scale by pi
        # shape: [L,L,1] * [K] -> broadcast to [L,L,K]
        angles = math.pi * x.unsqueeze(-1) * k  # [L, L, K]
        tk = torch.cos(angles)
        return tk
    elif basis_name.lower() == "legendre":
        # Legendre polynomials via recurrence
        t0 = torch.ones_like(x)
        if max_order == 0:
            return t0.unsqueeze(-1)
        t1 = x
        polys = [t0, t1]
        for n in range(2, max_order + 1):
            tn = ((2 * n - 1) * x * polys[-1] - (n - 1) * polys[-2]) / n
            polys.append(tn)
        return torch.stack(polys, dim=-1)  # [L, L, K]
    else:
        # Default fallback to DCT
        k = torch.arange(0, max_order + 1, device=device, dtype=dtype)
        angles = math.pi * x.unsqueeze(-1) * k
        return torch.cos(angles)


class OrthogonalBasisSelfAttention(nn.Module):
    """
    RobertaSelfAttention drop-in that adds an orthogonal basis positional bias per head.
    Uses a basis over relative positions with learnable coefficients.
    """

    def __init__(self, config, orig_self_attn: nn.Module, max_order: int = 5, basis_name: str = "dct", basis_rank: int = 512):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Reuse original projections to keep weights compatible
        self.query = orig_self_attn.query
        self.key = orig_self_attn.key
        self.value = orig_self_attn.value
        self.dropout = orig_self_attn.dropout

        self.max_order = int(max_order)
        # Number of basis functions used for positional bias (cap by max_order+1)
        self.num_basis = min(int(basis_rank), self.max_order + 1)
        self.basis_name = str(basis_name)

        # Learnable coefficients per head and basis index (positional bias)
        self.coefficients = nn.Parameter(torch.zeros(self.num_attention_heads, self.num_basis))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)

        # Projection rank controls additional projection of Q/K per head
        self.projection_rank = max(1, int(basis_rank))
        # Per-head projection weights: [H, D, R]
        self.q_proj_weight = nn.Parameter(torch.empty(self.num_attention_heads, self.attention_head_size, self.projection_rank))
        self.k_proj_weight = nn.Parameter(torch.empty(self.num_attention_heads, self.attention_head_size, self.projection_rank))
        # Safe orthogonal init: perform in float32 on CPU to avoid bfloat16 CUDA QR limitations
        with torch.no_grad():
            def _safe_orthogonal_init(param: torch.Tensor) -> None:
                h, d, r = param.shape
                tmp = torch.empty(h * d, r, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(tmp)
                tmp = tmp.view(h, d, r).to(param.device, dtype=param.dtype)
                param.copy_(tmp)

            _safe_orthogonal_init(self.q_proj_weight)
            _safe_orthogonal_init(self.k_proj_weight)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        return (
            x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _basis_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Build basis up to max_order, then select first num_basis
        full_basis = _orthogonal_basis_map(seq_len, self.max_order, self.basis_name, device, dtype)  # [L,L,K]
        if self.num_basis < full_basis.shape[-1]:
            basis = full_basis[..., : self.num_basis]
        else:
            basis = full_basis
        # Combine coefficients with basis across (i,j): [H,K] x [L,L,K] -> [H,L,L]
        bias = torch.einsum("hk,ijk->hij", self.coefficients, basis)
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
        # Self-attention only
        if encoder_hidden_states is not None:
            raise NotImplementedError("OrthogonalBasisSelfAttention supports self-attention only")

        bsz, tgt_len, _ = hidden_states.size()
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self._shape(query, bsz, tgt_len)
        key = self._shape(key, bsz, tgt_len)
        value = self._shape(value, bsz, tgt_len)

        # Project Q/K per head into rank-R basis space and compute attention in that space
        # query/key: [B, H, L, D], proj weights: [H, D, R] => [B, H, L, R]
        q_proj = torch.einsum("bhld,hdr->bhlr", query, self.q_proj_weight)
        k_proj = torch.einsum("bhld,hdr->bhlr", key, self.k_proj_weight)
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.projection_rank)

        # Add orthogonal basis positional bias per head
        bias = self._basis_bias(tgt_len, attn_weights.device, attn_weights.dtype)  # [H,L,L]
        attn_weights = attn_weights + bias.unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        attn_output = context.view(bsz, tgt_len, self.all_head_size)

        outputs: Tuple[torch.Tensor, ...] = (attn_output,)
        if output_attentions:
            outputs = outputs + (attn_probs,)
        return outputs  # type: ignore[return-value]


def replace_roberta_all_layers_with_basis(
    model: nn.Module, max_order: int = 5, basis_name: str = "dct", basis_rank: int = 512
) -> nn.Module:
    """
    Replace each RobertaSelfAttention with an orthogonal-basis-augmented variant.
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
        repl = OrthogonalBasisSelfAttention(
            model.config, orig_self, max_order=max_order, basis_name=basis_name, basis_rank=basis_rank
        )
        layer.attention.self = repl
    return model


