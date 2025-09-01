from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn

from .chebyshev_kan_orthogonal_projection import _orthogonal_basis_map


def _build_sparsity_mask(seq_len: int, sparsity: str, topk: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Build a boolean mask [L, L] indicating which positions are kept (True) under the sparsity scheme.
    - topk: keep top-k nearest neighbors around each token i by absolute distance |i-j|
    - window: keep a fixed window of size W around the diagonal: |i-j| <= window
    """
    idx = torch.arange(seq_len, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [L, L]
    if sparsity == "topk":
        # For each row i, keep the k smallest distances
        # Compute ranks per row
        ranks = dist.argsort(dim=-1)
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        k = min(topk, seq_len)
        cols = ranks[:, :k]  # [L, k]
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, k)
        mask[row_idx, cols] = True
        return mask
    else:
        # windowed scheme
        return dist <= max(0, int(window))


class SparseOrthogonalBasisSelfAttention(nn.Module):
    """
    RobertaSelfAttention drop-in adding orthogonal basis bias with sparsity mask.
    """

    def __init__(
        self,
        config,
        orig_self_attn: nn.Module,
        max_order: int = 5,
        basis_name: str = "dct",
        basis_rank: int = 512,
        sparsity: str = "topk",
        topk: int = 64,
        window: int = 128,
    ) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = orig_self_attn.query
        self.key = orig_self_attn.key
        self.value = orig_self_attn.value
        self.dropout = orig_self_attn.dropout

        self.max_order = int(max_order)
        self.num_basis = min(int(basis_rank), self.max_order + 1)
        self.basis_name = str(basis_name)

        self.sparsity = str(sparsity)
        self.topk = int(topk)
        self.window = int(window)

        self.coefficients = nn.Parameter(torch.zeros(self.num_attention_heads, self.num_basis))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)

        # Projection rank controls additional projection of Q/K per head
        self.projection_rank = max(1, int(basis_rank))
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
        full_basis = _orthogonal_basis_map(seq_len, self.max_order, self.basis_name, device, dtype)  # [L,L,K]
        basis = full_basis[..., : self.num_basis]
        bias = torch.einsum("hk,ijk->hij", self.coefficients, basis)  # [H,L,L]
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
        if encoder_hidden_states is not None:
            raise NotImplementedError("SparseOrthogonalBasisSelfAttention supports self-attention only")

        bsz, tgt_len, _ = hidden_states.size()
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self._shape(query, bsz, tgt_len)
        key = self._shape(key, bsz, tgt_len)
        value = self._shape(value, bsz, tgt_len)

        # Compute attention in projected rank space
        q_proj = torch.einsum("bhld,hdr->bhlr", query, self.q_proj_weight)
        k_proj = torch.einsum("bhld,hdr->bhlr", key, self.k_proj_weight)
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.projection_rank)

        bias = self._basis_bias(tgt_len, attn_weights.device, attn_weights.dtype)

        # Apply sparsity mask to the bias before adding
        mask_2d = _build_sparsity_mask(tgt_len, self.sparsity, self.topk, self.window, attn_weights.device)  # [L,L]
        # Where mask is False, set bias to 0
        bias = bias * mask_2d.unsqueeze(0)  # [H,L,L]

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


def replace_roberta_all_layers_with_basis_sparse(
    model: nn.Module,
    max_order: int = 5,
    basis_name: str = "dct",
    basis_rank: int = 512,
    sparsity: str = "topk",
    topk: int = 64,
    window: int = 128,
) -> nn.Module:
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
        repl = SparseOrthogonalBasisSelfAttention(
            model.config,
            orig_self,
            max_order=max_order,
            basis_name=basis_name,
            basis_rank=basis_rank,
            sparsity=sparsity,
            topk=topk,
            window=window,
        )
        layer.attention.self = repl
    return model


