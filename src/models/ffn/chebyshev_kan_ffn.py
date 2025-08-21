from typing import Optional

import torch
import torch.nn as nn

try:
    from kan.nn.global_control.basis import OrthogonalPolynomial
except Exception as e:  # pragma: no cover - allow import-time diagnostics
    OrthogonalPolynomial = None  # type: ignore


class ChebyshevMixing(nn.Module):
    """
    Chebyshev polynomial KAN mixing layer that maps hidden_size -> hidden_size.

    This layer expands each feature with Chebyshev polynomial basis and mixes them
    across features to produce the same-dimensional output, suitable as an FFN
    replacement within a Transformer block without widening the hidden size.
    """

    def __init__(
        self,
        hidden_size: int,
        order: int = 5,
        grid_epsilon: float = 1e-6,
        use_input_layernorm: bool = True,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if OrthogonalPolynomial is None:
            raise ImportError(
                "kan library not available. Ensure it is installed and importable."
            )

        self.hidden_size = hidden_size
        self.order = order
        self.grid_epsilon = grid_epsilon

        self.input_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps) if use_input_layernorm else None
        )

        # Chebyshev (first kind) basis generator
        self.chebyshev_basis = OrthogonalPolynomial(
            polynomial="chebyshev_first", order=order
        )

        # Learnable mixing weights across input features and polynomial degrees
        # Shape: [in_features, out_features, order+1] with in=out=hidden_size
        self.weights = nn.Parameter(
            torch.empty(hidden_size, hidden_size, order + 1)
        )
        nn.init.kaiming_uniform_(self.weights, a=5**0.5)  # similar to Linear init

        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, hidden_size] or [batch, hidden_size].

        Returns:
            Tensor with the same shape as input.
        """
        original_shape = x.shape

        if x.dim() == 2:
            b_t, h = x.shape
            x_2d = x
        elif x.dim() == 3:
            b, t, h = x.shape
            x_2d = x.reshape(b * t, h)
        else:
            raise ValueError(
                f"Unsupported input shape {x.shape}. Expected [B,H] or [B,T,H]."
            )

        if self.input_norm is not None:
            x_2d = self.input_norm(x_2d)

        # Clamp into Chebyshev domain [-1, 1]
        x_2d = torch.clamp(
            x_2d, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon
        )

        # Ensure basis lives on the same device as inputs
        if getattr(self.chebyshev_basis, "device", None) != x_2d.device:
            self.chebyshev_basis.to(device=x_2d.device)

        # Compute basis: [B*T, H, order+1]
        basis_values = self.chebyshev_basis.calculate_basis(x_2d)

        # Mix across input dims and polynomial degrees -> [B*T, H]
        # einsum over (input_dim i) and polynomial degree p: 'b i p , i o p -> b o'
        out_2d = torch.einsum("bip,iop->bo", basis_values, self.weights)
        if self.bias is not None:
            out_2d = out_2d + self.bias

        if len(original_shape) == 3:
            out = out_2d.reshape(original_shape[0], original_shape[1], original_shape[2])
        else:
            out = out_2d

        return out


class ChebyshevKANFFN(nn.Module):
    """
    RoBERTa-style FFN substitute using Chebyshev polynomial KAN, keeping hidden size constant.

    Structure:
      x -> KAN(Chebyshev) [hidden->hidden] -> dropout -> Linear(hidden->hidden) -> dropout -> Add&LayerNorm

    This mirrors the two-step FFN with an activation, but replaces the widening+activation
    with a same-width Chebyshev-KAN transformation.
    """

    def __init__(
        self,
        hidden_size: int,
        order: int = 5,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_input_layernorm_in_kan: bool = True,
    ) -> None:
        super().__init__()

        self.mixing = ChebyshevMixing(
            hidden_size=hidden_size,
            order=order,
            use_input_layernorm=use_input_layernorm_in_kan,
            layer_norm_eps=layer_norm_eps,
        )
        self.dropout1 = nn.Dropout(hidden_dropout_prob)

        # Final projection akin to BertOutput.dense
        self.dense = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

        self.dropout2 = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is None:
            residual = hidden_states

        x = self.mixing(hidden_states)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.layer_norm(x + residual)
        return x


class RobertaKANIntermediate(nn.Module):
    """
    Drop-in replacement for Hugging Face RobertaIntermediate that returns same-dim output.
    """

    def __init__(self, config, order: int = 5) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.kan = ChebyshevMixing(
            hidden_size=config.hidden_size,
            order=order,
            use_input_layernorm=True,
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.kan(hidden_states)


class RobertaKANOutput(nn.Module):
    """
    Drop-in replacement for Hugging Face RobertaOutput that applies Linear + Dropout + Add&LN.
    """

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-5))
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.dense(hidden_states)
        x = self.dropout(x)
        x = self.LayerNorm(x + input_tensor)
        return x


def replace_roberta_ffn_with_kan(model: nn.Module, order: int = 5) -> nn.Module:
    """
    Replace the FFN (Intermediate+Output) of each RobertaLayer with KAN-based versions.

    Args:
        model: A Hugging Face RobertaPreTrainedModel or similar with attribute `roberta.encoder.layer`.
        order: Chebyshev polynomial order.

    Returns:
        The model with in-place replaced FFN blocks.
    """
    if not hasattr(model, "roberta"):
        raise AttributeError("Expected model to have attribute 'roberta'.")
    encoder = getattr(model.roberta, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        raise AttributeError("Expected model.roberta.encoder.layer to exist.")

    config = getattr(model, "config", None)
    if config is None:
        raise AttributeError("Model missing config attribute required for dimensions.")

    for layer in encoder.layer:
        layer.intermediate = RobertaKANIntermediate(config=config, order=order)
        layer.output = RobertaKANOutput(config=config)

    return model


