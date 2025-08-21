from .chebyshev_kan import replace_roberta_attention_with_chebyshev
from .chebyshev_kan_sparse import replace_roberta_attention_with_chebyshev_sparse
from .chebyshev_kan_orthogonal_projection import replace_roberta_all_layers_with_basis
from .chebyshev_kan_orthogonal_projection_sparse import (
    replace_roberta_all_layers_with_basis_sparse,
)

__all__ = [
    "replace_roberta_attention_with_chebyshev",
    "replace_roberta_attention_with_chebyshev_sparse",
    "replace_roberta_all_layers_with_basis",
    "replace_roberta_all_layers_with_basis_sparse",
]


