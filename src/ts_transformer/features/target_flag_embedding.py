from __future__ import annotations

import torch
from torch import nn


class TargetFlagEmbedding(nn.Module):
    """
    Embedding para distinguir tokens de historia vs token target.

    - Recibe una máscara booleana is_target_mask ∈ {False, True}^{B x L}
      donde True indica que esa posición corresponde al token target.

    - Internamente usa una nn.Embedding de tamaño 2:
        índice 0 -> token de historia
        índice 1 -> token target

    - Devuelve un tensor de embeddings:
        out ∈ R^{B x L x d_model}
    """

    def __init__(self, d_model: int) -> None:
        """
        Parameters
        ----------
        d_model:
            Dimensión de los embeddings de flag (coincide con d_model del Transformer).
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

    def forward(self, is_target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        is_target_mask:
            Tensor bool [B, L], True en posiciones target.

        Returns
        -------
        out:
            Tensor [B, L, d_model], embedding de flag.
        """
        if is_target_mask.ndim != 2:
            raise ValueError(
                f"TargetFlagEmbedding espera is_target_mask 2D [B, L], pero recibió {is_target_mask.shape}."
            )

        # False -> 0 (historia), True -> 1 (target)
        indices = is_target_mask.to(torch.long)  # [B, L]
        out = self.embedding(indices)            # [B, L, d_model]
        return out