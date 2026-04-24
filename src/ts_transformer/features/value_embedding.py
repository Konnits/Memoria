from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    """
    Proyección de features continuas multivariadas a la dimensión del modelo.

    - Entrada:  x ∈ R^{B x L x d_in}
      (B = batch_size, L = seq_len, d_in = número de variables de entrada:
       temperatura, presión, etc.)

    - Salida:   e ∈ R^{B x L x d_model}

    Está pensada para ser flexible en d_in:
      * si quieres predecir (temperatura, presión), d_in = 2
      * si agregas más sensores, sólo cambias d_in al instanciar la clase.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        use_layernorm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        d_in:
            Dimensión de entrada (número de features por timestamp).
        d_model:
            Dimensión interna del Transformer.
        use_layernorm:
            Si True, aplica LayerNorm sobre la dimensión de modelo
            después de la proyección lineal.
        """
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model

        self.proj = nn.Linear(d_in, d_model)
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(d_model) if use_layernorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor de entrada, shape [B, L, d_in].

        Returns
        -------
        out:
            Tensor de salida, shape [B, L, d_model].
        """
        if x.ndim != 3:
            raise ValueError(
                f"FeatureEmbedding espera entrada 3D [B, L, d_in], pero recibió shape {x.shape}."
            )
        if x.size(-1) != self.d_in:
            raise ValueError(
                f"La última dimensión de x debe ser d_in={self.d_in}, "
                f"pero se obtuvo {x.size(-1)}."
            )

        out = self.proj(x)
        if self.use_layernorm and self.ln is not None:
            out = self.ln(out)
        return out