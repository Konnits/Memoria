from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RegressionHead(nn.Module):
    """
    Cabeza de salida para tareas de regresión sobre el token target.

    Toma un vector de representación de dimensión d_model y lo mapea
    a un vector de dimensión output_dim (por ejemplo, [temp, pres]).

    Opcionalmente puede incluir una pequeña MLP en lugar de una sola
    capa lineal, si se desea más capacidad.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            # Sólo una capa lineal
            self.net = nn.Linear(d_model, output_dim)
        else:
            if activation == "relu":
                act = nn.ReLU()
            elif activation == "gelu":
                act = nn.GELU()
            else:
                raise ValueError(f"Activación no soportada en RegressionHead: {activation}")

            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor de entrada, shape [..., d_model].

        Returns
        -------
        out:
            Tensor de salida, shape [..., output_dim].
        """
        return self.net(x)