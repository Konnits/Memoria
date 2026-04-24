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


class AttentionPooling(nn.Module):
    """
    Pooling atencional sobre la secuencia para obtener un vector global.

    Esta capa aprende pesos por token y permite fusionar contexto global con
    el estado del token target en tareas de forecasting.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(32, d_model // 2)
        self.proj = nn.Linear(d_model, h)
        self.score = nn.Linear(h, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Tensor [B, L, d_model].
        valid_mask:
            Máscara bool opcional [B, L] con True en posiciones válidas.

        Returns
        -------
        pooled:
            Tensor [B, d_model] con el resumen global.
        alpha:
            Tensor [B, L] con pesos de atención.
        """
        scores = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # [B, L]

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-1e4"))

        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.sum(alpha.unsqueeze(-1) * x, dim=1)
        return pooled, alpha