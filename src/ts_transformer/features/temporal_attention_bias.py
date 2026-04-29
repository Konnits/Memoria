from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemporalAttentionBias(nn.Module):
    """
    Bias de atención basado en diferencias temporales entre tokens.

    Agrega un sesgo aditivo a los scores de atención proporcional a
    la distancia temporal entre cada par de tokens:

        attention_score(i, j) = (Q_i · K_j) / √d_head  +  bias(Δt_{i,j})

    Donde bias(Δt) = -|Δt| / exp(log_tau_h) para cada cabeza h.

    Esto permite que cada cabeza de atención aprenda una escala
    temporal diferente:
      - Cabezas con τ pequeño → foco en observaciones cercanas
      - Cabezas con τ grande → foco en dependencias de largo plazo

    Especialmente relevante para series de tiempo irregulares, donde
    la distancia temporal (no ordinal) determina la relación entre
    observaciones.

    Inspirado en ALiBi (Press et al., 2022) y STAR-Set Transformer
    (Horn et al., 2024), adaptado para timestamps continuos.
    """

    def __init__(
        self,
        num_heads: int,
        init_tau_min: float = 8.0,
        init_tau_max: float = 128.0,
    ) -> None:
        """
        Parameters
        ----------
        num_heads:
            Número de cabezas de atención.
        init_tau_min:
            Escala temporal mínima inicial en unidades ya normalizadas por
            time_scale.
        init_tau_max:
            Escala temporal máxima inicial en unidades ya normalizadas por
            time_scale.
        """
        super().__init__()
        self.num_heads = num_heads
        if init_tau_min <= 0.0 or init_tau_max <= 0.0:
            raise ValueError("init_tau_min e init_tau_max deben ser > 0.")
        if init_tau_min > init_tau_max:
            raise ValueError("init_tau_min no puede ser mayor que init_tau_max.")

        init_tau = torch.logspace(
            torch.log10(torch.tensor(init_tau_min, dtype=torch.float32)),
            torch.log10(torch.tensor(init_tau_max, dtype=torch.float32)),
            steps=num_heads,
        )
        self.log_tau = nn.Parameter(init_tau.log())

    def forward(
        self,
        timestamps: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps:
            Tensor [B, L] con timestamps (ya normalizados por time_scale).

        Returns
        -------
        bias:
            Tensor [B, num_heads, L, L] con el sesgo temporal aditivo.
        """
        output_dtype = dtype if dtype is not None else timestamps.dtype
        compute_dtype = (
            output_dtype
            if output_dtype in {torch.float16, torch.bfloat16, torch.float32}
            else torch.float32
        )

        # Diferencias temporales absolutas: |t_i - t_j|
        # timestamps: [B, L] -> dt: [B, L, L]
        dt = (
            timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)
        ).abs().to(compute_dtype)

        # Escalas aprendidas por cabeza: [num_heads]
        # Clamp log_tau a [-5, 5] para mantener tau en [~0.007, ~148],
        # evitando que tau → 0 cause bias → -inf (y NaN en softmax).
        tau = self.log_tau.clamp(-5.0, 5.0).exp().to(compute_dtype)  # positivo garantizado

        # Penalización logarítmica suave: conserva contexto lejano en el arranque
        # y evita castigos excesivos cuando la ventana temporal es larga.
        # dt: [B, L, L] -> [B, 1, L, L]
        # tau: [H] -> [1, H, 1, 1]
        scaled_dt = dt.unsqueeze(1) / (tau.view(1, -1, 1, 1) + 1e-8)
        bias = -torch.log1p(scaled_dt)

        # Clamp final para evitar valores extremos en el softmax
        bias = bias.clamp(-12.0, 0.0).to(output_dtype)

        return bias  # [B, H, L, L]
