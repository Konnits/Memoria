from __future__ import annotations

from typing import Literal, Optional

import math
import torch
from torch import nn


class Time2Vec(nn.Module):
    """
    Time2Vec: encoding temporal con componentes aprendidos.

    Combina un componente lineal (captura tendencias aperiódicas)
    con d_model-1 componentes periódicos (captura patrones cíclicos
    a frecuencias aprendidas).

    Referencia: Kazemi et al., "Time2Vec: Representing Time in a
    Principled Framework", ICLR 2019.

    Formulación:
        Time2Vec(τ)[0]   = ω_0 · τ + φ_0       (lineal)
        Time2Vec(τ)[i>0] = sin(ω_i · τ + φ_i)  (periódico)

    A diferencia del sinusoidal fijo, las frecuencias ω y fases φ
    se aprenden durante el entrenamiento, adaptándose a la escala
    temporal de cada dataset.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        # 1 componente lineal + (d_model-1) componentes periódicos
        self.linear_weight = nn.Parameter(torch.randn(1) * 0.01)
        self.linear_bias = nn.Parameter(torch.zeros(1))

        n_periodic = d_model - 1
        # Inicializar frecuencias en un rango diverso para cubrir
        # múltiples escalas temporales desde el inicio.
        self.periodic_weights = nn.Parameter(
            torch.randn(n_periodic) * 0.1
        )
        self.periodic_biases = nn.Parameter(torch.zeros(n_periodic))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tau:
            Tensor [B, L] con tiempos relativos normalizados.

        Returns
        -------
        enc:
            Tensor [B, L, d_model].
        """
        tau_fp32 = tau.to(torch.float32)
        tau_exp = tau_fp32.unsqueeze(-1)  # [B, L, 1]

        # Mantener el canal aperiódico realmente lineal. Lo computamos en fp32
        # para evitar problemas numéricos bajo AMP sin saturar su señal con tanh.
        linear = tau_exp * self.linear_weight.to(torch.float32) + self.linear_bias.to(torch.float32)

        # Componentes periódicos: [B, L, n_periodic]
        periodic = torch.sin(
            tau_exp * self.periodic_weights.to(torch.float32)
            + self.periodic_biases.to(torch.float32)
        )

        return torch.cat([linear, periodic], dim=-1)  # [B, L, d_model]


class TimePositionalEncoding(nn.Module):
    """
    Encoding temporal continuo para timestamps no equiespaciados.

    Recibe timestamps numéricos (por ejemplo en segundos) y los transforma
    en vectores de dimensión d_model, de forma análoga al positional encoding
    sinusoidal del Transformer, pero usando tiempos reales.

    Idea básica:
      - Para cada secuencia en el batch, se calcula un tiempo relativo:
            τ = (t - t_0) / time_scale
        donde:
            t_0  = primer timestamp de la secuencia
            time_scale = hiperparámetro (por ejemplo 900 s = 15 min).
      - Sobre τ se aplica un encoding sinusoidal, MLP o Time2Vec.

    Modo por defecto: "sinusoidal".
    """

    def __init__(
        self,
        d_model: int,
        time_scale: float = 1.0,
        mode: Literal["sinusoidal", "mlp", "time2vec"] = "sinusoidal",
        time_transform: Literal["linear", "log1p"] = "log1p",
        mlp_hidden_dim: int = 64,
    ) -> None:
        """
        Parameters
        ----------
        d_model:
            Dimensión interna del Transformer.
        time_scale:
            Escala de tiempo para normalizar los timestamps.
            Si t está en segundos y time_scale = 900, entonces
                τ = (t - t_0) / 900
            hace que un "paso unitario" corresponda a 15 minutos.
        mode:
            "sinusoidal": encoding sinusoidal continuo (fijo).
            "mlp": encoding aprendido mediante un MLP sobre τ.
            "time2vec": encoding Time2Vec con componentes aprendidos.
        time_transform:
            Transformación previa aplicada a τ = (t - t0) / time_scale.
            "linear": usa τ sin modificar.
            "log1p": usa log1p(clamp(τ, min=0)), más robusto cuando hay
            horizontes o escalas temporales largas.
        mlp_hidden_dim:
            Dimensión oculta en el caso "mlp".
        """
        super().__init__()

        self.d_model = d_model
        self.time_scale = float(time_scale)
        self.mode = mode
        self.time_transform = time_transform

        if self.time_scale <= 0.0:
            raise ValueError("time_scale debe ser > 0.")

        if self.time_transform not in {"linear", "log1p"}:
            raise ValueError("time_transform debe ser 'linear' o 'log1p'.")

        if mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(1, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, d_model),
            )
        elif mode == "time2vec":
            self.time2vec = Time2Vec(d_model)
        else:
            self.mlp = None

        # Precalcular los "divisors" para el encoding sinusoidal
        if mode == "sinusoidal":
            # indices = [0, 1, 2, ..., d_model-1]
            positions = torch.arange(0, d_model, dtype=torch.float32).view(1, -1)
            # El denominador 10000^{2k/d_model} se aplica sólo a pares,
            # pero construimos el factor completo y luego dividimos.
            div_term = torch.exp(
                -math.log(10000.0) * (positions // 2 * 2) / d_model
            )  # //2*2 hace que par/impar compartan escala
            self.register_buffer("div_term", div_term)  # [1, d_model]
        else:
            self.register_buffer("div_term", torch.empty(1, d_model))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps:
            Tensor [B, L] con timestamps numéricos (float).
            Debe estar ordenado en el tiempo para cada secuencia.

        Returns
        -------
        enc:
            Tensor [B, L, d_model] con el encoding temporal.
        """
        if timestamps.ndim != 2:
            raise ValueError(
                f"TimePositionalEncoding espera timestamps 2D [B, L], pero recibió {timestamps.shape}."
            )

        B, L = timestamps.shape

        # t0: primer timestamp de cada secuencia, shape [B, 1]
        t0 = timestamps[:, :1]
        # τ = (t - t0) / time_scale
        tau = (timestamps - t0) / self.time_scale  # [B, L]

        if self.time_transform == "log1p":
            tau = torch.log1p(torch.clamp(tau, min=0.0))

        if self.mode == "sinusoidal":
            # tau: [B, L] -> [B, L, 1]
            tau_expanded = tau.unsqueeze(-1)  # [B, L, 1]
            # div_term: [1, d_model] -> broadcasting a [B, L, d_model]
            # arg = τ / (10000^{2k/d_model})
            arg = tau_expanded * self.div_term  # [B, L, d_model]

            # Construimos el encoding alternando sin/cos
            enc = torch.empty(B, L, self.d_model, device=timestamps.device, dtype=torch.float32)

            # Posiciones pares -> sin
            enc[..., 0::2] = torch.sin(arg[..., 0::2])
            # Posiciones impares -> cos
            enc[..., 1::2] = torch.cos(arg[..., 1::2])

            return enc

        elif self.mode == "mlp":
            # tau: [B, L] -> [B, L, 1]
            tau_input = tau.unsqueeze(-1)
            return self.mlp(tau_input)

        elif self.mode == "time2vec":
            return self.time2vec(tau)

        else:
            raise ValueError(f"Modo de encoding temporal desconocido: {self.mode}")