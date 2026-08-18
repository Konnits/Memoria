from __future__ import annotations

import torch
from torch import nn


class SensorEmbedding(nn.Module):
    """
    Embedding de sensor para tokens tipo evento (sensor_id, t, value).

    Parámetros
    ----------
    num_sensors:
        Número de sensores reales. Internamente se agrega un embedding extra
        para el token target (id = num_sensors).
    d_model:
        Dimensión de embedding.
    include_target_token:
        Si es ``True`` reserva la fila ``id = num_sensors`` usada por los
        modelos legacy. Las arquitecturas que canonicalizan el canal target
        antes del embedding pueden desactivarla para no optimizar una fila
        inalcanzable.
    """

    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        *,
        include_target_token: bool = True,
    ) -> None:
        super().__init__()
        if num_sensors <= 0:
            raise ValueError("num_sensors debe ser > 0.")

        self.num_sensors = int(num_sensors)
        self.include_target_token = bool(include_target_token)
        self.target_sensor_id = (
            int(num_sensors) if self.include_target_token else None
        )
        self.max_sensor_id = (
            int(num_sensors) if self.include_target_token else int(num_sensors) - 1
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.num_sensors + int(self.include_target_token),
            embedding_dim=d_model,
        )

    def forward(self, sensor_ids: torch.Tensor) -> torch.Tensor:
        if sensor_ids.ndim != 2:
            raise ValueError(
                f"SensorEmbedding espera tensor 2D [B, L], recibió {tuple(sensor_ids.shape)}"
            )

        if torch.any(sensor_ids < 0) or torch.any(sensor_ids > self.max_sensor_id):
            raise ValueError(
                "sensor_ids contiene ids fuera de rango permitido "
                f"[0, {self.max_sensor_id}]."
            )

        return self.embedding(sensor_ids)
