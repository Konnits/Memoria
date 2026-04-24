from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Mapping, Union, Optional, Sequence

import torch


TensorLike = Union[torch.Tensor]


@dataclass
class SequenceBuilder:
    """
    Construye la secuencia de entrada al Transformer añadiendo uno o más
    tokens target al final de la historia.

    Toma un sample del TimeSeriesDataset con claves:
        - "past_values": [L, input_dim]
        - "past_timestamps": [L]
        - "target_timestamp": escalar
        - "target_values": [output_dim]

    Y devuelve un diccionario con:
        - "input_values": [L+K, input_dim]
            (últimas K filas son tokens target, con features placeholder)
        - "input_timestamps": [L+K]
            (últimos K elementos son el timestamp objetivo)
        - "is_target_mask": [L+K] (bool)
            (True en las últimas K posiciones)
        - "target_values": [output_dim] (sin cambios)
        - "target_timestamp": escalar (sin cambios)

    El embedding del modelo se encargará de:
        value_embedding + time_encoding + flag_embedding
    """

    input_dim: int
    target_token_value: Literal["zeros", "last"] = "zeros"
    use_sensor_ids: bool = False
    num_sensors: int = 0
    num_target_tokens: int = 1
    target_sensor_ids: Optional[Sequence[int]] = None

    def __post_init__(self):
        if self.use_sensor_ids:
            if self.num_sensors <= 0:
                raise ValueError("En mode event, num_sensors debe ser > 0.")
            if self.num_target_tokens <= 0:
                raise ValueError("num_target_tokens debe ser > 0.")
        else:
            if self.num_target_tokens != 1:
                raise ValueError(
                    "En dense mode (use_sensor_ids=False), num_target_tokens debe ser 1."
                )
            if self.target_sensor_ids is not None:
                raise ValueError(
                    "En dense mode, target_sensor_ids debe ser None."
                )

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        past_values = torch.as_tensor(sample["past_values"], dtype=torch.float32)
        past_timestamps = torch.as_tensor(sample["past_timestamps"], dtype=torch.float32)
        # Ahora target_timestamp y target_values son arrays [M], [M, output_dim]
        target_timestamp = torch.as_tensor(sample["target_timestamp"], dtype=torch.float32)
        target_values = torch.as_tensor(sample["target_values"], dtype=torch.float32)
        target_loss_mask = sample.get("target_loss_mask", None)
        if target_loss_mask is not None:
            target_loss_mask = torch.as_tensor(target_loss_mask, dtype=torch.float32)

        if past_values.ndim != 2:
            raise ValueError(f"past_values debe ser 2D [L, D], se obtuvo {past_values.shape}.")
        if past_values.size(1) != self.input_dim:
            raise ValueError(
                f"Dimensión de entrada inconsistente: past_values.shape[1]={past_values.size(1)} "
                f"pero input_dim={self.input_dim}."
            )

        L, D = past_values.shape
        M = target_timestamp.shape[0]

        # Cada timestamp a futuro demandará num_target_tokens (1 en dense, o output_dim en events)
        K_per_m = int(self.num_target_tokens)
        K_total = M * K_per_m

        if self.target_token_value == "zeros":
            target_token_values = torch.zeros(K_total, D, dtype=past_values.dtype)
        elif self.target_token_value == "last":
            target_token_values = past_values[-1:, :].clone().repeat(K_total, 1)
        else:
            raise ValueError(f"target_token_value desconocido: {self.target_token_value}")

        input_values = torch.cat([past_values, target_token_values], dim=0)  # [L + K_total, D]
        
        # Repetir cada timestamp futuro K_per_m veces
        target_timestamps_expanded = target_timestamp.repeat_interleave(K_per_m)
        input_timestamps = torch.cat([past_timestamps, target_timestamps_expanded], dim=0)

        is_target_mask = torch.zeros(L + K_total, dtype=torch.bool)
        is_target_mask[-K_total:] = True

        out: Dict[str, torch.Tensor] = {
            "input_values": input_values,
            "input_timestamps": input_timestamps,
            "is_target_mask": is_target_mask,
            "target_values": target_values, # [M, output_dim]
            "target_timestamp": target_timestamp, # [M]
        }

        if self.use_sensor_ids:
            past_sensor_ids = torch.as_tensor(sample["past_sensor_ids"], dtype=torch.long)
            if self.target_sensor_ids is not None:
                # Repetir el array target_sensor_ids para los M timestamps
                tsid_tensor = torch.as_tensor(self.target_sensor_ids, dtype=torch.long)
                target_sensor_ids = tsid_tensor.repeat(M)
            else:
                target_sensor_ids = torch.full((K_total,), self.num_sensors, dtype=torch.long)

            input_sensor_ids = torch.cat([past_sensor_ids, target_sensor_ids], dim=0)
            out["input_sensor_ids"] = input_sensor_ids

        if target_loss_mask is not None:
            out["target_loss_mask"] = target_loss_mask

        return out


@dataclass
class AutoregressiveSequenceBuilder(SequenceBuilder):
    """
    Construye secuencias usando Teacher Forcing para entrenamiento autoregresivo.
    En lugar de poblar targets con 'zeros', inserta los valores reales desplazados
    una posición a la derecha. El primer token a predecir recibe 'zeros'.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        past_values = torch.as_tensor(sample["past_values"], dtype=torch.float32)
        past_timestamps = torch.as_tensor(sample["past_timestamps"], dtype=torch.float32)
        target_timestamp = torch.as_tensor(sample["target_timestamp"], dtype=torch.float32)
        target_values = torch.as_tensor(sample["target_values"], dtype=torch.float32)
        target_loss_mask = sample.get("target_loss_mask", None)
        if target_loss_mask is not None:
            target_loss_mask = torch.as_tensor(target_loss_mask, dtype=torch.float32)

        L, D = past_values.shape
        M = target_timestamp.shape[0]

        K_per_m = int(self.num_target_tokens)
        K_total = M * K_per_m

        if K_per_m != 1:
            raise NotImplementedError("Autoregressive mode solo soporta Dense (num_target_tokens=1) por ahora.")

        # Teacher Forcing: desplazamos 1 paso.
        # Para el token en M_i, introducimos el valor de M_{i-1}.
        # Para M_0, introducimos ceros.
        shifted_targets = torch.zeros(M, D, dtype=past_values.dtype)
        out_d = min(D, target_values.size(1))
        
        if M > 1:
            shifted_targets[1:, :out_d] = target_values[:-1, :out_d]

        target_token_values = shifted_targets

        input_values = torch.cat([past_values, target_token_values], dim=0)  # [L + M, D]
        
        target_timestamps_expanded = target_timestamp.repeat_interleave(K_per_m)
        input_timestamps = torch.cat([past_timestamps, target_timestamps_expanded], dim=0)

        is_target_mask = torch.zeros(L + K_total, dtype=torch.bool)
        is_target_mask[-K_total:] = True

        out: Dict[str, torch.Tensor] = {
            "input_values": input_values,
            "input_timestamps": input_timestamps,
            "is_target_mask": is_target_mask,
            "target_values": target_values,
            "target_timestamp": target_timestamp,
        }

        if self.use_sensor_ids:
            past_sensor_ids = torch.as_tensor(sample["past_sensor_ids"], dtype=torch.long)
            if self.target_sensor_ids is not None:
                tsid_tensor = torch.as_tensor(self.target_sensor_ids, dtype=torch.long)
                target_sensor_ids = tsid_tensor.repeat(M)
            else:
                target_sensor_ids = torch.full((K_total,), self.num_sensors, dtype=torch.long)

            input_sensor_ids = torch.cat([past_sensor_ids, target_sensor_ids], dim=0)
            out["input_sensor_ids"] = input_sensor_ids

        if target_loss_mask is not None:
            out["target_loss_mask"] = target_loss_mask

        return out