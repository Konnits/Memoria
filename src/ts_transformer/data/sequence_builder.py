from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Mapping, Union, Optional, Sequence, Tuple

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
        - "target_timestamp": [M]
        - "target_values": [M, output_dim]

    Y devuelve un diccionario con:
        - "input_values": [L+K, input_dim]
            (últimas K filas son tokens target, con features placeholder)
        - "input_timestamps": [L+K]
            (últimos K elementos son el timestamp objetivo)
        - "is_target_mask": [L+K] (bool)
            (True en las últimas K posiciones)
        - "target_values": [output_dim] (sin cambios)
        - "target_timestamp": [M], relativo al primer evento cuando
          ``relative_timestamps=True``.
        - "absolute_target_timestamps": [M] float64 para auditoría.
        - "input_observation_counts": masa/densidad representada por token.

    El embedding del modelo se encargará de:
        value_embedding + time_encoding + flag_embedding

    Los timestamps absolutos se mantienen en float64 hasta restar el origen de
    la ventana. Sólo los deltas relativos entregados al modelo se convierten a
    float32, evitando el colapso de gaps pequeños cerca de timestamps UNIX.
    """

    input_dim: int
    target_token_value: Literal["zeros", "last"] = "zeros"
    use_sensor_ids: bool = False
    num_sensors: int = 0
    num_target_tokens: int = 1
    target_sensor_ids: Optional[Sequence[int]] = None
    relative_timestamps: bool = True

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

    def _prepare_timestamps(
        self,
        sample: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Relativiza en float64 *antes* de convertir la entrada del modelo a fp32.

        El origen numérico es el primer evento histórico, de modo que todos los
        tiempos válidos siguen siendo no negativos (importante para log1p). El
        ``forecast_origin`` separado continúa siendo el último evento histórico.
        """
        past_absolute = torch.as_tensor(
            sample["past_timestamps"], dtype=torch.float64
        )
        target_absolute = torch.as_tensor(
            sample["target_timestamp"], dtype=torch.float64
        )
        if past_absolute.ndim != 1 or past_absolute.numel() == 0:
            raise ValueError("past_timestamps debe ser 1D y contener al menos un valor.")
        if target_absolute.ndim != 1 or target_absolute.numel() == 0:
            raise ValueError("target_timestamp debe ser 1D y contener al menos un valor.")
        if not bool(torch.isfinite(past_absolute).all()) or not bool(
            torch.isfinite(target_absolute).all()
        ):
            raise ValueError("Los timestamps deben ser finitos.")

        numerical_origin = past_absolute[0]
        forecast_origin = torch.as_tensor(
            sample.get("forecast_origin", past_absolute[-1]), dtype=torch.float64
        ).reshape(())
        if self.relative_timestamps:
            past_model = (past_absolute - numerical_origin).to(torch.float32)
            target_model = (target_absolute - numerical_origin).to(torch.float32)
        else:
            # Compatibilidad explícita para checkpoints antiguos. Puede perder
            # gaps pequeños si los timestamps absolutos son grandes.
            past_model = past_absolute.to(torch.float32)
            target_model = target_absolute.to(torch.float32)
        return (
            past_model,
            target_model,
            past_absolute,
            target_absolute,
            torch.stack([numerical_origin, forecast_origin]),
        )

    @staticmethod
    def _copy_temporal_metadata(
        out: Dict[str, torch.Tensor],
        sample: Mapping[str, Any],
        absolute_target_timestamps: torch.Tensor,
        origins: torch.Tensor,
    ) -> None:
        out["absolute_target_timestamps"] = absolute_target_timestamps
        out["time_origin"] = origins[0]
        out["forecast_origin"] = origins[1]
        if "last_observation_age" in sample:
            out["last_observation_age"] = torch.as_tensor(
                sample["last_observation_age"], dtype=torch.float64
            ).reshape(())
        for key in (
            "past_original_observation_count",
            "past_original_max_gap",
            "past_original_median_gap",
        ):
            if key in sample:
                out[key] = torch.as_tensor(
                    sample[key], dtype=torch.float64
                ).reshape(())
        for key in (
            "past_sensor_observation_count",
            "past_sensor_max_gap",
            "past_sensor_median_gap",
            "sensor_last_observation_age",
        ):
            if key in sample:
                value = torch.as_tensor(sample[key], dtype=torch.float64)
                if value.ndim != 1:
                    raise ValueError(f"{key} debe tener shape [num_sensors].")
                out[key] = value

        # Los horizontes son features relativos: ya se calcularon en fp64 y se
        # pueden transportar en fp32. Los timestamps de auditoría siguen en fp64.
        for key in ("target_horizons", "requested_target_horizons"):
            if key in sample:
                out[key] = torch.as_tensor(sample[key], dtype=torch.float64).to(
                    torch.float32
                )
        for key in ("requested_target_timestamps", "target_source_timestamps"):
            if key in sample:
                out[key] = torch.as_tensor(sample[key], dtype=torch.float64)

    @staticmethod
    def _build_observation_counts(
        sample: Mapping[str, Any],
        history_length: int,
        target_tokens: int,
    ) -> torch.Tensor:
        counts = torch.as_tensor(
            sample.get(
                "past_observation_counts",
                torch.ones(history_length, dtype=torch.float32),
            ),
            dtype=torch.float32,
        )
        if counts.shape != (history_length,):
            raise ValueError(
                "past_observation_counts debe tener un valor por token histórico."
            )
        if not bool(torch.isfinite(counts).all()) or bool((counts <= 0.0).any()):
            raise ValueError("past_observation_counts debe ser finito y > 0.")
        return torch.cat(
            [counts, torch.zeros(target_tokens, dtype=torch.float32)], dim=0
        )

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        past_values = torch.as_tensor(sample["past_values"], dtype=torch.float32)
        (
            past_timestamps,
            target_timestamp,
            _past_absolute,
            target_absolute,
            origins,
        ) = self._prepare_timestamps(sample)
        # Ahora target_timestamp y target_values son arrays [M], [M, output_dim]
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
            "input_observation_counts": self._build_observation_counts(
                sample, L, K_total
            ),
        }
        self._copy_temporal_metadata(out, sample, target_absolute, origins)

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
        (
            past_timestamps,
            target_timestamp,
            _past_absolute,
            target_absolute,
            origins,
        ) = self._prepare_timestamps(sample)
        target_values = torch.as_tensor(sample["target_values"], dtype=torch.float32)
        target_loss_mask = sample.get("target_loss_mask", None)
        if target_loss_mask is not None:
            target_loss_mask = torch.as_tensor(target_loss_mask, dtype=torch.float32)

        L, D = past_values.shape
        M = target_timestamp.shape[0]

        K_per_m = int(self.num_target_tokens)
        K_total = M * K_per_m

        # Teacher forcing por horizonte: el bloque de sensores del horizonte
        # i recibe los valores del bloque i - 1; el primer bloque queda en cero.
        shifted_targets = torch.zeros(K_total, D, dtype=past_values.dtype)
        flattened_targets = target_values.reshape(-1)
        if flattened_targets.numel() != K_total:
            raise ValueError(
                "target_values debe contener un valor por token target en modo autoregresivo."
            )
        if M > 1:
            shifted_targets[K_per_m:, 0] = flattened_targets[:-K_per_m]

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
            "input_observation_counts": self._build_observation_counts(
                sample, L, K_total
            ),
        }
        self._copy_temporal_metadata(out, sample, target_absolute, origins)

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
