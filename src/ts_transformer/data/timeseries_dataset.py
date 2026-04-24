from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequence_builder import SequenceBuilder


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class TimeSeriesDatasetConfig:
    """
    Configuración del TimeSeriesDataset.

    Parámetros principales:
    - history_length: número de pasos de historia que alimentan al modelo.
    - target_offset: cuántos pasos después del último punto de la historia queremos predecir.
        Ejemplos (índices en el array original):
        * history_length=4, target_offset=0:
            historia = [0,1,2,3], target = 3
        * history_length=4, target_offset=1:
            historia = [0,1,2,3], target = 4
    - stride: salto de un ejemplo al siguiente (en índices del array original).
    - min_history_length: si se define, se samplea una historia de largo variable
      entre [min_history_length, history_length] en cada __getitem__.
    - target_offset_choices: si se define, los offsets se samplean de esta lista
      (por ejemplo [1, 2, 3] para distintos horizontes de predicción).
        - target_offset_min / target_offset_max: alternativa compacta para samplear
            offsets enteros en el rango [min, max] (inclusive).
            Si target_offset_choices está definido, tiene prioridad.
    """

    history_length: int
    target_offset: int = 1
    stride: int = 1
    min_history_length: Optional[int] = None
    target_offset_choices: Optional[Sequence[int]] = None
    target_offset_min: Optional[int] = None
    target_offset_max: Optional[int] = None
    num_targets: int = 1  # Añadido para multi-objetivo


def _resolve_offsets(cfg: TimeSeriesDatasetConfig) -> List[int]:
    """
    Resuelve el conjunto de offsets válidos para target.

    Prioridad:
    1) target_offset_choices
    2) target_offset_min/target_offset_max
    3) target_offset fijo
    """
    if cfg.target_offset_choices is not None:
        offsets = [int(o) for o in cfg.target_offset_choices]
    elif cfg.target_offset_max is not None:
        o_min = int(cfg.target_offset_min if cfg.target_offset_min is not None else cfg.target_offset)
        o_max = int(cfg.target_offset_max)
        if o_min > o_max:
            raise ValueError("target_offset_min no puede ser mayor que target_offset_max.")
        offsets = list(range(o_min, o_max + 1))
    else:
        offsets = [int(cfg.target_offset)]

    if len(offsets) == 0:
        raise ValueError("No hay offsets válidos para target.")
    if any(o < 0 for o in offsets):
        raise ValueError("Todos los target offsets deben ser >= 0.")

    return offsets


class TimeSeriesDataset(Dataset):
    """
    Dataset básico para series de tiempo univariadas o multivariadas.

    Supone una serie temporal ya ordenada por tiempo, con:
    - values: [T, D_total] (D_total = dimensión total disponible)
    - timestamps: [T] (numérico; p.ej. segundos, o datetime convertido a float)

    Permite separar explícitamente:
    - input_dim: cuántas dimensiones se usan como entrada (features).
    - output_dim: cuántas dimensiones se usan como target (salida).
      Si `targets` es None, se asume que:
          * las primeras `input_dim` columnas son entrada,
          * las siguientes `output_dim` columnas son salida.

    Cada elemento del dataset es un dict con:
    - "past_values": [history_length, input_dim]
    - "past_timestamps": [history_length]
    - "target_timestamp": escalar (float)
    - "target_values": [output_dim]
    """

    def __init__(
        self,
        values: ArrayLike,
        timestamps: ArrayLike,
        config: TimeSeriesDatasetConfig,
        input_dim: int,
        output_dim: int,
        targets: Optional[ArrayLike] = None,
        sequence_builder: Optional[SequenceBuilder] = None,
    ) -> None:
        """
        Parameters
        ----------
        values:
            Matriz de valores de la serie, shape [T, D_total].
            Puede ser np.ndarray o torch.Tensor.
        timestamps:
            Vector de timestamps, shape [T].
        config:
            Objeto TimeSeriesDatasetConfig con history_length, target_offset, stride.
        input_dim:
            Número de dimensiones de entrada.
        output_dim:
            Número de dimensiones del target.
        targets:
            Matriz opcional de targets explícitos, shape [T, output_dim].
            Si es None, se usa un slice de `values`.
        sequence_builder:
            Instancia opcional de SequenceBuilder para transformar el sample.
        """
        super().__init__()

        # Optimización 1.2: Guardar como tensores Torch contiguos en CPU
        self.values = self._to_torch_2d(values).contiguous()
        self.timestamps = self._to_torch_1d(timestamps).contiguous()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_builder = sequence_builder

        if self.values.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                f"values y timestamps deben tener la misma longitud. "
                f"{self.values.shape[0]} != {self.timestamps.shape[0]}"
            )

        if targets is not None:
            self.targets = self._to_torch_2d(targets).contiguous()
            if self.targets.shape[0] != self.values.shape[0]:
                raise ValueError(
                    "targets y values deben tener la misma longitud temporal."
                )
        else:
            # Tomamos los targets como un subset de `values`
            total_dim = self.values.shape[1]
            if self.input_dim + self.output_dim > total_dim:
                raise ValueError(
                    f"input_dim + output_dim ({self.input_dim} + {self.output_dim}) "
                    f"supera la dimensión total de values ({total_dim})."
                )
            self.targets = self.values[:, self.input_dim : self.input_dim + self.output_dim].contiguous()

        # Optimización 3 y 4: Precomputar todo lo posible
        self.offsets = _resolve_offsets(self.config)
        self.max_offset = max(self.offsets)
        self.offsets_t = torch.tensor(self.offsets, dtype=torch.long)
        
        self.history_length = int(self.config.history_length)
        self.min_history_length = int(self.config.min_history_length) if self.config.min_history_length is not None else self.history_length
        self.fixed_history_length = (self.min_history_length == self.history_length)
        
        self.num_available_offsets = len(self.offsets)
        self.k_targets = min(self.config.num_targets, self.num_available_offsets)
        self.single_target_offset = (self.k_targets == 1 and self.num_available_offsets == 1)

        # Precompute índices de los ejemplos
        self._example_indices = self._build_example_indices()

    @staticmethod
    def _to_torch_2d(x: ArrayLike) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim != 2:
            raise ValueError(f"Se esperaba un array 2D, pero se obtuvo shape {x.shape}.")
        return x.to(torch.float32)

    @staticmethod
    def _to_torch_1d(x: ArrayLike) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim != 1:
            raise ValueError(f"Se esperaba un array 1D, pero se obtuvo shape {x.shape}.")
        return x.to(torch.float32)

    def _build_example_indices(self) -> List[int]:
        """
        Construye la lista de índices 'anchor' para cada ejemplo.
        """
        T = self.values.shape[0]
        h_max = self.history_length
        stride = self.config.stride

        if h_max <= 0:
            raise ValueError("history_length debe ser > 0.")

        # Último índice permitido para la historia + offset máximo
        max_anchor = T - 1 - self.max_offset
        if max_anchor < h_max:
            raise ValueError(
                "No hay suficientes datos para construir al menos un ejemplo con "
                f"history_length={h_max} y max_target_offset={self.max_offset} en una serie de longitud T={T}."
            )

        indices = list(range(h_max, max_anchor + 1, stride))
        return indices

    def __len__(self) -> int:
        return len(self._example_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor = self._example_indices[idx]
        T = self.values.shape[0]

        # -----------------------------
        # 1) Elegir longitud de historia
        # -----------------------------
        if self.fixed_history_length:
            h = self.history_length
        else:
            h = int(torch.randint(self.min_history_length, self.history_length + 1, (1,)).item())

        if h > anchor:
            h = anchor

        history_start = anchor - h
        history_end = anchor

        # -----------------------------
        # 2) Elegir target_offset(s)
        # -----------------------------
        if self.single_target_offset:
            chosen_offsets = self.offsets
        else:
            if self.k_targets == 1:
                offset_idx = torch.randint(0, self.num_available_offsets, (1,))
                chosen_offsets = [self.offsets_t[offset_idx].item()]
            else:
                perm = torch.randperm(self.num_available_offsets)[:self.k_targets]
                chosen_offsets = self.offsets_t[perm].tolist()
                chosen_offsets.sort()

        target_indices = [anchor + off for off in chosen_offsets]
        if len(target_indices) != self.k_targets:
            raise RuntimeError(
                f"Se esperaban {self.k_targets} targets válidos, pero se obtuvieron {len(target_indices)}."
            )

        # -----------------------------
        # 3) Extraer valores y timestamps (Slicing directo en tensores torch)
        # -----------------------------
        past_values = self.values[history_start:history_end, :self.input_dim]
        past_timestamps = self.timestamps[history_start:history_end]

        target_timestamps = self.timestamps[target_indices]
        target_values = self.targets[target_indices]

        sample = {
            "past_values": past_values,
            "past_timestamps": past_timestamps,
            "target_timestamp": target_timestamps,
            "target_values": target_values,
            "target_loss_mask": torch.ones((len(target_indices), self.output_dim), dtype=torch.float32),
        }

        # Optimización 1.1: Aplicar sequence_builder aquí si existe
        if self.sequence_builder is not None:
            return self.sequence_builder(sample)
        
        return sample


class EventTimeSeriesDataset(Dataset):
    """
    Dataset en formato evento para sensores asíncronos.

    En lugar de usar una matriz densa [L, D], convierte cada medición observada
    dentro de la historia en tokens independientes (sensor_id, t, value).
    """

    def __init__(

        self,
        values: ArrayLike,
        timestamps: ArrayLike,
        targets: ArrayLike,
        config: TimeSeriesDatasetConfig,
        input_dim: int,
        output_dim: int,
        sequence_builder: Optional[SequenceBuilder] = None,
    ) -> None:
        super().__init__()

        # Optimización 1.2: Guardar como tensores Torch contiguos en CPU
        self.values = TimeSeriesDataset._to_torch_2d(values).contiguous()
        self.timestamps = TimeSeriesDataset._to_torch_1d(timestamps).contiguous()
        self.targets = TimeSeriesDataset._to_torch_2d(targets).contiguous()
        self.config = config
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sequence_builder = sequence_builder

        if self.values.shape[0] != self.timestamps.shape[0] or self.targets.shape[0] != self.timestamps.shape[0]:
            raise ValueError("values, timestamps y targets deben tener la misma longitud temporal.")

        if self.values.shape[1] != self.input_dim:
            raise ValueError("input_dim no coincide con values.shape[1].")
        if self.targets.shape[1] != self.output_dim:
            raise ValueError("output_dim no coincide con targets.shape[1].")

        # Optimización 3 y 4: Precomputar todo lo posible
        self.offsets = _resolve_offsets(self.config)
        self.max_offset = max(self.offsets)
        self.offsets_t = torch.tensor(self.offsets, dtype=torch.long)
        
        self.history_length = int(self.config.history_length)
        self.min_history_length = int(self.config.min_history_length) if self.config.min_history_length is not None else self.history_length
        self.fixed_history_length = (self.min_history_length == self.history_length)
        
        self.num_available_offsets = len(self.offsets)
        self.k_targets = min(self.config.num_targets, self.num_available_offsets)
        self.single_target_offset = (self.k_targets == 1 and self.num_available_offsets == 1)

        self._example_indices = self._build_example_indices()

    def _build_example_indices(self) -> List[int]:
        T = self.values.shape[0]
        h_max = self.history_length
        stride = self.config.stride

        if h_max <= 0:
            raise ValueError("history_length debe ser > 0.")

        max_anchor = T - 1 - self.max_offset
        if max_anchor < h_max:
            raise ValueError(
                "No hay suficientes datos para construir ejemplos con "
                f"history_length={h_max} y max_target_offset={self.max_offset}."
            )

        return list(range(h_max, max_anchor + 1, stride))

    def get_approx_lengths(self) -> List[int]:
        """
        Devuelve una lista con la cantidad aproximada de tokens (eventos) 
        esperada para cada ejemplo. Útil para el BucketBatchSampler.
        """
        # Contar cuántos valores válidos (no NaN) hay por timestamp
        valid_counts = (~torch.isnan(self.values)).sum(dim=1)
        cumsum_valid = valid_counts.cumsum(dim=0)
        
        lengths = []
        for anchor in self._example_indices:
            h = self.history_length
            start = anchor - h
            end = anchor
            
            count = cumsum_valid[end - 1] - (cumsum_valid[start - 1] if start > 0 else 0)
            lengths.append(int(count.item()) + self.k_targets)
            
        return lengths

    def __len__(self) -> int:
        return len(self._example_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor = self._example_indices[idx]
        T = self.values.shape[0]

        if self.fixed_history_length:
            h = self.history_length
        else:
            h = int(torch.randint(self.min_history_length, self.history_length + 1, (1,)).item())

        if h > anchor:
            h = anchor

        history_start = anchor - h
        history_end = anchor

        if self.single_target_offset:
            chosen_offsets = self.offsets
        else:
            if self.k_targets == 1:
                offset_idx = torch.randint(0, self.num_available_offsets, (1,))
                chosen_offsets = [self.offsets_t[offset_idx].item()]
            else:
                perm = torch.randperm(self.num_available_offsets)[:self.k_targets]
                chosen_offsets = self.offsets_t[perm].tolist()
                chosen_offsets.sort()

        target_indices = [anchor + off for off in chosen_offsets]
        if len(target_indices) != self.k_targets:
            raise RuntimeError(
                f"Se esperaban {self.k_targets} targets válidos, pero se obtuvieron {len(target_indices)}."
            )

        # Optimización 1.3: Vectorizar la extracción de eventos
        hist_values = self.values[history_start:history_end]
        hist_timestamps = self.timestamps[history_start:history_end]

        # Encontrar índices (row_idx, col_idx) donde no hay NaNs
        valid_mask = ~torch.isnan(hist_values)
        rows, cols = torch.where(valid_mask)

        if rows.numel() > 0:
            event_values = hist_values[rows, cols].view(-1, 1)
            event_timestamps = hist_timestamps[rows]
            event_sensor_ids = cols
        else:
            # Fallback si no hay observaciones válidas
            fallback_t = hist_timestamps[-1] if hist_timestamps.numel() > 0 else self.timestamps[anchor - 1]
            event_values = torch.tensor([[0.0]], dtype=torch.float32)
            event_timestamps = fallback_t.unsqueeze(0)
            event_sensor_ids = torch.tensor([0], dtype=torch.long)

        # Manejo de targets
        target_raw = self.targets[target_indices]
        target_loss_mask = (~torch.isnan(target_raw)).to(torch.float32)
        target_values = torch.nan_to_num(target_raw, nan=0.0)

        sample = {
            "past_values": event_values,
            "past_timestamps": event_timestamps,
            "past_sensor_ids": event_sensor_ids,
            "target_timestamp": self.timestamps[target_indices],
            "target_values": target_values,
            "target_loss_mask": target_loss_mask,
        }

        # Optimización 1.1: Aplicar sequence_builder aquí si existe
        if self.sequence_builder is not None:
            return self.sequence_builder(sample)
        
        return sample