from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Union, List, Literal, Tuple

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
            historia = [0,1,2,3], target = 4
        * history_length=4, target_offset=1:
            historia = [0,1,2,3], target = 5
    - stride: salto de un ejemplo al siguiente (en índices del array original).
    - min_history_length: si se define, se samplea una historia de largo variable
      entre [min_history_length, history_length] en cada __getitem__.
    - target_offset_choices: si se define, los offsets se samplean de esta lista
      (por ejemplo [1, 2, 3] para distintos horizontes de predicción).
        - target_offset_min / target_offset_max: alternativa compacta para samplear
            offsets enteros en el rango [min, max] (inclusive).
            Si target_offset_choices está definido, tiene prioridad.
    - target_horizon_choices: horizontes expresados en las mismas unidades que
      ``timestamps``. Si se define (o se define el rango min/max), el dataset
      cambia explícitamente de offsets por evento a consultas por tiempo físico.
    - target_match_mode:
        * ``"next"`` (default): desplaza la consulta a la primera observación
          real en o después del horizonte solicitado. Es la opción honesta para
          datos observacionales, pues no inventa una verdad entre mediciones.
        * ``"linear"``: mantiene exactamente el tiempo solicitado e interpola
          cada target entre sus observaciones válidas. Debe usarse sólo cuando
          sea defendible asumir una señal continua (p.ej. datos sintéticos).
    - randomize_query_order: permuta conjuntamente timestamps, valores y máscaras
      para impedir que la posición ordinal revele el horizonte.
    - sampling_seed: hace reproducible el muestreo por índice. ``set_epoch``
      permite obtener otra muestra reproducible en cada época.
    - history_duration: si se define, la historia se selecciona por duración
      física en lugar de por cantidad de filas. ``max_history_events`` limita
      el costo; si se omite se usa ``history_length`` como límite.
    - history_subsampling: selecciona filas representativas en todo el intervalo.
      ``uniform_time`` preserva cobertura/gaps, ``uniform_index`` densidad por
      rango y ``random`` muestrea reproduciblemente. Siempre conserva el último
      evento (y también el primero cuando el límite es >= 2).
    - cache_deterministic_history: cache lazy y opt-in de filas/conteos de
      historia para ``uniform_time``/``uniform_index``. No cachea targets ni
      queries. Es especialmente útil con ``num_workers=0``; con workers cada
      proceso mantiene su propia copia y el beneficio no persiste entre épocas.
    """

    history_length: int
    target_offset: int = 1
    stride: int = 1
    min_history_length: Optional[int] = None
    target_offset_choices: Optional[Sequence[int]] = None
    target_offset_min: Optional[int] = None
    target_offset_max: Optional[int] = None
    num_targets: int = 1  # Añadido para multi-objetivo
    target_horizon_choices: Optional[Sequence[float]] = None
    target_horizon_min: Optional[float] = None
    target_horizon_max: Optional[float] = None
    target_horizon_sampling: Literal["uniform", "log_uniform"] = "uniform"
    target_match_mode: Literal["next", "linear"] = "next"
    randomize_query_order: bool = False
    sampling_seed: int = 0
    history_duration: Optional[float] = None
    max_history_events: Optional[int] = None
    history_subsampling: Literal["uniform_time", "uniform_index", "random"] = (
        "uniform_time"
    )
    compute_history_diagnostics: bool = True
    cache_deterministic_history: bool = False


def _uses_physical_horizons(cfg: TimeSeriesDatasetConfig) -> bool:
    return (
        cfg.target_horizon_choices is not None
        or cfg.target_horizon_min is not None
        or cfg.target_horizon_max is not None
    )


def _validate_config(cfg: TimeSeriesDatasetConfig) -> None:
    if int(cfg.num_targets) <= 0:
        raise ValueError("num_targets debe ser > 0.")
    if int(cfg.stride) <= 0:
        raise ValueError("stride debe ser > 0.")
    if cfg.target_horizon_sampling not in {"uniform", "log_uniform"}:
        raise ValueError("target_horizon_sampling debe ser 'uniform' o 'log_uniform'.")
    if cfg.target_match_mode not in {"next", "linear"}:
        raise ValueError("target_match_mode debe ser 'next' o 'linear'.")
    if cfg.history_duration is not None:
        duration = float(cfg.history_duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("history_duration debe ser finito y > 0.")
    if cfg.max_history_events is not None and int(cfg.max_history_events) <= 0:
        raise ValueError("max_history_events debe ser > 0.")
    if cfg.history_subsampling not in {"uniform_time", "uniform_index", "random"}:
        raise ValueError(
            "history_subsampling debe ser 'uniform_time', 'uniform_index' o 'random'."
        )
    if not isinstance(cfg.compute_history_diagnostics, (bool, np.bool_)):
        raise ValueError("compute_history_diagnostics debe ser booleano.")
    if not isinstance(cfg.cache_deterministic_history, (bool, np.bool_)):
        raise ValueError("cache_deterministic_history debe ser booleano.")


def _resolve_horizon_bounds(cfg: TimeSeriesDatasetConfig) -> Tuple[List[float], float, float]:
    """Valida y devuelve choices, mínimo y máximo de horizontes físicos."""
    if cfg.target_horizon_choices is not None:
        choices = [float(h) for h in cfg.target_horizon_choices]
        if not choices:
            raise ValueError("target_horizon_choices no puede estar vacío.")
        if not np.isfinite(choices).all() or any(h < 0.0 for h in choices):
            raise ValueError("Los horizontes físicos deben ser finitos y >= 0.")
        return choices, min(choices), max(choices)

    if cfg.target_horizon_min is None or cfg.target_horizon_max is None:
        raise ValueError(
            "Para samplear horizontes físicos se requieren target_horizon_min y "
            "target_horizon_max."
        )
    h_min = float(cfg.target_horizon_min)
    h_max = float(cfg.target_horizon_max)
    if not np.isfinite([h_min, h_max]).all() or h_min < 0.0 or h_max < h_min:
        raise ValueError(
            "Los límites físicos deben ser finitos y cumplir 0 <= min <= max."
        )
    if cfg.target_horizon_sampling == "log_uniform" and h_min <= 0.0:
        raise ValueError("log_uniform requiere target_horizon_min > 0.")
    return [], h_min, h_max


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


class _TemporalTargetMixin:
    """Implementación compartida de offsets y consultas en tiempo físico."""

    _FORECAST_ORIGIN_DISCARD_CAUSES = (
        "insufficient_history_coverage",
        "origin_after_last_observation",
        "empty_history",
        "target_before_available_range",
        "target_after_available_range",
    )

    config: TimeSeriesDatasetConfig
    timestamps: torch.Tensor
    target_timestamps: torch.Tensor
    targets: torch.Tensor
    output_dim: int
    forecast_origin_timestamps: Optional[torch.Tensor]

    def _init_temporal_targeting(self) -> None:
        _validate_config(self.config)
        if self.timestamps.dtype != torch.float64:
            raise TypeError("timestamps debe conservarse internamente como torch.float64.")
        if self.timestamps.numel() == 0:
            raise ValueError("timestamps no puede estar vacío.")
        if not bool(torch.isfinite(self.timestamps).all()):
            raise ValueError("timestamps debe contener sólo valores finitos.")
        if bool((self.timestamps[1:] < self.timestamps[:-1]).any()):
            raise ValueError("timestamps debe estar ordenado de forma no decreciente.")
        if self.target_timestamps.dtype != torch.float64:
            raise TypeError("target_timestamps debe conservarse como torch.float64.")
        if self.target_timestamps.numel() != self.targets.shape[0]:
            raise ValueError(
                "target_timestamps y targets deben tener la misma longitud."
            )
        if not bool(torch.isfinite(self.target_timestamps).all()):
            raise ValueError("target_timestamps debe contener sólo valores finitos.")
        if bool((self.target_timestamps[1:] < self.target_timestamps[:-1]).any()):
            raise ValueError("target_timestamps debe estar ordenado de forma no decreciente.")

        self.uses_physical_horizons = _uses_physical_horizons(self.config)
        if self.forecast_origin_timestamps is not None:
            origins = self.forecast_origin_timestamps
            if not self.uses_physical_horizons:
                raise ValueError(
                    "forecast_origin_timestamps sólo se admite con horizontes físicos."
                )
            if self.config.history_duration is None:
                raise ValueError(
                    "forecast_origin_timestamps requiere history_duration para "
                    "definir una historia física completa."
                )
            if origins.dtype != torch.float64 or origins.ndim != 1:
                raise ValueError(
                    "forecast_origin_timestamps debe ser un vector float64."
                )
            if origins.numel() == 0 or not bool(torch.isfinite(origins).all()):
                raise ValueError(
                    "forecast_origin_timestamps debe contener valores finitos."
                )
            if bool((origins[1:] < origins[:-1]).any()):
                raise ValueError(
                    "forecast_origin_timestamps debe estar ordenado de forma no decreciente."
                )
        self._epoch = 0
        self._example_forecast_origins: Optional[torch.Tensor] = None
        self.num_discarded_forecast_origins = 0
        self.forecast_origin_candidate_count = 0
        self.forecast_origin_accepted_count = 0
        self.forecast_origin_discard_counts = {
            cause: 0 for cause in self._FORECAST_ORIGIN_DISCARD_CAUSES
        }
        self.forecast_origin_audit: Dict[str, Any] = {}
        self._update_forecast_origin_audit()
        self.history_length = int(self.config.history_length)
        self.min_history_length = (
            int(self.config.min_history_length)
            if self.config.min_history_length is not None
            else self.history_length
        )
        if self.history_length <= 0:
            raise ValueError("history_length debe ser > 0.")
        if not 0 < self.min_history_length <= self.history_length:
            raise ValueError(
                "min_history_length debe cumplir 0 < min_history_length <= history_length."
            )
        self.fixed_history_length = self.min_history_length == self.history_length
        self.history_duration = (
            float(self.config.history_duration)
            if self.config.history_duration is not None
            else None
        )
        self.max_history_events = int(
            self.config.max_history_events
            if self.config.max_history_events is not None
            else self.history_length
        )
        # El benchmark físico vuelve a recorrer exactamente los mismos anchors
        # durante épocas, arquitecturas y seeds. Guardar sólo la selección de
        # historia evita repetir búsquedas/diagnósticos costosos sin congelar
        # horizontes ni el orden de queries, que se construyen después usando
        # el stream dependiente de época. Es opt-in para acotar memoria y nunca
        # se habilita con subsampling aleatorio.
        self._history_cache_enabled = bool(
            self.config.cache_deterministic_history
            and self.config.history_subsampling in {"uniform_time", "uniform_index"}
            and (self.history_duration is not None or self.fixed_history_length)
        )
        self._dense_history_cache: Dict[
            int,
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        ] = {}

        if self.uses_physical_horizons:
            self.horizon_choices, self.min_horizon, self.max_horizon = (
                _resolve_horizon_bounds(self.config)
            )
            self.offsets = []
            self.max_offset = 0
            self.num_available_offsets = 0
            self.k_targets = (
                min(int(self.config.num_targets), len(self.horizon_choices))
                if self.horizon_choices
                else int(self.config.num_targets)
            )
            self.single_target_offset = False
        else:
            self.offsets = _resolve_offsets(self.config)
            self.max_offset = max(self.offsets)
            self.num_available_offsets = len(self.offsets)
            self.k_targets = min(int(self.config.num_targets), self.num_available_offsets)
            self.single_target_offset = (
                self.k_targets == 1 and self.num_available_offsets == 1
            )
            if self.target_timestamps.shape != self.timestamps.shape or not torch.equal(
                self.target_timestamps, self.timestamps
            ):
                raise ValueError(
                    "Los offsets por evento requieren targets alineados con timestamps; "
                    "target_timestamps independiente sólo se admite con horizontes físicos."
                )

        random_horizon_values = bool(
            self.uses_physical_horizons
            and (
                not self.horizon_choices
                or self.k_targets < len(self.horizon_choices)
            )
        )
        random_offset_values = bool(
            not self.uses_physical_horizons
            and not self.single_target_offset
            and self.k_targets < self.num_available_offsets
        )
        self.epoch_dependent_sampling = bool(
            not self.fixed_history_length
            or random_horizon_values
            or random_offset_values
            or self.config.randomize_query_order
            or self.config.history_subsampling == "random"
        )

        # Sólo se reserva este índice (potencialmente grande) cuando realmente
        # se solicita interpolación. El modo ``next`` no añade memoria O(T).
        self._valid_target_indices = (
            [
                torch.where(torch.isfinite(self.targets[:, channel]))[0]
                for channel in range(self.output_dim)
            ]
            if self.uses_physical_horizons
            and self.config.target_match_mode == "linear"
            else []
        )

    def _update_forecast_origin_audit(self) -> None:
        """Expone un resumen serializable de la selección de orígenes.

        Los descartes son mutuamente excluyentes: cuando un origen incumple
        varias condiciones se asigna a la primera causa en ``cause_order``.
        Una historia sin ninguna observación real se descarta; nunca se crea un
        token sintético para convertirla en una muestra válida.
        """

        discarded = int(sum(self.forecast_origin_discard_counts.values()))
        self.num_discarded_forecast_origins = discarded
        self.forecast_origin_audit = {
            "uses_explicit_origins": self.forecast_origin_timestamps is not None,
            "candidate_count": int(self.forecast_origin_candidate_count),
            "accepted_count": int(self.forecast_origin_accepted_count),
            "discarded_count": discarded,
            "discarded_by_cause": dict(self.forecast_origin_discard_counts),
            "cause_order": list(self._FORECAST_ORIGIN_DISCARD_CAUSES),
            "discard_assignment_policy": "first_matching_cause",
            "empty_history_policy": (
                "discard_origin_without_synthetic_observation"
            ),
        }

    def _configure_history_observation_index(
        self, row_has_observation: torch.Tensor
    ) -> None:
        """Prepara conteos O(1) de observaciones reales por intervalo de filas."""

        valid = torch.as_tensor(row_has_observation, dtype=torch.bool)
        if valid.ndim != 1 or valid.shape[0] != self.timestamps.shape[0]:
            raise ValueError("row_has_observation debe tener shape [N].")
        if bool(valid.all()):
            # El modo largo usual tiene una observación real en cada fila. No
            # reservar un prefijo int64 evita ~80 MB extra para N=10 millones.
            self._history_observation_prefix: Optional[torch.Tensor] = None
            return
        prefix = torch.zeros(valid.numel() + 1, dtype=torch.int64)
        prefix[1:] = valid.to(torch.int64).cumsum(dim=0)
        self._history_observation_prefix = prefix

    def _history_observation_counts(
        self, starts: torch.Tensor, stops: torch.Tensor
    ) -> torch.Tensor:
        prefix = getattr(self, "_history_observation_prefix", None)
        if prefix is None:
            return (stops - starts).clamp_min(0)
        return prefix[stops.to(torch.long)] - prefix[starts.to(torch.long)]

    def set_epoch(self, epoch: int) -> None:
        """Selecciona una época reproducible para historias/horizontes aleatorios."""
        if int(epoch) < 0:
            raise ValueError("epoch debe ser >= 0.")
        self._epoch = int(epoch)

    def _rng(self, idx: int, stream: int) -> np.random.Generator:
        seed = int(self.config.sampling_seed)
        if seed < 0:
            raise ValueError("sampling_seed debe ser >= 0.")
        sequence = np.random.SeedSequence([seed, self._epoch, int(idx), int(stream)])
        return np.random.default_rng(sequence)

    def _choose_history_length(self, idx: int, anchor: int) -> int:
        if self.fixed_history_length:
            return min(self.history_length, anchor)
        rng = self._rng(idx, stream=0)
        h = int(rng.integers(self.min_history_length, self.history_length + 1))
        return min(h, anchor)

    def _subsample_history_positions(
        self,
        timestamps: torch.Tensor,
        limit: int,
        idx: int,
    ) -> torch.Tensor:
        """Selecciona posiciones representativas, ordenadas y con el final."""
        count = int(timestamps.numel())
        if count <= limit:
            return torch.arange(count, dtype=torch.long)
        if limit == 1:
            return torch.tensor([count - 1], dtype=torch.long)

        mode = self.config.history_subsampling
        if mode == "uniform_index":
            positions = torch.linspace(0, count - 1, steps=limit).round().to(torch.long)
        elif mode == "random":
            interior_needed = max(0, limit - 2)
            interior = self._rng(idx, stream=4).choice(
                np.arange(1, count - 1, dtype=np.int64),
                size=interior_needed,
                replace=False,
            )
            positions = torch.as_tensor(
                np.concatenate(([0], np.sort(interior), [count - 1])),
                dtype=torch.long,
            )
        else:
            # La consulta temporal encuentra la observación más cercana a cada
            # punto de una grilla física. En gaps grandes pueden repetirse; se
            # completa luego con cuantiles por índice para retener densidad.
            grid = torch.linspace(
                timestamps[0].item(),
                timestamps[-1].item(),
                steps=limit,
                dtype=torch.float64,
            )
            right = torch.searchsorted(timestamps, grid, right=False).clamp(max=count - 1)
            left = (right - 1).clamp(min=0)
            choose_left = (grid - timestamps[left]).abs() <= (
                timestamps[right] - grid
            ).abs()
            positions = torch.where(choose_left, left, right)
            positions = torch.unique(
                torch.cat(
                    [
                        positions,
                        torch.tensor([0, count - 1], dtype=torch.long),
                    ]
                ),
                sorted=True,
            )
            if positions.numel() < limit:
                index_quantiles = torch.linspace(
                    0, count - 1, steps=limit
                ).round().to(torch.long)
                positions = torch.unique(
                    torch.cat([positions, index_quantiles]), sorted=True
                )
            if positions.numel() > limit:
                keep = torch.linspace(
                    0, positions.numel() - 1, steps=limit
                ).round().to(torch.long)
                positions = positions[keep]

        positions = torch.unique(positions, sorted=True)
        # Con count > limit, linspace produce índices únicos. Esta guarda hace
        # robusto el contrato ante futuras estrategias/duplicados.
        if positions.numel() < limit:
            available = np.setdiff1d(
                np.arange(count, dtype=np.int64),
                positions.numpy(),
                assume_unique=True,
            )
            fill = available[: limit - positions.numel()]
            positions = torch.unique(
                torch.cat([positions, torch.from_numpy(fill)]), sorted=True
            )
        positions[-1] = count - 1
        return torch.sort(positions[:limit]).values

    @staticmethod
    def _represented_counts(count: int, positions: torch.Tensor) -> torch.Tensor:
        """Número de filas originales representadas por cada fila seleccionada."""
        if positions.numel() == 1:
            return torch.tensor([float(count)], dtype=torch.float32)
        boundaries = torch.empty(positions.numel() + 1, dtype=torch.long)
        boundaries[0] = 0
        boundaries[-1] = count
        boundaries[1:-1] = (positions[:-1] + positions[1:] + 1) // 2
        return (boundaries[1:] - boundaries[:-1]).to(torch.float32)

    def _history_rows(
        self,
        idx: int,
        anchor: int,
        forecast_origin: Optional[torch.Tensor] = None,
        *,
        subsample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Devuelve filas, masa representada y gaps antes del subsampling."""
        if subsample and self._history_cache_enabled:
            cached = self._dense_history_cache.get(int(idx))
            if cached is not None:
                return cached

        start, stop = self._history_bounds(idx, anchor, forecast_origin)
        original_times = self.timestamps[start:stop]
        count = int(original_times.numel())
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.config.compute_history_diagnostics:
            original_gaps = torch.diff(original_times)
            diagnostics = {
                "past_original_observation_count": torch.as_tensor(
                    count, dtype=torch.float64
                ),
                "past_original_max_gap": (
                    original_gaps.max()
                    if original_gaps.numel()
                    else torch.zeros((), dtype=torch.float64)
                ),
                "past_original_median_gap": (
                    original_gaps.median()
                    if original_gaps.numel()
                    else torch.zeros((), dtype=torch.float64)
                ),
            }
        if not subsample:
            return (
                torch.arange(start, stop, dtype=torch.long),
                torch.ones(count, dtype=torch.float32),
                diagnostics,
            )
        positions = self._subsample_history_positions(
            original_times, self.max_history_events, idx
        )
        counts = self._represented_counts(count, positions)
        result = (start + positions, counts, diagnostics)
        if self._history_cache_enabled:
            self._dense_history_cache[int(idx)] = result
        return result

    def _history_bounds(
        self,
        idx: int,
        anchor: int,
        forecast_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Límites contiguos de historia sin materializar un ``arange`` grande."""
        if self.history_duration is None:
            h = self._choose_history_length(idx, anchor)
            return anchor - h, anchor
        if forecast_origin is None:
            forecast_origin = self.timestamps[anchor - 1]
        cutoff = forecast_origin - self.history_duration
        start = int(torch.searchsorted(self.timestamps, cutoff, right=False).item())
        if start >= anchor:
            start = max(0, anchor - 1)
        return start, anchor

    def _build_external_forecast_examples(self) -> torch.Tensor:
        """Filtra orígenes explícitos y audita una causa por descarte.

        La precedencia de causas es estable y está publicada en
        ``forecast_origin_audit["cause_order"]``. En particular, una ventana
        que contiene filas temporales pero ninguna medición real se clasifica
        como ``empty_history`` y no llega a ``__getitem__``.
        """
        if self.forecast_origin_timestamps is None or self.history_duration is None:
            raise RuntimeError("No hay orígenes de forecast externos configurados.")

        origins = self.forecast_origin_timestamps
        cutoffs = origins - self.history_duration
        history_starts = torch.searchsorted(self.timestamps, cutoffs, right=False)
        anchors = torch.searchsorted(self.timestamps, origins, right=True)
        observation_counts = self._history_observation_counts(
            history_starts, anchors
        )

        failure_masks = {
            "insufficient_history_coverage": cutoffs < self.timestamps[0],
            # Un origen dentro de un gap sí es válido. Uno posterior al último
            # evento se descarta conservadoramente porque no se puede distinguir
            # de una fuente truncada.
            "origin_after_last_observation": origins > self.timestamps[-1],
            "empty_history": observation_counts <= 0,
            "target_before_available_range": (
                origins + self.min_horizon < self.target_timestamps[0]
            ),
            "target_after_available_range": (
                origins + self.max_horizon > self.target_timestamps[-1]
            ),
        }
        valid = torch.ones_like(origins, dtype=torch.bool)
        discard_counts: Dict[str, int] = {}
        for cause in self._FORECAST_ORIGIN_DISCARD_CAUSES:
            rejected = valid & failure_masks[cause]
            discard_counts[cause] = int(rejected.sum().item())
            valid &= ~rejected

        self.forecast_origin_candidate_count = int(origins.numel())
        self.forecast_origin_accepted_count = int(valid.sum().item())
        self.forecast_origin_discard_counts = discard_counts
        self._update_forecast_origin_audit()
        self._example_forecast_origins = origins[valid].contiguous()
        valid_anchors = anchors[valid].to(torch.long).contiguous()
        if valid_anchors.numel() == 0:
            raise ValueError(
                "Ningún forecast_origin tiene historia física completa y targets "
                "hasta max_horizon. Descartes: "
                f"{self.forecast_origin_discard_counts}."
            )
        return valid_anchors

    def _forecast_origin(self, idx: int, anchor: int) -> torch.Tensor:
        if self._example_forecast_origins is not None:
            return self._example_forecast_origins[idx]
        return self.timestamps[anchor - 1]

    def _build_anchor_indices(self, total_rows: int) -> Union[range, torch.Tensor]:
        """Construye anchors contiguos como ``range`` sin materializar millones."""
        if self.forecast_origin_timestamps is not None:
            return self._build_external_forecast_examples()

        stride = int(self.config.stride)
        base_start = 1 if self.history_duration is not None else self.history_length
        earliest_anchor = base_start

        if self.history_duration is not None:
            earliest_base_time = self.timestamps[0] + self.history_duration
            earliest_base_row = int(
                torch.searchsorted(self.timestamps, earliest_base_time, right=False).item()
            )
            earliest_anchor = max(earliest_anchor, earliest_base_row + 1)

        if self.uses_physical_horizons:
            earliest_query_base = self.target_timestamps[0] - self.min_horizon
            earliest_query_row = int(
                torch.searchsorted(self.timestamps, earliest_query_base, right=False).item()
            )
            earliest_anchor = max(earliest_anchor, earliest_query_row + 1)

            latest_base_time = self.target_timestamps[-1] - self.max_horizon
            latest_base_row = int(
                torch.searchsorted(self.timestamps, latest_base_time, right=True).item()
            ) - 1
            # Con una grilla de verdad que se extiende más allá de las
            # observaciones, también es válido usar la última observación como
            # fin de historia (anchor == total_rows).
            latest_anchor = min(total_rows, latest_base_row + 1)
        else:
            latest_anchor = total_rows - 1 - self.max_offset

        phase = base_start
        if earliest_anchor > phase:
            steps = (earliest_anchor - phase + stride - 1) // stride
            first_anchor = phase + steps * stride
        else:
            first_anchor = phase
        if first_anchor > latest_anchor:
            target_description = (
                f"max_horizon={self.max_horizon}"
                if self.uses_physical_horizons
                else f"max_target_offset={self.max_offset}"
            )
            raise ValueError(
                "No hay ejemplos con historia suficiente y " + target_description + "."
            )
        return range(first_anchor, latest_anchor + 1, stride)

    def _choose_offsets(self, idx: int) -> List[int]:
        if self.single_target_offset:
            return list(self.offsets)
        rng = self._rng(idx, stream=1)
        selected = rng.choice(
            np.asarray(self.offsets, dtype=np.int64),
            size=self.k_targets,
            replace=False,
        ).tolist()
        selected = [int(value) for value in selected]
        if not self.config.randomize_query_order:
            selected.sort()
        return selected

    def _choose_horizons(self, idx: int) -> List[float]:
        rng = self._rng(idx, stream=2)
        if self.horizon_choices:
            if self.k_targets == len(self.horizon_choices):
                selected = list(self.horizon_choices)
            else:
                selected = rng.choice(
                    np.asarray(self.horizon_choices, dtype=np.float64),
                    size=self.k_targets,
                    replace=False,
                ).tolist()
        elif self.config.target_horizon_sampling == "log_uniform":
            selected = np.exp(
                rng.uniform(
                    np.log(self.min_horizon),
                    np.log(self.max_horizon),
                    size=self.k_targets,
                )
            ).tolist()
        else:
            selected = rng.uniform(
                self.min_horizon,
                self.max_horizon,
                size=self.k_targets,
            ).tolist()

        selected = [float(value) for value in selected]
        if not self.config.randomize_query_order:
            selected.sort()
        return selected

    def _permutation(self, idx: int, count: int) -> torch.Tensor:
        if not self.config.randomize_query_order or count <= 1:
            return torch.arange(count, dtype=torch.long)
        permutation = self._rng(idx, stream=3).permutation(count)
        return torch.from_numpy(permutation.astype(np.int64, copy=False))

    def _linear_targets(
        self,
        query_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpola por canal y expone los dos timestamps fuente usados."""
        count = int(query_timestamps.numel())
        values = torch.zeros((count, self.output_dim), dtype=torch.float32)
        mask = torch.zeros((count, self.output_dim), dtype=torch.float32)
        sources = torch.full(
            (count, self.output_dim, 2),
            float("nan"),
            dtype=torch.float64,
        )

        for channel, valid_indices in enumerate(self._valid_target_indices):
            if valid_indices.numel() == 0:
                continue
            valid_times = self.target_timestamps[valid_indices]
            valid_values = self.targets[valid_indices, channel]
            positions = torch.searchsorted(valid_times, query_timestamps, right=False)
            for query_idx, position_tensor in enumerate(positions):
                position = int(position_tensor.item())
                query = query_timestamps[query_idx]

                if position < valid_times.numel() and bool(valid_times[position] == query):
                    values[query_idx, channel] = valid_values[position]
                    mask[query_idx, channel] = 1.0
                    sources[query_idx, channel] = valid_times[position]
                    continue
                if position == 0 or position >= valid_times.numel():
                    continue

                left = position - 1
                right = position
                t_left = valid_times[left]
                t_right = valid_times[right]
                denominator = t_right - t_left
                if bool(denominator <= 0.0):
                    continue
                weight = ((query - t_left) / denominator).to(torch.float32)
                values[query_idx, channel] = (
                    valid_values[left] + weight * (valid_values[right] - valid_values[left])
                )
                mask[query_idx, channel] = 1.0
                sources[query_idx, channel, 0] = t_left
                sources[query_idx, channel, 1] = t_right
        return values, mask, sources

    def _build_targets(
        self,
        anchor: int,
        idx: int,
        forecast_origin: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Construye targets y metadatos manteniendo toda alineación temporal."""
        base_timestamp = (
            forecast_origin
            if forecast_origin is not None
            else self.timestamps[anchor - 1]
        )

        if self.uses_physical_horizons:
            requested_horizons = torch.as_tensor(
                self._choose_horizons(idx), dtype=torch.float64
            )
            requested_timestamps = base_timestamp + requested_horizons

            if self.config.target_match_mode == "next":
                target_indices = torch.searchsorted(
                    self.target_timestamps, requested_timestamps, right=False
                )
                if bool((target_indices >= self.target_timestamps.numel()).any()):
                    raise RuntimeError(
                        "Una consulta física quedó fuera de la serie; revise max_horizon."
                    )
                query_timestamps = self.target_timestamps[target_indices]
                raw_values = self.targets[target_indices]
                target_mask = torch.isfinite(raw_values).to(torch.float32)
                target_values = torch.nan_to_num(
                    raw_values, nan=0.0, posinf=0.0, neginf=0.0
                )
                sources = query_timestamps[:, None, None].expand(
                    -1, self.output_dim, 2
                ).clone()
            else:
                query_timestamps = requested_timestamps
                target_values, target_mask, sources = self._linear_targets(
                    query_timestamps
                )
        else:
            chosen_offsets = self._choose_offsets(idx)
            target_indices = torch.as_tensor(
                [anchor + offset for offset in chosen_offsets], dtype=torch.long
            )
            query_timestamps = self.timestamps[target_indices]
            requested_timestamps = query_timestamps.clone()
            requested_horizons = query_timestamps - base_timestamp
            raw_values = self.targets[target_indices]
            target_mask = torch.isfinite(raw_values).to(torch.float32)
            target_values = torch.nan_to_num(
                raw_values, nan=0.0, posinf=0.0, neginf=0.0
            )
            sources = query_timestamps[:, None, None].expand(
                -1, self.output_dim, 2
            ).clone()

        permutation = self._permutation(idx, int(query_timestamps.numel()))
        query_timestamps = query_timestamps[permutation]
        requested_timestamps = requested_timestamps[permutation]
        requested_horizons = requested_horizons[permutation]
        target_values = target_values[permutation]
        target_mask = target_mask[permutation]
        sources = sources[permutation]

        return {
            "target_timestamp": query_timestamps,
            "target_values": target_values,
            "target_loss_mask": target_mask,
            "target_horizons": query_timestamps - base_timestamp,
            "requested_target_horizons": requested_horizons,
            "requested_target_timestamps": requested_timestamps,
            "target_source_timestamps": sources,
            # Origen semántico del forecast (último timestamp histórico). No es
            # el origen numérico que SequenceBuilder usa para relativizar.
            "forecast_origin": base_timestamp,
        }


class TimeSeriesDataset(_TemporalTargetMixin, Dataset):
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
    - "target_timestamp": [num_targets], siempre float64 sin relativizar.
    - "target_values": [num_targets, output_dim]
    - "target_horizons": distancia física efectiva desde la última observación.
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
        target_timestamps: Optional[ArrayLike] = None,
        forecast_origin_timestamps: Optional[ArrayLike] = None,
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
        target_timestamps:
            Grilla temporal opcional de ``targets``. Permite usar, por ejemplo,
            el ``truth.parquet`` latente como verdad limpia y las observaciones
            irregulares como historia. Sólo aplica a horizontes físicos.
        forecast_origin_timestamps:
            Orígenes físicos explícitos e independientes de los eventos. Cada
            historia incluye observaciones ``<= origen`` dentro de
            ``history_duration``. Permite evaluar consultas dentro de gaps.
        sequence_builder:
            Instancia opcional de SequenceBuilder para transformar el sample.
        """
        super().__init__()

        # Valores en fp32; tiempo absoluto en fp64 hasta construir deltas por
        # ventana. Esto evita que epochs/UNIX timestamps colapsen gaps pequeños.
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
        if self.values.shape[1] < self.input_dim:
            raise ValueError("input_dim supera values.shape[1].")

        if targets is not None:
            self.targets = self._to_torch_2d(targets).contiguous()
            if self.targets.shape[1] != self.output_dim:
                raise ValueError("output_dim no coincide con targets.shape[1].")
            if target_timestamps is None and self.targets.shape[0] != self.values.shape[0]:
                raise ValueError(
                    "targets y values deben tener la misma longitud si no se "
                    "proporciona target_timestamps."
                )
        else:
            if target_timestamps is not None:
                raise ValueError("target_timestamps requiere targets explícitos.")
            # Tomamos los targets como un subset de `values`
            total_dim = self.values.shape[1]
            if self.input_dim + self.output_dim > total_dim:
                raise ValueError(
                    f"input_dim + output_dim ({self.input_dim} + {self.output_dim}) "
                    f"supera la dimensión total de values ({total_dim})."
                )
            self.targets = self.values[:, self.input_dim : self.input_dim + self.output_dim].contiguous()

        self.target_timestamps = (
            self._to_torch_1d(target_timestamps).contiguous()
            if target_timestamps is not None
            else self.timestamps
        )
        self.forecast_origin_timestamps = (
            self._to_torch_1d(forecast_origin_timestamps).contiguous()
            if forecast_origin_timestamps is not None
            else None
        )

        self._init_temporal_targeting()
        self._configure_history_observation_index(
            torch.isfinite(self.values[:, : self.input_dim]).any(dim=1)
        )

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
        return x.to(torch.float64)

    def _build_example_indices(self) -> Union[range, torch.Tensor]:
        """
        Construye la lista de índices 'anchor' para cada ejemplo.
        """
        return self._build_anchor_indices(int(self.values.shape[0]))

    def __len__(self) -> int:
        return len(self._example_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor = int(self._example_indices[idx])
        forecast_origin = self._forecast_origin(idx, anchor)
        history_rows, represented_counts, history_diagnostics = self._history_rows(
            idx, anchor, forecast_origin
        )

        past_values = self.values[history_rows, :self.input_dim]
        past_timestamps = self.timestamps[history_rows]
        if not bool(torch.isfinite(past_values).any()):
            raise RuntimeError(
                "La ventana histórica no contiene observaciones reales; "
                "no se fabrican tokens para historias vacías."
            )
        sample = {
            "past_values": past_values,
            "past_timestamps": past_timestamps,
            "past_observation_counts": represented_counts,
            "last_observation_age": forecast_origin - past_timestamps[-1],
            **history_diagnostics,
            **self._build_targets(anchor, idx, forecast_origin),
        }
        if self.config.compute_history_diagnostics:
            sample.update(
                {
                    "past_sensor_observation_count": history_diagnostics[
                        "past_original_observation_count"
                    ].repeat(self.input_dim),
                    "past_sensor_max_gap": history_diagnostics[
                        "past_original_max_gap"
                    ].repeat(self.input_dim),
                    "past_sensor_median_gap": history_diagnostics[
                        "past_original_median_gap"
                    ].repeat(self.input_dim),
                    "sensor_last_observation_age": (
                        forecast_origin - past_timestamps[-1]
                    ).repeat(self.input_dim),
                }
            )

        # Optimización 1.1: Aplicar sequence_builder aquí si existe
        if self.sequence_builder is not None:
            return self.sequence_builder(sample)
        
        return sample


class EventTimeSeriesDataset(_TemporalTargetMixin, Dataset):
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
        target_timestamps: Optional[ArrayLike] = None,
        forecast_origin_timestamps: Optional[ArrayLike] = None,
        event_sensor_ids: Optional[ArrayLike] = None,
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
        self.event_sensor_ids: Optional[torch.Tensor] = None
        if event_sensor_ids is not None:
            sensor_ids = torch.as_tensor(event_sensor_ids, dtype=torch.long)
            if sensor_ids.ndim != 1 or sensor_ids.shape[0] != self.values.shape[0]:
                raise ValueError("event_sensor_ids debe tener shape [N].")
            if torch.any(sensor_ids < 0) or torch.any(sensor_ids >= self.input_dim):
                raise ValueError("event_sensor_ids contiene un sensor fuera de rango.")
            if self.values.shape[1] != 1:
                raise ValueError(
                    "El modo evento largo requiere values con shape [N,1]."
                )
            if not bool(torch.isfinite(self.values).all()):
                raise ValueError("El modo evento largo no admite values NaN/Inf.")
            self.event_sensor_ids = sensor_ids.contiguous()

        if self.values.shape[0] != self.timestamps.shape[0]:
            raise ValueError("values y timestamps deben tener la misma longitud temporal.")
        if target_timestamps is None and self.targets.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                "targets y timestamps deben tener la misma longitud si no se "
                "proporciona target_timestamps."
            )

        self.target_timestamps = (
            TimeSeriesDataset._to_torch_1d(target_timestamps).contiguous()
            if target_timestamps is not None
            else self.timestamps
        )
        self.forecast_origin_timestamps = (
            TimeSeriesDataset._to_torch_1d(
                forecast_origin_timestamps
            ).contiguous()
            if forecast_origin_timestamps is not None
            else None
        )

        if self.event_sensor_ids is None and self.values.shape[1] != self.input_dim:
            raise ValueError("input_dim no coincide con values.shape[1].")
        if self.targets.shape[1] != self.output_dim:
            raise ValueError("output_dim no coincide con targets.shape[1].")

        self._init_temporal_targeting()
        if self.event_sensor_ids is not None:
            # El constructor del modo largo ya validó que cada fila es un
            # evento finito; evitamos construir un prefijo de N+1 enteros.
            self._history_observation_prefix = None
        else:
            self._configure_history_observation_index(
                torch.isfinite(self.values).any(dim=1)
            )

        if (
            self.history_duration is not None
            or self.config.max_history_events is not None
        ) and self.max_history_events < self.input_dim:
            raise ValueError(
                "max_history_events debe ser >= num_sensors para preservar "
                "el último evento de cada sensor."
            )

        self._event_history_cache: Dict[
            int,
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        ] = {}
        self._event_history_cache_enabled = bool(
            self._history_cache_enabled and self.event_sensor_ids is not None
        )
        # Los diagnósticos exactos de test se calculan sobre ventanas previas al
        # subsampling. Preindexar una vez las filas de cada sensor evita volver
        # a enmascarar toda la ventana (potencialmente cientos de miles de
        # eventos) una vez por sensor y forecast origin.
        self._sensor_event_rows: Tuple[torch.Tensor, ...] = ()
        if self.event_sensor_ids is not None and self.config.compute_history_diagnostics:
            self._sensor_event_rows = tuple(
                torch.where(self.event_sensor_ids == sensor)[0]
                for sensor in range(self.input_dim)
            )
        self._example_indices = self._build_example_indices()

    def _build_example_indices(self) -> Union[range, torch.Tensor]:
        return self._build_anchor_indices(int(self.values.shape[0]))

    def _subsample_event_positions(
        self,
        event_timestamps: torch.Tensor,
        event_sensor_ids: torch.Tensor,
        limit: int,
        idx: int,
    ) -> torch.Tensor:
        """Reduce eventos preservando cobertura y el último de cada sensor."""
        count = int(event_timestamps.numel())
        if count <= limit:
            return torch.arange(count, dtype=torch.long)

        # Un único reduce reemplaza un scan completo de la ventana por sensor.
        positions = torch.arange(count, dtype=torch.long)
        last_by_sensor = torch.full((self.input_dim,), -1, dtype=torch.long)
        last_by_sensor.scatter_reduce_(
            0,
            event_sensor_ids,
            positions,
            reduce="amax",
            include_self=True,
        )
        mandatory = set(last_by_sensor[last_by_sensor >= 0].tolist())
        if len(mandatory) > limit:
            raise ValueError(
                "El límite de historia no permite retener un evento por sensor."
            )
        # Retener también el inicio físico cuando queda presupuesto.
        if len(mandatory) < limit:
            mandatory.add(0)

        mandatory_tensor = torch.as_tensor(sorted(mandatory), dtype=torch.long)
        if mandatory_tensor.numel() == limit:
            return mandatory_tensor

        # Elegir cobertura temporal global y luego insertar los últimos eventos
        # obligatorios evita candidate_mask/torch.where sobre toda la ventana.
        proposed = self._subsample_history_positions(event_timestamps, limit, idx)
        optional = proposed[~torch.isin(proposed, mandatory_tensor)]
        remaining_budget = limit - int(mandatory_tensor.numel())
        if optional.numel() < remaining_budget:
            fallback = torch.linspace(
                0, count - 1, steps=limit * 2
            ).round().to(torch.long)
            optional = torch.unique(
                torch.cat(
                    (optional, fallback[~torch.isin(fallback, mandatory_tensor)])
                ),
                sorted=True,
            )
        if optional.numel() > remaining_budget:
            keep = torch.linspace(
                0, optional.numel() - 1, steps=remaining_budget
            ).round().to(torch.long)
            optional = optional[keep]
        return torch.sort(torch.cat((mandatory_tensor, optional))).values

    def _sensor_diagnostics(
        self,
        event_timestamps: torch.Tensor,
        event_sensor_ids: torch.Tensor,
        forecast_origin: torch.Tensor,
        *,
        history_start: Optional[int] = None,
        history_stop: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        use_preindex = bool(
            self._sensor_event_rows
            and history_start is not None
            and history_stop is not None
        )
        counts = (
            torch.zeros(self.input_dim, dtype=torch.float64)
            if use_preindex
            else torch.bincount(
                event_sensor_ids, minlength=self.input_dim
            ).to(torch.float64)
        )
        max_gaps = torch.zeros(self.input_dim, dtype=torch.float64)
        median_gaps = torch.zeros(self.input_dim, dtype=torch.float64)
        last_ages = torch.full(
            (self.input_dim,),
            float(self.history_duration or 0.0),
            dtype=torch.float64,
        )
        for sensor in range(self.input_dim):
            if use_preindex:
                rows = self._sensor_event_rows[sensor]
                left = int(
                    torch.searchsorted(rows, int(history_start), right=False).item()
                )
                right = int(
                    torch.searchsorted(rows, int(history_stop), right=False).item()
                )
                selected_rows = rows[left:right]
                sensor_times = self.timestamps[selected_rows]
                counts[sensor] = int(selected_rows.numel())
            else:
                sensor_times = event_timestamps[event_sensor_ids == sensor]
            if sensor_times.numel():
                last_ages[sensor] = forecast_origin - sensor_times[-1]
            gaps = torch.diff(sensor_times)
            if gaps.numel():
                max_gaps[sensor] = gaps.max()
                median_gaps[sensor] = gaps.median()
        return {
            "past_sensor_observation_count": counts,
            "past_sensor_max_gap": max_gaps,
            "past_sensor_median_gap": median_gaps,
            "sensor_last_observation_age": last_ages,
        }

    def get_approx_lengths(self) -> List[int]:
        """
        Devuelve una lista con la cantidad aproximada de tokens (eventos) 
        esperada para cada ejemplo. Útil para el BucketBatchSampler.
        """
        # En modo largo cada fila ya es exactamente un evento.
        valid_counts = (
            torch.ones(self.values.shape[0], dtype=torch.long)
            if self.event_sensor_ids is not None
            else (~torch.isnan(self.values)).sum(dim=1)
        )
        lengths = []
        for idx, anchor in enumerate(self._example_indices):
            anchor_int = int(anchor)
            start, stop = self._history_bounds(
                idx, anchor_int, self._forecast_origin(idx, anchor_int)
            )
            # Evita materializar un arange del tamaño de la ventana (cientos de
            # miles de filas en los datasets de 10M eventos).
            event_count = (
                stop - start
                if self.event_sensor_ids is not None
                else int(valid_counts[start:stop].sum().item())
            )
            if self.history_duration is not None or self.config.max_history_events is not None:
                event_count = min(event_count, self.max_history_events)
            target_tokens = (
                int(self.sequence_builder.num_target_tokens)
                if self.sequence_builder is not None
                else 1
            )
            lengths.append(event_count + self.k_targets * target_tokens)

        return lengths

    def __len__(self) -> int:
        return len(self._example_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor = int(self._example_indices[idx])
        forecast_origin = self._forecast_origin(idx, anchor)
        cached = (
            self._event_history_cache.get(int(idx))
            if self._event_history_cache_enabled
            else None
        )
        if cached is not None:
            event_rows, event_observation_counts, history_diagnostics = cached
            event_values = self.values[event_rows]
            event_timestamps = self.timestamps[event_rows]
            if self.event_sensor_ids is None:
                raise RuntimeError("Cache event-based requiere event_sensor_ids.")
            event_sensor_ids = self.event_sensor_ids[event_rows]
        else:
            start, stop = self._history_bounds(idx, anchor, forecast_origin)
            hist_values = self.values[start:stop]
            hist_timestamps = self.timestamps[start:stop]
            history_diagnostics = {}

            if self.event_sensor_ids is not None:
                event_values = hist_values
                event_timestamps = hist_timestamps
                event_sensor_ids = self.event_sensor_ids[start:stop]
            else:
                # Compatibilidad con la representación wide/NaN histórica.
                valid_mask = torch.isfinite(hist_values)
                rows, cols = torch.where(valid_mask)
                event_values = hist_values[rows, cols].view(-1, 1)
                event_timestamps = hist_timestamps[rows]
                event_sensor_ids = cols

            if event_values.shape[0] > 0:
                event_observation_counts = torch.ones(
                    event_values.shape[0], dtype=torch.float32
                )
                if self.config.compute_history_diagnostics:
                    event_gaps = torch.diff(event_timestamps)
                    history_diagnostics = {
                        "past_original_observation_count": torch.as_tensor(
                            event_values.shape[0], dtype=torch.float64
                        ),
                        "past_original_max_gap": (
                            event_gaps.max()
                            if event_gaps.numel()
                            else torch.zeros((), dtype=torch.float64)
                        ),
                        "past_original_median_gap": (
                            event_gaps.median()
                            if event_gaps.numel()
                            else torch.zeros((), dtype=torch.float64)
                        ),
                        **self._sensor_diagnostics(
                            event_timestamps,
                            event_sensor_ids,
                            forecast_origin,
                            history_start=start,
                            history_stop=stop,
                        ),
                    }
                if (
                    self.history_duration is not None
                    or self.config.max_history_events is not None
                ) and event_values.shape[0] > self.max_history_events:
                    event_positions = self._subsample_event_positions(
                        event_timestamps,
                        event_sensor_ids,
                        self.max_history_events,
                        idx,
                    )
                    event_observation_counts = self._represented_counts(
                        int(event_values.shape[0]), event_positions
                    )
                    event_values = event_values[event_positions]
                    event_timestamps = event_timestamps[event_positions]
                    event_sensor_ids = event_sensor_ids[event_positions]
                else:
                    event_positions = torch.arange(
                        event_values.shape[0], dtype=torch.long
                    )
            else:
                raise RuntimeError(
                    "La ventana histórica no contiene observaciones reales; "
                    "no se fabrican tokens para historias vacías."
                )

            if self._event_history_cache_enabled:
                event_rows = start + event_positions
                self._event_history_cache[int(idx)] = (
                    event_rows,
                    event_observation_counts,
                    history_diagnostics,
                )

        sample = {
            "past_values": event_values,
            "past_timestamps": event_timestamps,
            "past_sensor_ids": event_sensor_ids,
            "past_observation_counts": event_observation_counts,
            "last_observation_age": forecast_origin - event_timestamps[-1],
            **history_diagnostics,
            **self._build_targets(anchor, idx, forecast_origin),
        }

        # Optimización 1.1: Aplicar sequence_builder aquí si existe
        if self.sequence_builder is not None:
            return self.sequence_builder(sample)
        
        return sample
