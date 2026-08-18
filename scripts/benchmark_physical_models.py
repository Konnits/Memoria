"""Benchmark neural sobre consultas en tiempo físico.

Entrena arquitecturas para series irregulares usando ``observations.parquet``
como historia ruidosa y ``truth.parquet`` como target continuo independiente.
El protocolo evita offsets por fila: cada historia de duración fija se consulta
en los horizontes físicos configurados y el orden de esas consultas se permuta.

Ejemplo acotado::

    conda run -n memoria python scripts/benchmark_physical_models.py \
        --kinds univariate multivariate --limit-datasets-per-kind 1 \
        --models QueryCross NoTime QueryOnly CTSSM Persistence \
        --epochs 1 --max-train-samples 32 --max-val-samples 16 \
        --max-test-samples 16 --max-observation-rows-per-split 20000

``--validate-only`` construye los datos, instancia todos los modelos y ejecuta
inferencia mediante :class:`Trainer`, sin optimización ni checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as pads
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")])

from ts_transformer.data import (  # noqa: E402
    EventTimeSeriesDataset,
    SequenceBuilder,
    TimeSeriesDataset,
    build_collate_fn,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig  # noqa: E402
from ts_transformer.models.continuous_basis_decoder import (  # noqa: E402
    ContinuousBasisDecoderConfig,
    TimeSeriesContinuousBasisDecoder,
)
from ts_transformer.models.query_cross_attention import (  # noqa: E402
    QueryCrossAttentionConfig,
    TimeSeriesQueryCrossAttention,
)
from ts_transformer.models.time_series_transformer import (  # noqa: E402
    TimeSeriesTransformerConfig,
)
from ts_transformer.training import Trainer  # noqa: E402
from ts_transformer.training.metrics import (  # noqa: E402
    compute_gaussian_metrics,
    compute_regression_metrics,
    compute_structured_regression_metrics,
)
from ts_transformer.training.optimizers import OptimizerConfig  # noqa: E402
from ts_transformer.training.train_loop import TrainingConfig  # noqa: E402
from ts_transformer.utils.seed import set_global_seed  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "benchmark" / "physical_models.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "physical_models"
MODEL_NAMES = (
    "QueryCross",
    "QueryCross-Gaussian",
    "BasisDecoder",
    "BasisDecoder-Gaussian",
    "BasisDecoder-CTSSM",
    "NoTime",
    "QueryOnly",
    "CTSSM",
    "Persistence",
)
TIMESTAMP_ABLATIONS = (
    "real",
    "real_no_count_feature",
    "all_equal_no_count_feature",
    # Aliases históricos: sólo eliminan la feature explícita de count, no los
    # gaps/ages que el modelo deriva desde los timestamps.
    "real_no_density",
    "all_equal",
    "all_equal_no_density",
    "permuted_gaps",
    "regular_grid",
    "query_only",
    "history_only",
)
SPLIT_NAMES = ("train", "validation", "test")

# El fingerprint incluye todo el paquete efectivo, no una lista parcial que
# pueda olvidar una dependencia nueva. Los tres entry points del protocolo se
# agregan explícitamente porque viven fuera de ``src``.
IMPLEMENTATION_SOURCE_PATHS = tuple(
    sorted((REPOSITORY_ROOT / "src/ts_transformer").rglob("*.py"))
) + (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "scripts/temporal_identifiability_benchmark.py",
    REPOSITORY_ROOT / "scripts/run_thesis_physical_benchmark.py",
)

_ABLATION_ALIASES = {
    "real_no_density": "real_no_count_feature",
    "all_equal_no_density": "all_equal_no_count_feature",
}


def canonical_ablation_name(name: str) -> str:
    """Devuelve el nombre científicamente preciso de una ablación."""
    return _ABLATION_ALIASES.get(name, name)


@dataclass(frozen=True)
class ChannelScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform_long(
        self, values: np.ndarray, channels: np.ndarray | None = None
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if channels is None:
            return ((values - self.mean[0]) / self.std[0]).astype(np.float32)
        channel_indices = np.asarray(channels, dtype=np.int64)
        return (
            (values - self.mean[channel_indices]) / self.std[channel_indices]
        ).astype(np.float32)

    def transform_matrix(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_channel(self, values: np.ndarray, channel: int) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std[channel] + self.mean[channel]


@dataclass
class PreparedPhysicalData:
    train: Dataset
    validation: Dataset
    test: Dataset
    scaler: ChannelScaler
    kind: str
    preset: str
    dataset_id: str
    n_channels: int
    max_history_events: int
    row_counts: dict[str, int]
    forecast_origin_audit: dict[str, dict[str, Any]] | None = None


class PhysicalCollate:
    """Collate estándar más masa/densidad de observaciones por token."""

    def __init__(self) -> None:
        self.base = build_collate_fn(pad_to_max_length=True)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base(samples)
        counts = batch["input_observation_counts"].clamp_min(0.0)
        batch["temporal_features"] = torch.log1p(counts).unsqueeze(-1)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QueryCross sobre horizontes físicos y truth limpio."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--kinds", nargs="+", choices=("univariate", "multivariate"), default=None
    )
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Filtra IDs exactos (p.ej. long_gaps_gseed3031_0000).",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=float, default=None)
    parser.add_argument("--train-horizon-min", type=float, default=None)
    parser.add_argument("--train-horizon-max", type=float, default=None)
    parser.add_argument(
        "--train-horizon-sampling",
        choices=("uniform", "log_uniform"),
        default=None,
    )
    parser.add_argument("--queries-per-sample", type=int, default=None)
    parser.add_argument("--history-duration", type=float, default=None)
    parser.add_argument("--max-history-events-univariate", type=int, default=None)
    parser.add_argument("--max-history-events-multivariate", type=int, default=None)
    parser.add_argument("--history-subsampling", choices=("uniform_time", "uniform_index", "random"), default=None)
    parser.add_argument("--limit-datasets-per-kind", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--max-observation-rows-per-split",
        type=int,
        default=None,
        help="Cap global opcional sólo para smoke tests; se registra en metadata.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="Activa algoritmos reproducibles y desactiva rutas CUDA no deterministas.",
    )
    deterministic.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Permite rutas rápidas no deterministas (no usar en el benchmark final).",
    )
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--cross-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--timestamp-ablations",
        nargs="+",
        choices=TIMESTAMP_ABLATIONS,
        default=None,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignora sentinels compatibles y vuelve a ejecutar cada run.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Configuración física inválida: {path}")
    return config


def _option(args: argparse.Namespace, name: str, config: Mapping[str, Any], key: str):
    value = getattr(args, name)
    return config[key] if value is None else value


def resolve_options(args: argparse.Namespace, raw: Mapping[str, Any]) -> argparse.Namespace:
    data = raw["data"]
    task = raw["task"]
    sampling = raw["sampling"]
    model = raw["model"]
    training = raw["training"]
    evaluation = raw["evaluation"]

    args.data_root = Path(args.data_root or data.get("root", REPOSITORY_ROOT / "data"))
    if not args.data_root.is_absolute():
        args.data_root = REPOSITORY_ROOT / args.data_root
    args.output_dir = Path(args.output_dir or raw.get("output_dir", DEFAULT_OUTPUT))
    if not args.output_dir.is_absolute():
        args.output_dir = REPOSITORY_ROOT / args.output_dir
    args.kinds = tuple(args.kinds or data.get("kinds", ("univariate", "multivariate")))
    args.models = tuple(args.models or raw.get("models", MODEL_NAMES))
    args.seeds = tuple(args.seeds or training.get("seeds", (42,)))
    args.horizons = tuple(float(value) for value in (args.horizons or task["horizons"]))
    train_range = task.get("train_horizon_range", [min(args.horizons), max(args.horizons)])
    args.train_horizon_min = float(
        args.train_horizon_min if args.train_horizon_min is not None else train_range[0]
    )
    args.train_horizon_max = float(
        args.train_horizon_max if args.train_horizon_max is not None else train_range[1]
    )
    args.train_horizon_sampling = (
        args.train_horizon_sampling or task.get("train_horizon_sampling", "log_uniform")
    )
    args.queries_per_sample = int(
        args.queries_per_sample
        if args.queries_per_sample is not None
        else task.get("queries_per_sample", len(args.horizons))
    )
    args.history_duration = float(_option(args, "history_duration", task, "history_duration"))
    args.max_history_events_univariate = int(
        _option(args, "max_history_events_univariate", task, "max_history_events_univariate")
    )
    args.max_history_events_multivariate = int(
        _option(args, "max_history_events_multivariate", task, "max_history_events_multivariate")
    )
    args.history_subsampling = args.history_subsampling or task.get("history_subsampling", "uniform_time")
    args.cache_deterministic_history = args.history_subsampling in {
        "uniform_time",
        "uniform_index",
    }
    declared_cache = task.get("cache_deterministic_history")
    if declared_cache is not None and bool(declared_cache) != args.cache_deterministic_history:
        raise ValueError(
            "cache_deterministic_history sólo puede activarse con "
            "uniform_time/uniform_index."
        )
    args.max_train_samples = int(_option(args, "max_train_samples", sampling, "max_train_samples"))
    args.max_val_samples = int(_option(args, "max_val_samples", sampling, "max_val_samples"))
    args.max_test_samples = int(_option(args, "max_test_samples", sampling, "max_test_samples"))
    args.epochs = int(_option(args, "epochs", training, "epochs"))
    args.batch_size = int(_option(args, "batch_size", training, "batch_size"))
    args.learning_rate = float(_option(args, "learning_rate", training, "learning_rate"))
    args.weight_decay = float(_option(args, "weight_decay", training, "weight_decay"))
    args.early_stopping_patience = int(
        _option(args, "early_stopping_patience", training, "early_stopping_patience")
    )
    args.device = args.device or training.get("device", "auto")
    args.num_workers = int(_option(args, "num_workers", training, "num_workers"))
    args.deterministic = bool(
        training.get("deterministic", False)
        if args.deterministic is None
        else args.deterministic
    )
    args.d_model = int(_option(args, "d_model", model, "d_model"))
    args.num_heads = int(_option(args, "num_heads", model, "num_heads"))
    args.num_layers = int(_option(args, "num_layers", model, "num_layers"))
    args.cross_layers = int(_option(args, "cross_layers", model, "cross_layers"))
    args.dim_feedforward = int(_option(args, "dim_feedforward", model, "dim_feedforward"))
    args.dropout = float(_option(args, "dropout", model, "dropout"))
    requested_ablations = (
        args.timestamp_ablations
        or evaluation.get("timestamp_ablations", ("real",))
    )
    # Configs antiguas siguen siendo válidas; metadata/resultados nuevos usan
    # nombres que no sugieren que se eliminaron gaps o ages derivados.
    args.timestamp_ablations = tuple(
        dict.fromkeys(canonical_ablation_name(name) for name in requested_ablations)
    )
    args.protocol_seed = int(sampling.get("protocol_seed", 2026))
    args.save_predictions = bool(
        evaluation.get("save_predictions", True) and not args.no_save_predictions
    )
    validate_options(args)
    return args


def validate_options(args: argparse.Namespace) -> None:
    if not args.horizons or any(not np.isfinite(value) or value <= 0 for value in args.horizons):
        raise ValueError("Todos los horizontes deben ser finitos y > 0.")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("Los horizontes físicos no pueden repetirse.")
    if not (0 < args.train_horizon_min < args.train_horizon_max):
        raise ValueError("train horizon debe cumplir 0 < min < max.")
    if args.queries_per_sample < 2:
        raise ValueError("queries_per_sample debe ser >= 2.")
    for name in (
        "history_duration",
        "max_history_events_univariate",
        "max_history_events_multivariate",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
        "batch_size",
        "d_model",
        "num_heads",
        "num_layers",
        "cross_layers",
        "dim_feedforward",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} debe ser > 0.")
    if args.epochs <= 0 and not args.validate_only:
        raise ValueError("epochs debe ser > 0 salvo con --validate-only.")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model debe ser divisible por num_heads.")
    if args.max_observation_rows_per_split is not None and args.max_observation_rows_per_split < 100:
        raise ValueError("max_observation_rows_per_split debe ser >= 100.")
    if "real" not in args.timestamp_ablations:
        raise ValueError(
            "timestamp_ablations debe incluir 'real' como referencia para el resumen."
        )


def discover_datasets(
    data_root: Path,
    kinds: Sequence[str],
    presets: Sequence[str] | None,
    limit_per_kind: int | None,
    dataset_ids: Sequence[str] | None = None,
) -> list[tuple[str, str, Path, Path]]:
    selected = set(presets) if presets else None
    selected_ids = set(dataset_ids) if dataset_ids else None
    discovered_ids: set[str] = set()
    result: list[tuple[str, str, Path, Path]] = []
    for kind in kinds:
        root = data_root / kind
        if not root.is_dir():
            raise FileNotFoundError(f"No existe {root}")
        found = []
        for preset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if selected is not None and preset_dir.name not in selected:
                continue
            for observations in sorted(preset_dir.glob("*/observations.parquet")):
                if (
                    selected_ids is not None
                    and observations.parent.name not in selected_ids
                ):
                    continue
                truth = observations.with_name("truth.parquet")
                if not truth.is_file():
                    raise FileNotFoundError(f"Falta {truth}")
                found.append((kind, preset_dir.name, observations, truth))
                discovered_ids.add(observations.parent.name)
        result.extend(found[:limit_per_kind] if limit_per_kind else found)
    if selected_ids is not None:
        missing = sorted(selected_ids - discovered_ids)
        if missing:
            raise FileNotFoundError(
                "No se encontraron los dataset IDs solicitados: " + ", ".join(missing)
            )
    if not result:
        raise FileNotFoundError("No se encontraron datasets físicos.")
    return result


def read_truth(truth_path: Path, kind: str) -> pd.DataFrame:
    columns = ["time", "clean_value", "split"]
    if kind == "multivariate":
        columns.extend(["channel_index", "channel"])
    truth = pd.read_parquet(truth_path, columns=columns)
    truth["time"] = truth["time"].astype(np.float64)
    if kind == "univariate":
        truth["channel_index"] = 0
        truth["channel"] = "x00"
    truth["channel_index"] = truth["channel_index"].astype(np.int64)
    return truth.sort_values(["time", "channel_index"], kind="stable").reset_index(drop=True)


def _time_bounds_for_split(
    truth: pd.DataFrame, split: str, history_duration: float
) -> tuple[float, float]:
    selected = truth[truth["split"].astype(str) == split]
    if selected.empty:
        raise ValueError(f"truth no contiene split '{split}'.")
    start = float(selected["time"].min()) - history_duration
    end = float(selected["time"].max())
    return max(float(truth["time"].min()), start), end


def read_observation_interval(
    observations_path: Path,
    kind: str,
    bounds: tuple[float, float],
    *,
    max_rows: int | None,
    seed: int,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Lee por batches y mantiene una muestra uniforme si se solicita cap."""
    columns = ["time", "value", "split", "event_index"]
    if kind == "multivariate":
        columns.append("channel_index")
    dataset = pads.dataset(observations_path, format="parquet")
    condition = (pads.field("time") >= bounds[0]) & (pads.field("time") <= bounds[1])
    scanner = dataset.scanner(columns=columns, filter=condition, batch_size=batch_size)
    generator = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    reservoir: pd.DataFrame | None = None
    for record_batch in scanner.to_batches():
        frame = record_batch.to_pandas()
        if frame.empty:
            continue
        if max_rows is None:
            chunks.append(frame)
            continue
        frame["_priority"] = generator.random(len(frame))
        combined = frame if reservoir is None else pd.concat([reservoir, frame], ignore_index=True)
        if len(combined) > max_rows:
            selected = np.argpartition(
                combined["_priority"].to_numpy(), max_rows - 1
            )[:max_rows]
            reservoir = combined.iloc[selected].copy()
        else:
            reservoir = combined

    if max_rows is None:
        if not chunks:
            raise ValueError(f"No hay observaciones en intervalo {bounds}.")
        result = pd.concat(chunks, ignore_index=True)
    else:
        if reservoir is None or reservoir.empty:
            raise ValueError(f"No hay observaciones en intervalo {bounds}.")
        result = reservoir.drop(columns="_priority")
    sort_columns = ["time", "event_index"]
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def fit_channel_scaler(
    train_observations: pd.DataFrame, kind: str, n_channels: int
) -> ChannelScaler:
    selected = train_observations[
        train_observations["split"].astype(str) == "train"
    ]
    if selected.empty:
        raise ValueError("No hay observaciones train para ajustar el scaler.")
    if kind == "univariate":
        mean = np.asarray([selected["value"].mean()], dtype=np.float64)
        std = np.asarray([selected["value"].std(ddof=0)], dtype=np.float64)
    else:
        grouped = selected.groupby("channel_index", observed=True)["value"]
        mean = grouped.mean().reindex(range(n_channels)).to_numpy(dtype=np.float64)
        std = grouped.std(ddof=0).reindex(range(n_channels)).to_numpy(dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("El scaler no pudo estimarse para todos los canales.")
    std[std < 1e-8] = 1.0
    return ChannelScaler(mean=mean, std=std)


def truth_arrays(
    truth: pd.DataFrame,
    split: str,
    n_channels: int,
    scaler: ChannelScaler,
) -> tuple[np.ndarray, np.ndarray]:
    selected = truth[truth["split"].astype(str) == split]
    if n_channels == 1:
        return (
            selected["time"].to_numpy(dtype=np.float64),
            scaler.transform_matrix(
                selected[["clean_value"]].to_numpy(dtype=np.float32)
            ),
        )
    matrix = (
        selected.pivot(index="time", columns="channel_index", values="clean_value")
        .reindex(columns=range(n_channels))
        .sort_index()
    )
    if matrix.isna().any().any():
        raise ValueError("truth multivariado debe tener todos los canales por timestamp.")
    return (
        matrix.index.to_numpy(dtype=np.float64),
        scaler.transform_matrix(matrix.to_numpy(dtype=np.float32)),
    )


def forecast_origins_for_split(truth: pd.DataFrame, split: str) -> np.ndarray:
    """Orígenes en una grilla física, independientes de la densidad de eventos."""
    selected = truth[truth["split"].astype(str) == split]
    origins = np.sort(selected["time"].unique().astype(np.float64, copy=False))
    if origins.size == 0:
        raise ValueError(f"truth no contiene orígenes para split '{split}'.")
    return origins


def build_split_dataset(
    observations: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    kind: str,
    split: str,
    n_channels: int,
    scaler: ChannelScaler,
    horizons: Sequence[float],
    history_duration: float,
    max_history_events: int,
    history_subsampling: str,
    sampling_seed: int,
    train_horizon_range: tuple[float, float] | None = None,
    train_horizon_sampling: str = "log_uniform",
    queries_per_sample: int | None = None,
    compute_history_diagnostics: bool = True,
) -> Dataset:
    use_continuous_train_horizons = split == "train" and train_horizon_range is not None
    num_queries = int(queries_per_sample or len(horizons))
    config = TimeSeriesDatasetConfig(
        history_length=max_history_events,
        stride=1,
        target_horizon_choices=(
            None if use_continuous_train_horizons else list(horizons)
        ),
        target_horizon_min=(
            float(train_horizon_range[0]) if use_continuous_train_horizons else None
        ),
        target_horizon_max=(
            float(train_horizon_range[1]) if use_continuous_train_horizons else None
        ),
        target_horizon_sampling=train_horizon_sampling,
        num_targets=num_queries if use_continuous_train_horizons else len(horizons),
        target_match_mode="linear",
        randomize_query_order=True,
        sampling_seed=sampling_seed,
        history_duration=history_duration,
        max_history_events=max_history_events,
        history_subsampling=history_subsampling,
        compute_history_diagnostics=compute_history_diagnostics,
        # Este runner reutiliza el mismo PreparedPhysicalData en todas las
        # épocas/modelos/seeds. Sólo se cachea la selección histórica fija;
        # queries y horizons continúan dependiendo de set_epoch.
        cache_deterministic_history=history_subsampling
        in {"uniform_time", "uniform_index"},
    )
    target_times, target_values = truth_arrays(
        truth, split, n_channels, scaler
    )
    forecast_origins = forecast_origins_for_split(truth, split)
    timestamps = observations["time"].to_numpy(dtype=np.float64)
    if kind == "univariate":
        values = scaler.transform_long(
            observations["value"].to_numpy(dtype=np.float32)
        ).reshape(-1, 1)
        return TimeSeriesDataset(
            values,
            timestamps,
            config,
            input_dim=1,
            output_dim=1,
            targets=target_values,
            target_timestamps=target_times,
            forecast_origin_timestamps=forecast_origins,
            sequence_builder=SequenceBuilder(input_dim=1, relative_timestamps=True),
        )

    channel_indices = observations["channel_index"].to_numpy(dtype=np.int64)
    scaled = scaler.transform_long(
        observations["value"].to_numpy(dtype=np.float32), channel_indices
    )
    builder = SequenceBuilder(
        input_dim=1,
        use_sensor_ids=True,
        num_sensors=n_channels,
        num_target_tokens=n_channels,
        target_sensor_ids=list(range(n_channels)),
        relative_timestamps=True,
    )
    return EventTimeSeriesDataset(
        scaled.reshape(-1, 1),
        timestamps,
        target_values,
        config,
        input_dim=n_channels,
        output_dim=n_channels,
        target_timestamps=target_times,
        forecast_origin_timestamps=forecast_origins,
        sequence_builder=builder,
        event_sensor_ids=channel_indices,
    )


def evenly_spaced_subset(dataset: Dataset, max_samples: int) -> Dataset:
    if len(dataset) <= max_samples:
        return dataset
    indices = np.linspace(0, len(dataset) - 1, max_samples).round().astype(np.int64)
    indices = np.unique(indices)
    return Subset(dataset, indices.tolist())


def prepare_physical_data(
    dataset_spec: tuple[str, str, Path, Path],
    *,
    horizons: Sequence[float],
    history_duration: float,
    max_history_events: int,
    history_subsampling: str,
    max_samples: Mapping[str, int],
    max_observation_rows_per_split: int | None,
    protocol_seed: int,
    train_horizon_range: tuple[float, float] | None = None,
    train_horizon_sampling: str = "log_uniform",
    queries_per_sample: int | None = None,
) -> PreparedPhysicalData:
    kind, preset, observations_path, truth_path = dataset_spec
    truth = read_truth(truth_path, kind)
    n_channels = int(truth["channel_index"].max()) + 1
    observation_frames: dict[str, pd.DataFrame] = {}
    for split_index, split in enumerate(SPLIT_NAMES):
        bounds = _time_bounds_for_split(truth, split, history_duration)
        observation_frames[split] = read_observation_interval(
            observations_path,
            kind,
            bounds,
            max_rows=max_observation_rows_per_split,
            seed=protocol_seed + split_index,
        )
    scaler = fit_channel_scaler(observation_frames["train"], kind, n_channels)
    datasets = {}
    forecast_origin_audit: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(SPLIT_NAMES):
        base = build_split_dataset(
            observation_frames[split],
            truth,
            kind=kind,
            split=split,
            n_channels=n_channels,
            scaler=scaler,
            horizons=horizons,
            history_duration=history_duration,
            max_history_events=max_history_events,
            history_subsampling=history_subsampling,
            sampling_seed=protocol_seed + split_index * 100,
            train_horizon_range=train_horizon_range,
            train_horizon_sampling=train_horizon_sampling,
            queries_per_sample=queries_per_sample,
            # Sólo test genera estratos detallados. Train/validation evitan
            # diffs y medianas sobre ventanas con cientos de miles de eventos.
            compute_history_diagnostics=split == "test",
        )
        selected = evenly_spaced_subset(base, max_samples[split])
        audit = {
            **base.forecast_origin_audit,
            "discarded_by_cause": dict(
                base.forecast_origin_audit["discarded_by_cause"]
            ),
            "cause_order": list(base.forecast_origin_audit["cause_order"]),
            "selected_examples_after_cap": len(selected),
        }
        forecast_origin_audit[split] = audit
        datasets[split] = selected
    return PreparedPhysicalData(
        train=datasets["train"],
        validation=datasets["validation"],
        test=datasets["test"],
        scaler=scaler,
        kind=kind,
        preset=preset,
        dataset_id=observations_path.parent.name,
        n_channels=n_channels,
        max_history_events=max_history_events,
        row_counts={key: len(value) for key, value in observation_frames.items()},
        forecast_origin_audit=forecast_origin_audit,
    )


def make_loaders(
    data: PreparedPhysicalData,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    collate = PhysicalCollate()
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": device.startswith("cuda"),
    }
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(data.train, shuffle=True, generator=generator, **common)
    validation = DataLoader(data.validation, shuffle=False, **common)
    test = DataLoader(data.test, shuffle=False, **common)
    return train, validation, test


def base_model_config(
    data: PreparedPhysicalData,
    args: argparse.Namespace,
    *,
    prediction_head: str = "point",
) -> TimeSeriesTransformerConfig:
    return TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=data.n_channels,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_num_layers=args.cross_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="gelu",
        time_scale=1.0,
        time_transform="log1p",
        time_encoding_mode="sinusoidal",
        use_sensor_embedding=data.kind == "multivariate",
        num_sensors=data.n_channels if data.kind == "multivariate" else 0,
        prediction_head=prediction_head,
    )


def query_variant_config(model_name: str) -> QueryCrossAttentionConfig:
    common = {"temporal_feature_dim": 1}
    if model_name in {"QueryCross", "QueryCross-Gaussian"}:
        return QueryCrossAttentionConfig(**common)
    if model_name == "NoTime":
        return QueryCrossAttentionConfig(
            **common,
            derive_temporal_features=False,
            use_relative_time_bias=False,
            use_temporal_film=False,
            use_query_horizon=False,
            use_history_time_encoding=False,
            use_ctssm=False,
            mask_history_after_query=False,
        )
    if model_name == "QueryOnly":
        return QueryCrossAttentionConfig(
            **common,
            derive_temporal_features=False,
            use_relative_time_bias=False,
            use_temporal_film=False,
            use_query_horizon=True,
            use_history_time_encoding=False,
            use_ctssm=False,
            mask_history_after_query=False,
        )
    if model_name == "CTSSM":
        return QueryCrossAttentionConfig(**common, use_ctssm=True)
    raise ValueError(f"{model_name} no es una variante neural.")


def basis_variant_config(model_name: str) -> ContinuousBasisDecoderConfig:
    if model_name not in {
        "BasisDecoder",
        "BasisDecoder-Gaussian",
        "BasisDecoder-CTSSM",
    }:
        raise ValueError(f"{model_name} no es una variante BasisDecoder.")
    return ContinuousBasisDecoderConfig(
        temporal_feature_dim=1,
        use_ctssm=model_name == "BasisDecoder-CTSSM",
    )


def is_gaussian_model(model_name: str) -> bool:
    return model_name in {"QueryCross-Gaussian", "BasisDecoder-Gaussian"}


def build_model(
    model_name: str, data: PreparedPhysicalData, args: argparse.Namespace
) -> nn.Module | None:
    if model_name == "Persistence":
        return None
    prediction_head = "gaussian" if is_gaussian_model(model_name) else "point"
    if model_name.startswith("BasisDecoder"):
        return TimeSeriesContinuousBasisDecoder(
            base_model_config(data, args, prediction_head=prediction_head),
            basis_variant_config(model_name),
        )
    return TimeSeriesQueryCrossAttention(
        base_model_config(data, args, prediction_head=prediction_head),
        query_variant_config(model_name),
    )


def count_parameters(model: nn.Module | None) -> int:
    if model is None:
        return 0
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_paired_reference_parameters(
    model: nn.Module,
    reference_state: Mapping[str, torch.Tensor],
) -> int:
    """Carga por nombre/shape los pesos comunes de la variante completa.

    Crear módulos opcionales cambia el orden del RNG aunque se use la misma
    seed. Esta carga explícita hace realmente pareadas las partes compartidas
    entre Full, NoTime, QueryOnly, CTSSM y Gaussian.
    """
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in reference_state.items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def persistence_predictions(
    batch: Mapping[str, torch.Tensor], output_dim: int
) -> torch.Tensor:
    values = batch["input_values"]
    target_mask = batch["is_target_mask"]
    padding = batch.get("padding_mask", torch.zeros_like(target_mask))
    history_valid = (~target_mask) & (~padding)
    target_shape = batch["target_values"].shape
    batch_size, horizons, _ = target_shape
    sensor_ids = batch.get("input_sensor_ids")
    predictions = values.new_zeros(target_shape)
    positions = torch.arange(values.shape[1], device=values.device).view(1, -1)

    if sensor_ids is None:
        last = positions.masked_fill(~history_valid, -1).amax(dim=1)
        if torch.any(last < 0):
            raise ValueError("Cada ejemplo requiere historia para Persistence.")
        last_values = values[
            torch.arange(batch_size, device=values.device), last, :output_dim
        ]
        return last_values.unsqueeze(1).expand(-1, horizons, -1).clone()

    for sensor in range(output_dim):
        matches = history_valid & (sensor_ids == sensor)
        last = positions.masked_fill(~matches, -1).amax(dim=1)
        available = last >= 0
        safe = last.clamp_min(0)
        sensor_value = values[
            torch.arange(batch_size, device=values.device), safe, 0
        ]
        sensor_value = torch.where(available, sensor_value, torch.zeros_like(sensor_value))
        predictions[:, :, sensor] = sensor_value.unsqueeze(1)
    return predictions


def ablate_batch_timestamps(
    batch: Mapping[str, torch.Tensor],
    variant: str,
    *,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    if variant not in TIMESTAMP_ABLATIONS:
        raise ValueError(f"Ablación desconocida: {variant}")
    variant = canonical_ablation_name(variant)
    transformed = batch["input_timestamps"].clone()
    timestamp_variant = variant.removesuffix("_no_count_feature")
    if timestamp_variant == "real":
        return transformed
    target_mask = batch["is_target_mask"]
    padding = batch.get("padding_mask", torch.zeros_like(target_mask))
    generator = np.random.default_rng(seed)
    for sample_index in range(transformed.shape[0]):
        history_positions = torch.where(~target_mask[sample_index] & ~padding[sample_index])[0]
        target_positions = torch.where(target_mask[sample_index] & ~padding[sample_index])[0]
        history = transformed[sample_index, history_positions]
        queries = transformed[sample_index, target_positions]
        if timestamp_variant == "all_equal":
            transformed[sample_index, history_positions] = 0.0
            transformed[sample_index, target_positions] = 0.0
        elif timestamp_variant == "query_only":
            transformed[sample_index, history_positions] = history[-1]
        elif timestamp_variant == "history_only":
            transformed[sample_index, target_positions] = history[-1]
        elif timestamp_variant == "regular_grid":
            transformed[sample_index, history_positions] = torch.linspace(
                float(history[0]), float(history[-1]), len(history), dtype=history.dtype
            )
        elif timestamp_variant == "permuted_gaps":
            if len(history) > 1:
                gaps = torch.diff(history).cpu().numpy()
                rebuilt = np.concatenate(([0.0], np.cumsum(generator.permutation(gaps))))
                rebuilt += float(history[0])
                transformed[sample_index, history_positions] = torch.as_tensor(
                    rebuilt, dtype=history.dtype
                )
        else:  # ordinal
            raise AssertionError("La ablación ordinal no forma parte del runner neural.")
        if queries.numel() % output_dim != 0:
            raise ValueError("Cantidad de target tokens incompatible con output_dim.")
    return transformed


def _forward_model(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    timestamps: torch.Tensor,
    *,
    zero_temporal_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    gaussian = (
        str(getattr(getattr(model, "config", None), "prediction_head", "point")).lower()
        == "gaussian"
    )
    kwargs: dict[str, Any] = {
        "input_values": batch["input_values"].to(device),
        "input_timestamps": timestamps.to(device),
        "is_target_mask": batch["is_target_mask"].to(device),
        "padding_mask": batch.get("padding_mask", None),
        "lengths": batch.get("lengths", None),
        "input_sensor_ids": batch.get("input_sensor_ids", None),
        "temporal_features": batch.get("temporal_features", None),
        "return_dict": gaussian,
    }
    if zero_temporal_features and kwargs["temporal_features"] is not None:
        kwargs["temporal_features"] = torch.zeros_like(
            kwargs["temporal_features"]
        )
    for key in ("padding_mask", "lengths", "input_sensor_ids", "temporal_features"):
        if kwargs[key] is not None:
            kwargs[key] = kwargs[key].to(device)
    prediction = model(**kwargs)
    log_scale = None
    if isinstance(prediction, dict):
        log_scale = prediction.get("log_scale")
        prediction = prediction["preds"]
    target = batch["target_values"]
    if prediction.ndim == 2 and target.ndim == 3:
        prediction = prediction.unsqueeze(1)
        if log_scale is not None:
            log_scale = log_scale.unsqueeze(1)
    return (
        prediction.detach().cpu(),
        log_scale.detach().cpu() if log_scale is not None else None,
    )


def _gaussian_diagnostics(
    prediction_z: float,
    target_z: float,
    log_scale_z: float,
    channel_std: float,
) -> dict[str, float]:
    scale_z = math.exp(float(np.clip(log_scale_z, -20.0, 20.0)))
    standardized_error = (target_z - prediction_z) / scale_z
    nll_z = (
        0.5 * math.log(2.0 * math.pi)
        + math.log(scale_z)
        + 0.5 * standardized_error**2
    )
    phi = math.exp(-0.5 * standardized_error**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(standardized_error / math.sqrt(2.0)))
    crps_z = scale_z * (
        standardized_error * (2.0 * cdf - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )
    return {
        "log_scale_z": log_scale_z,
        "scale_z": scale_z,
        "nll_z": nll_z,
        "crps_z": crps_z,
        "coverage_90": float(abs(standardized_error) <= 1.6448536269514722),
        "coverage_95": float(abs(standardized_error) <= 1.959963984540054),
        "scale": scale_z * channel_std,
        "nll": nll_z + math.log(channel_std),
        "crps": crps_z * channel_std,
    }


def _history_statistics(
    batch: Mapping[str, torch.Tensor],
    sample_index: int,
    history_duration: float,
    n_channels: int,
) -> dict[str, torch.Tensor | float]:
    """Obtiene diagnósticos por sensor previos al subsampling.

    ``past_sensor_*`` es la fuente exacta del protocolo físico. El fallback
    mantiene compatibilidad con batches antiguos, pero sólo puede reconstruir
    lo visible en los tokens retenidos.
    """
    valid = (~batch["is_target_mask"][sample_index]) & (
        ~batch.get("padding_mask", torch.zeros_like(batch["is_target_mask"]))[sample_index]
    )
    timestamps = batch["input_timestamps"][sample_index, valid].to(torch.float64)
    represented_counts = batch["input_observation_counts"][sample_index, valid]
    gaps = torch.diff(timestamps)
    original_count = float(
        batch["past_original_observation_count"][sample_index].item()
        if "past_original_observation_count" in batch
        else represented_counts.sum().item()
    )
    original_max_gap = float(
        batch["past_original_max_gap"][sample_index].item()
        if "past_original_max_gap" in batch
        else (gaps.max().item() if gaps.numel() else 0.0)
    )
    original_median_gap = float(
        batch["past_original_median_gap"][sample_index].item()
        if "past_original_median_gap" in batch
        else (gaps.median().item() if gaps.numel() else 0.0)
    )
    last_observation_age = float(
        batch.get(
            "last_observation_age",
            torch.zeros(batch["input_timestamps"].shape[0], dtype=torch.float64),
        )[sample_index].item()
    )

    sensor_ids = batch.get("input_sensor_ids")
    if sensor_ids is None:
        sampled_counts = torch.full(
            (n_channels,), float(valid.sum().item()), dtype=torch.float64
        )
    else:
        history_sensor_ids = sensor_ids[sample_index, valid].to(torch.long)
        sampled_counts = torch.bincount(
            history_sensor_ids, minlength=n_channels
        ).to(torch.float64)[:n_channels]

    def vector_metadata(name: str, fallback: torch.Tensor) -> torch.Tensor:
        if name not in batch:
            return fallback
        value = batch[name][sample_index].to(torch.float64)
        if value.shape != (n_channels,):
            raise ValueError(
                f"{name} debe tener shape [{n_channels}] por ejemplo; "
                f"se obtuvo {tuple(value.shape)}."
            )
        return value

    fallback_counts = (
        torch.full((n_channels,), original_count, dtype=torch.float64)
        if sensor_ids is None
        else sampled_counts
    )
    return {
        "sampled_counts": sampled_counts,
        "sensor_counts": vector_metadata(
            "past_sensor_observation_count", fallback_counts
        ),
        "sensor_max_gaps": vector_metadata(
            "past_sensor_max_gap",
            torch.full((n_channels,), original_max_gap, dtype=torch.float64),
        ),
        "sensor_median_gaps": vector_metadata(
            "past_sensor_median_gap",
            torch.full((n_channels,), original_median_gap, dtype=torch.float64),
        ),
        "sensor_last_ages": vector_metadata(
            "sensor_last_observation_age",
            torch.full((n_channels,), last_observation_age, dtype=torch.float64),
        ),
        "global_count": original_count,
        "global_max_gap": original_max_gap,
        "global_median_gap": original_median_gap,
        "global_last_age": last_observation_age,
    }


@torch.inference_mode()
def evaluate_prediction_rows(
    model: nn.Module | None,
    loader: DataLoader,
    data: PreparedPhysicalData,
    *,
    model_name: str,
    seed: int,
    device: str,
    timestamp_ablations: Sequence[str],
    history_duration: float,
) -> pd.DataFrame:
    device_object = torch.device(device)
    if model is not None:
        model.eval()
    records: list[dict[str, float | int | str]] = []
    example_offset = 0
    for batch_index, batch in enumerate(loader):
        targets_z = batch["target_values"]
        masks = batch.get("target_loss_mask", torch.ones_like(targets_z)) > 0
        horizons = batch["requested_target_horizons"].to(torch.float64)
        absolute_queries = batch["absolute_target_timestamps"].to(torch.float64)
        statistics = [
            _history_statistics(batch, index, history_duration, data.n_channels)
            for index in range(targets_z.shape[0])
        ]
        for ablation_index, ablation in enumerate(timestamp_ablations):
            canonical_ablation = canonical_ablation_name(ablation)
            timestamps = ablate_batch_timestamps(
                batch,
                canonical_ablation,
                output_dim=data.n_channels,
                seed=seed + batch_index * 1000 + ablation_index,
            )
            if model is None:
                predictions_z = persistence_predictions(batch, data.n_channels)
                log_scales_z = None
            else:
                predictions_z, log_scales_z = _forward_model(
                    model,
                    batch,
                    device_object,
                    timestamps,
                    zero_temporal_features=canonical_ablation.endswith(
                        "_no_count_feature"
                    ),
                )
            if predictions_z.shape != targets_z.shape:
                raise ValueError(
                    f"Predicción {tuple(predictions_z.shape)} != target {tuple(targets_z.shape)}"
                )
            for sample in range(targets_z.shape[0]):
                sample_statistics = statistics[sample]
                for horizon_index in range(targets_z.shape[1]):
                    for channel in range(data.n_channels):
                        if not bool(masks[sample, horizon_index, channel]):
                            continue
                        prediction_z = float(predictions_z[sample, horizon_index, channel])
                        target_z = float(targets_z[sample, horizon_index, channel])
                        prediction = float(
                            data.scaler.inverse_channel(np.asarray(prediction_z), channel)
                        )
                        target = float(
                            data.scaler.inverse_channel(np.asarray(target_z), channel)
                        )
                        sensor_count = float(
                            sample_statistics["sensor_counts"][channel]
                        )
                        sampled_sensor_count = int(
                            sample_statistics["sampled_counts"][channel]
                        )
                        max_gap = float(
                            sample_statistics["sensor_max_gaps"][channel]
                        )
                        median_gap = float(
                            sample_statistics["sensor_median_gaps"][channel]
                        )
                        last_observation_age = float(
                            sample_statistics["sensor_last_ages"][channel]
                        )
                        distribution = (
                            _gaussian_diagnostics(
                                prediction_z,
                                target_z,
                                float(log_scales_z[sample, horizon_index, channel]),
                                float(data.scaler.std[channel]),
                            )
                            if log_scales_z is not None
                            else {
                                "log_scale_z": math.nan,
                                "scale_z": math.nan,
                                "nll_z": math.nan,
                                "crps_z": math.nan,
                                "coverage_90": math.nan,
                                "coverage_95": math.nan,
                                "scale": math.nan,
                                "nll": math.nan,
                                "crps": math.nan,
                            }
                        )
                        records.append(
                            ({
                                "Dataset_ID": data.dataset_id,
                                "Kind": data.kind,
                                "Preset": data.preset,
                                "Seed": seed,
                                "Model": model_name,
                                "Ablation": canonical_ablation,
                                "Example": example_offset + sample,
                                "Target_Index": horizon_index,
                                "Horizon": float(horizons[sample, horizon_index]),
                                "Query_Time": float(absolute_queries[sample, horizon_index]),
                                "Channel": channel,
                                "history_events": int(sensor_count),
                                "sampled_history_events": sampled_sensor_count,
                                "density": sensor_count / history_duration,
                                "max_gap": max_gap,
                                "median_gap": median_gap,
                                "last_observation_age": last_observation_age,
                                "global_history_events": int(
                                    sample_statistics["global_count"]
                                ),
                                "global_density": float(
                                    sample_statistics["global_count"]
                                ) / history_duration,
                                "global_max_gap": float(
                                    sample_statistics["global_max_gap"]
                                ),
                                "global_median_gap": float(
                                    sample_statistics["global_median_gap"]
                                ),
                                "global_last_observation_age": float(
                                    sample_statistics["global_last_age"]
                                ),
                                "prediction_z": prediction_z,
                                "target_z": target_z,
                                "prediction": prediction,
                                "target": target,
                            } | distribution)
                        )
        example_offset += targets_z.shape[0]
    return pd.DataFrame.from_records(records)


def trainer_metrics_from_prediction_rows(rows: pd.DataFrame) -> dict[str, float]:
    """Reconstruye las métricas de :class:`Trainer` sin otro forward de test.

    El loop común ya conserva cada predicción válida de la ablación ``real``.
    Además de las métricas planas, ``Target_Index`` permite recuperar los
    estratos por posición de query aun cuando sus horizontes estén permutados.
    """
    selected = rows.loc[rows["Ablation"] == "real"]
    if selected.empty:
        raise ValueError("No hay predicciones de la ablación 'real'.")
    required = {
        "Example",
        "Target_Index",
        "Channel",
        "prediction_z",
        "target_z",
        "log_scale_z",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(
            "Faltan columnas para derivar métricas Trainer: "
            + ", ".join(sorted(missing))
        )
    key_columns = ["Example", "Target_Index", "Channel"]
    if selected.duplicated(key_columns).any():
        raise ValueError("Hay predicciones reales duplicadas para una misma celda.")

    flat_predictions = torch.as_tensor(
        selected["prediction_z"].to_numpy(), dtype=torch.float32
    ).reshape(-1, 1)
    flat_targets = torch.as_tensor(
        selected["target_z"].to_numpy(), dtype=torch.float32
    ).reshape(-1, 1)
    metrics = compute_regression_metrics(
        flat_predictions, flat_targets, prefix="test_"
    )

    example_codes, examples = pd.factorize(selected["Example"], sort=True)
    target_indices = selected["Target_Index"].to_numpy(dtype=np.int64)
    channels = selected["Channel"].to_numpy(dtype=np.int64)
    if (target_indices < 0).any() or (channels < 0).any():
        raise ValueError("Target_Index y Channel deben ser no negativos.")
    shape = (
        len(examples),
        int(target_indices.max()) + 1,
        int(channels.max()) + 1,
    )
    structured_predictions = torch.zeros(shape, dtype=torch.float32)
    structured_targets = torch.zeros(shape, dtype=torch.float32)
    structured_mask = torch.zeros(shape, dtype=torch.float32)
    index = (
        torch.as_tensor(example_codes, dtype=torch.long),
        torch.as_tensor(target_indices, dtype=torch.long),
        torch.as_tensor(channels, dtype=torch.long),
    )
    structured_predictions[index] = flat_predictions[:, 0]
    structured_targets[index] = flat_targets[:, 0]
    structured_mask[index] = 1.0
    metrics.update(
        compute_structured_regression_metrics(
            structured_predictions,
            structured_targets,
            structured_mask,
            prefix="test_",
        )
    )

    probabilistic = selected["log_scale_z"].notna()
    if probabilistic.any():
        if not probabilistic.all():
            raise ValueError(
                "La ablación real mezcla predicciones puntuales y gaussianas."
            )
        log_scales = torch.as_tensor(
            selected["log_scale_z"].to_numpy(), dtype=torch.float32
        ).reshape(-1, 1)
        metrics.update(
            compute_gaussian_metrics(
                flat_predictions,
                flat_targets,
                log_scales,
                prefix="test_",
            )
        )
        # Trainer._compute_loss usa el log-scale crudo; la métrica NLL aplica
        # clamps sólo para robustez del reporte. Conservamos ambas semánticas.
        raw_nll = (
            log_scales
            + 0.5
            * (flat_targets - flat_predictions).square()
            * torch.exp(-2.0 * log_scales)
            + 0.5 * math.log(2.0 * math.pi)
        )
        metrics["test_loss"] = raw_nll.mean().item()
    else:
        metrics["test_loss"] = metrics["test_mse"]
    return metrics


def _metric_record(group: pd.DataFrame) -> dict[str, float | int]:
    error_z = group["prediction_z"].to_numpy() - group["target_z"].to_numpy()
    error = group["prediction"].to_numpy() - group["target"].to_numpy()
    result: dict[str, float | int] = {
        "n": int(len(group)),
        "rmse_z": float(np.sqrt(np.mean(np.square(error_z)))),
        "mae_z": float(np.mean(np.abs(error_z))),
        "bias_z": float(np.mean(error_z)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }
    probabilistic = group["nll_z"].notna()
    if probabilistic.any():
        selected = group.loc[probabilistic]
        result.update(
            {
                "nll_z": float(selected["nll_z"].mean()),
                "crps_z": float(selected["crps_z"].mean()),
                "coverage_90": float(selected["coverage_90"].mean()),
                "coverage_95": float(selected["coverage_95"].mean()),
                "mean_scale_z": float(selected["scale_z"].mean()),
                "nll": float(selected["nll"].mean()),
                "crps": float(selected["crps"].mean()),
                "mean_scale": float(selected["scale"].mean()),
            }
        )
    else:
        result.update(
            {
                "nll_z": math.nan,
                "crps_z": math.nan,
                "coverage_90": math.nan,
                "coverage_95": math.nan,
                "mean_scale_z": math.nan,
                "nll": math.nan,
                "crps": math.nan,
                "mean_scale": math.nan,
            }
        )
    return result


def summarize_predictions(rows: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    keys = ["Dataset_ID", "Kind", "Preset", "Seed", "Model", "Ablation"]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True):
        base = dict(zip(keys, key))
        records.append(base | {"Scope": "overall", "Level": "all"} | _metric_record(group))
        for horizon, selected in group.groupby("Horizon", sort=True):
            records.append(base | {"Scope": "horizon", "Level": f"{horizon:g}"} | _metric_record(selected))
        for channel, selected in group.groupby("Channel", sort=True):
            records.append(base | {"Scope": "channel", "Level": str(channel)} | _metric_record(selected))
        for column in ("density", "max_gap", "last_observation_age"):
            if group[column].nunique(dropna=True) < 2:
                records.append(
                    base
                    | {"Scope": f"{column}_bin", "Level": "constant"}
                    | _metric_record(group)
                )
                continue
            try:
                labels = pd.qcut(group[column], q=bins, duplicates="drop")
            except ValueError:
                continue
            for interval, selected in group.groupby(labels, observed=True, sort=True):
                records.append(
                    base
                    | {"Scope": f"{column}_bin", "Level": str(interval)}
                    | _metric_record(selected)
                )
        # Los estratos globales anteriores pueden confundir identidad de canal
        # con irregularidad (p.ej. un sensor siempre más denso que otro). Estos
        # contrastes calculan los cuantiles de forma independiente dentro de
        # cada canal y codifican ambos componentes en Level.
        channel_strata = (
            ("density", "channel_density_bin"),
            ("max_gap", "channel_max_gap_bin"),
            ("last_observation_age", "channel_last_age_bin"),
        )
        for column, scope in channel_strata:
            for channel, channel_group in group.groupby("Channel", sort=True):
                level_prefix = f"channel={int(channel)}|"
                if channel_group[column].nunique(dropna=True) < 2:
                    records.append(
                        base
                        | {"Scope": scope, "Level": level_prefix + "constant"}
                        | _metric_record(channel_group)
                    )
                    continue
                try:
                    labels = pd.qcut(
                        channel_group[column], q=bins, duplicates="drop"
                    )
                except ValueError:
                    continue
                for interval, selected in channel_group.groupby(
                    labels, observed=True, sort=True
                ):
                    records.append(
                        base
                        | {
                            "Scope": scope,
                            "Level": level_prefix + str(interval),
                        }
                        | _metric_record(selected)
                    )
    return pd.DataFrame.from_records(records)


def restore_best_in_memory(trainer: Trainer) -> None:
    state = getattr(trainer, "_best_model_state_in_memory", None)
    if state is not None:
        trainer.model.load_state_dict(state)


def checkpoint_selection_metric(model_name: str) -> str | None:
    """Métrica de validación congelada para seleccionar pesos, sin mirar test."""
    if model_name == "Persistence":
        return None
    return "val_nll" if is_gaussian_model(model_name) else "val_rmse"


def training_config(
    args: argparse.Namespace,
    checkpoint_dir: Path | None,
    *,
    model_name: str = "QueryCross",
) -> TrainingConfig:
    optimizer = OptimizerConfig(
        optimizer_name="adamw",
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_name="cosine",
        scheduler_T_max=max(1, args.epochs),
    )
    return TrainingConfig(
        num_epochs=max(1, args.epochs),
        device=args.device,
        loss_name="mse",
        optimizer_config=optimizer,
        grad_clip_norm=1.0,
        log_every_n_steps=0,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        save_best_on=checkpoint_selection_metric(model_name) or "val_rmse",
        early_stopping_patience=args.early_stopping_patience,
        restore_best_weights=True,
        use_amp=args.device.startswith("cuda"),
        enable_cuda_runtime_optimizations=not bool(
            getattr(args, "deterministic", False)
        ),
    )


def run_model(
    model_name: str,
    data: PreparedPhysicalData,
    args: argparse.Namespace,
    *,
    seed: int,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    deterministic = bool(getattr(args, "deterministic", False))
    if deterministic:
        # Debe definirse antes de la primera operación CUDA que use cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    elif hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(deterministic)
    set_global_seed(seed, deterministic=deterministic)
    train_loader, val_loader, test_loader = make_loaders(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        device=args.device,
    )
    paired_parameter_tensors = 0
    if model_name == "Persistence":
        model = None
    else:
        # Construir siempre la misma referencia antes de cada ablación evita
        # que módulos opcionales desplacen la secuencia de inicialización.
        set_global_seed(seed, deterministic=deterministic)
        reference = build_model("QueryCross", data, args)
        if reference is None:
            raise RuntimeError("QueryCross de referencia no pudo construirse.")
        reference_state = {
            name: value.detach().cpu().clone()
            for name, value in reference.state_dict().items()
        }
        set_global_seed(seed, deterministic=deterministic)
        model = build_model(model_name, data, args)
        if model is None:
            raise RuntimeError(f"No se pudo construir {model_name}.")
        paired_parameter_tensors = load_paired_reference_parameters(
            model, reference_state
        )
        del reference, reference_state
    checkpoint_dir = None
    if model is not None and not args.no_checkpoints and not args.validate_only:
        checkpoint_dir = run_dir / "checkpoints"
    history: dict[str, Any] = {}
    trainer_metrics: dict[str, float] = {}
    train_seconds = 0.0
    prediction_eval_seconds = 0.0
    metric_derivation_seconds = 0.0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    if model is not None:
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            config=training_config(args, checkpoint_dir, model_name=model_name),
        )
        if not args.validate_only:
            start = time.perf_counter()
            history = trainer.fit()
            train_seconds = time.perf_counter() - start
            restore_best_in_memory(trainer)

    checkpoint_metric = checkpoint_selection_metric(model_name)
    history_payload = {
        "schema_version": 1,
        "model": model_name,
        "seed": int(seed),
        "deterministic": deterministic,
        "checkpoint_selection": checkpoint_metric,
        "best_epoch": (
            int(trainer.best_epoch)
            if model is not None and trainer.best_epoch is not None
            else None
        ),
        "best_metric_value": (
            float(trainer.best_metric_value)
            if model is not None and trainer.best_metric_value is not None
            else None
        ),
        "history": history,
    }
    atomic_write_json(run_dir / "history.json", history_payload)

    evaluation_start = time.perf_counter()
    rows = evaluate_prediction_rows(
        model,
        test_loader,
        data,
        model_name=model_name,
        seed=seed,
        device=args.device,
        timestamp_ablations=args.timestamp_ablations,
        history_duration=args.history_duration,
    )
    prediction_eval_seconds = time.perf_counter() - evaluation_start
    metric_start = time.perf_counter()
    metrics = summarize_predictions(rows)
    if model is not None:
        trainer_metrics = trainer_metrics_from_prediction_rows(rows)
    metric_derivation_seconds = time.perf_counter() - metric_start
    real_overall = metrics[
        (metrics["Ablation"] == "real") & (metrics["Scope"] == "overall")
    ].iloc[0]
    result: dict[str, Any] = {
        "Dataset_ID": data.dataset_id,
        "Kind": data.kind,
        "Preset": data.preset,
        "Seed": seed,
        "Model": model_name,
        "deterministic": deterministic,
        "checkpoint_selection": checkpoint_metric,
        "best_epoch": history_payload["best_epoch"],
        "best_metric_value": history_payload["best_metric_value"],
        "validate_only": bool(args.validate_only),
        "epochs_requested": 0 if args.validate_only else args.epochs,
        "epochs_run": len(history.get("train_loss", [])),
        "train_time_s": train_seconds,
        # Sólo el loop común de predicción es comparable con Persistence.
        "eval_time_s": prediction_eval_seconds,
        "metric_derivation_time_s": metric_derivation_seconds,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20)
            if args.device.startswith("cuda")
            else 0.0
        ),
        "n_params": count_parameters(model),
        "paired_parameter_tensors": paired_parameter_tensors,
        "test_rmse_z": float(real_overall["rmse_z"]),
        "test_mae_z": float(real_overall["mae_z"]),
        "test_rmse": float(real_overall["rmse"]),
        "test_mae": float(real_overall["mae"]),
        "test_nll_z": float(real_overall["nll_z"]),
        "test_crps_z": float(real_overall["crps_z"]),
        "test_nll": float(real_overall["nll"]),
        "test_crps": float(real_overall["crps"]),
        "test_mean_scale_z": float(real_overall["mean_scale_z"]),
        "test_mean_scale": float(real_overall["mean_scale"]),
        "test_coverage_90": float(real_overall["coverage_90"]),
        "test_coverage_95": float(real_overall["coverage_95"]),
        "train_samples": len(data.train),
        "val_samples": len(data.validation),
        "test_samples": len(data.test),
        "max_history_events": data.max_history_events,
        "Run_Dir": str(run_dir),
    }
    result.update(
        {f"trainer_{key}": float(value) for key, value in trainer_metrics.items()}
    )
    for ablation in args.timestamp_ablations:
        selected = metrics[
            (metrics["Ablation"] == ablation) & (metrics["Scope"] == "overall")
        ]
        if not selected.empty:
            result[f"rmse_z_{ablation}"] = float(selected.iloc[0]["rmse_z"])
    return result, rows, metrics


def _device_name(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA pero torch.cuda.is_available() es False.")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def _sha256_for_file_signature(
    resolved_path: str, size: int, mtime_ns: int
) -> str:
    """Hashea una versión concreta del archivo y reutiliza el digest en el proceso."""
    del size, mtime_ns  # Forman parte de la clave del cache.
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_for_file_signature(str(path), stat.st_size, stat.st_mtime_ns)


def file_provenance(path: Path, *, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Identidad por contenido de un archivo de código, configuración o datos."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        label = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(path)
    return {
        "path": label,
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


@lru_cache(maxsize=None)
def repository_provenance(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Commit y estado del worktree; los bytes efectivos se hashean aparte."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.replace("\r\n", "\n").strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=all")
    return {
        "git_commit": commit,
        "git_available": commit is not None,
        "worktree_dirty": bool(status) if status is not None else None,
        "status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status is not None
            else None
        ),
    }


def run_configuration(
    data: PreparedPhysicalData,
    args: argparse.Namespace,
    *,
    model_name: str,
    seed: int,
    protocol_seed: int,
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    """Configuración completa usada para decidir si un run es reanudable."""
    sources = [file_provenance(path) for path in source_paths]
    return {
        "schema_version": 1,
        "implementation": implementation_provenance(),
        "protocol_config": file_provenance(args.config),
        "dataset": {
            "dataset_id": data.dataset_id,
            "kind": data.kind,
            "preset": data.preset,
            "sources": sources,
            "observation_rows": data.row_counts,
            "scaler_mean": data.scaler.mean.tolist(),
            "scaler_std": data.scaler.std.tolist(),
            "train_samples": len(data.train),
            "validation_samples": len(data.validation),
            "test_samples": len(data.test),
        },
        "task": {
            "horizons": args.horizons,
            "train_horizon_range": [
                args.train_horizon_min,
                args.train_horizon_max,
            ],
            "train_horizon_sampling": args.train_horizon_sampling,
            "queries_per_sample": args.queries_per_sample,
            "history_duration": args.history_duration,
            "max_history_events": data.max_history_events,
            "history_subsampling": args.history_subsampling,
            "cache_deterministic_history": args.cache_deterministic_history,
            "max_observation_rows_per_split": args.max_observation_rows_per_split,
            "protocol_seed": protocol_seed,
        },
        "sampling": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "model_name": model_name,
        "seed": seed,
        "base_model": asdict(
            base_model_config(
                data,
                args,
                prediction_head=(
                    "gaussian" if is_gaussian_model(model_name) else "point"
                ),
            )
        ),
        "query_variant": (
            asdict(query_variant_config(model_name))
            if model_name not in {"Persistence", "BasisDecoder", "BasisDecoder-Gaussian", "BasisDecoder-CTSSM"}
            else None
        ),
        "basis_variant": (
            asdict(basis_variant_config(model_name))
            if model_name.startswith("BasisDecoder")
            else None
        ),
        "training": asdict(training_config(args, None, model_name=model_name)),
        "evaluation": {
            "timestamp_ablations": args.timestamp_ablations,
            "save_predictions": bool(args.save_predictions),
            "validate_only": bool(args.validate_only),
            "device": args.device,
            "checkpoints": not args.no_checkpoints,
            "deterministic": bool(args.deterministic),
        },
    }


def run_configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _environment_versions() -> dict[str, str | int | None]:
    """Versiones que pueden modificar semántica numérica o serialización."""
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pyarrow": str(pyarrow.__version__),
        "pytorch": str(torch.__version__),
        "pytorch_cuda": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn": torch.backends.cudnn.version(),
        "pyyaml": str(yaml.__version__),
    }


def implementation_provenance(
    source_paths: Sequence[Path] | None = None,
    *,
    root: Path = REPOSITORY_ROOT,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Huella reproducible de código efectivo y versiones del entorno.

    Se hashean bytes de fuentes, no el commit Git, por lo que cambios locales
    quedan representados aunque el worktree esté sucio. Las rutas se ordenan y
    se guardan relativas a ``root`` para no incorporar la ubicación del clone.
    """
    root = root.resolve()
    files: list[dict[str, str]] = []
    requested_paths = source_paths or IMPLEMENTATION_SOURCE_PATHS
    for path in sorted(
        (Path(item).resolve() for item in requested_paths),
        key=lambda item: item.as_posix(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Fuente de implementación ausente: {path}")
        try:
            label = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"La fuente {path} no está contenida en la raíz {root}."
            ) from exc
        files.append(
            {
                "path": label,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "sources": files,
        "environment": dict(environment or _environment_versions()),
        "repository": repository_provenance(root),
    }
    payload["fingerprint"] = run_configuration_fingerprint(payload)
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publica JSON sólo cuando el contenido completo está en disco."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_artifact_manifest(
    run_dir: Path, relative_paths: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Registra tamaño y SHA-256 de cada artefacto publicado antes del sentinel."""
    manifest: dict[str, dict[str, Any]] = {}
    for relative in sorted(dict.fromkeys(relative_paths)):
        path = run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Artefacto de run ausente: {path}")
        manifest[Path(relative).as_posix()] = {
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return manifest


def _artifact_manifest_is_valid(
    run_dir: Path,
    manifest: Mapping[str, Any],
    required: Sequence[str],
) -> bool:
    for relative in required:
        record = manifest.get(Path(relative).as_posix())
        path = run_dir / relative
        if not isinstance(record, Mapping) or not path.is_file():
            return False
        try:
            size = int(record["size"])
            digest = str(record["sha256"])
        except (KeyError, TypeError, ValueError):
            return False
        if path.stat().st_size != size or sha256_file(path) != digest:
            return False
    return True


def completed_run_result(
    run_dir: Path,
    fingerprint: str,
    *,
    require_predictions: bool,
) -> dict[str, Any] | None:
    """Carga un run sólo si sentinel, fingerprint y artefactos concuerdan."""
    sentinel_path = run_dir / "result.json"
    required = ["run_config.json", "metrics.csv", "history.json"]
    if require_predictions:
        required.append("predictions.parquet")
    if not sentinel_path.is_file() or any(
        not (run_dir / relative).is_file() for relative in required
    ):
        return None
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        run_config = json.loads(
            (run_dir / "run_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if sentinel.get("schema_version") != 1 or sentinel.get("fingerprint") != fingerprint:
        return None
    if (
        not isinstance(run_config, dict)
        or run_config.get("schema_version") != 1
        or run_config.get("fingerprint") != fingerprint
        or not isinstance(run_config.get("configuration"), dict)
        or run_configuration_fingerprint(run_config["configuration"]) != fingerprint
    ):
        return None
    configuration = run_config["configuration"]
    evaluation = configuration.get("evaluation", {})
    if (
        configuration.get("model_name") != "Persistence"
        and not bool(evaluation.get("validate_only", False))
        and bool(evaluation.get("checkpoints", False))
    ):
        required.append("checkpoints/best_model.pt")
    manifest = sentinel.get("artifacts")
    if not isinstance(manifest, Mapping) or not _artifact_manifest_is_valid(
        run_dir, manifest, required
    ):
        return None
    result = sentinel.get("result")
    return result if isinstance(result, dict) else None


def completed_run_fingerprint(run_dir: Path) -> str | None:
    """Lee el fingerprint de un sentinel completo, sin validar artefactos."""
    path = run_dir / "result.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("fingerprint")
    return value if isinstance(value, str) else None


def invalidate_result_sentinel(run_dir: Path) -> Path | None:
    """Retira atómicamente el sentinel antes de sobrescribir un run.

    Si el proceso nuevo falla, el directorio ya no puede confundirse con una
    ejecución completa. El sentinel anterior queda archivado para auditoría.
    """
    sentinel = run_dir / "result.json"
    if not sentinel.is_file():
        return None
    archive_dir = run_dir / "invalidated_results"
    archive_dir.mkdir(parents=True, exist_ok=True)
    while True:
        archived = archive_dir / f"result.{time.time_ns()}.json"
        if not archived.exists():
            sentinel.replace(archived)
            return archived


def main() -> None:
    parsed = parse_args()
    args = resolve_options(parsed, load_config(parsed.config))
    args.device = _device_name(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets(
        args.data_root,
        args.kinds,
        args.presets,
        args.limit_datasets_per_kind,
        args.dataset_ids,
    )
    results = []
    for dataset_index, spec in enumerate(datasets):
        kind, preset, observations_path, truth_path = spec
        print(f"[{dataset_index + 1}/{len(datasets)}] Preparando {kind}/{preset}/{observations_path.parent.name}")
        max_history_events = (
            args.max_history_events_univariate
            if kind == "univariate"
            else args.max_history_events_multivariate
        )
        dataset_protocol_seed = args.protocol_seed + dataset_index * 10_000
        data = prepare_physical_data(
            spec,
            horizons=args.horizons,
            history_duration=args.history_duration,
            max_history_events=max_history_events,
            history_subsampling=args.history_subsampling,
            max_samples={
                "train": args.max_train_samples,
                "validation": args.max_val_samples,
                "test": args.max_test_samples,
            },
            max_observation_rows_per_split=args.max_observation_rows_per_split,
            protocol_seed=dataset_protocol_seed,
            train_horizon_range=(args.train_horizon_min, args.train_horizon_max),
            train_horizon_sampling=args.train_horizon_sampling,
            queries_per_sample=args.queries_per_sample,
        )
        dataset_dir = args.output_dir / kind / preset / data.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "data_metadata.json").write_text(
            json.dumps(
                {
                    "dataset_id": data.dataset_id,
                    "kind": kind,
                    "preset": preset,
                    "horizons": args.horizons,
                    "train_horizon_range": [
                        args.train_horizon_min,
                        args.train_horizon_max,
                    ],
                    "train_horizon_sampling": args.train_horizon_sampling,
                    "queries_per_train_sample": args.queries_per_sample,
                    "history_duration": args.history_duration,
                    "max_history_events": max_history_events,
                    "history_subsampling": args.history_subsampling,
                    "cache_deterministic_history": args.cache_deterministic_history,
                    "observation_rows": data.row_counts,
                    "observation_row_cap": args.max_observation_rows_per_split,
                    "forecast_origins": data.forecast_origin_audit,
                    "scaler_mean": data.scaler.mean.tolist(),
                    "scaler_std": data.scaler.std.tolist(),
                    "source_provenance": [
                        file_provenance(observations_path),
                        file_provenance(truth_path),
                    ],
                    "protocol_config": file_provenance(args.config),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for seed in args.seeds:
            for model_name in args.models:
                print(f"  - seed={seed} model={model_name}")
                run_dir = dataset_dir / f"seed_{seed}" / model_name
                run_dir.mkdir(parents=True, exist_ok=True)
                configuration = run_configuration(
                    data,
                    args,
                    model_name=model_name,
                    seed=seed,
                    protocol_seed=dataset_protocol_seed,
                    source_paths=(observations_path, truth_path),
                )
                fingerprint = run_configuration_fingerprint(configuration)
                existing_fingerprint = completed_run_fingerprint(run_dir)
                if (
                    not args.force_rerun
                    and existing_fingerprint is not None
                    and existing_fingerprint != fingerprint
                ):
                    raise RuntimeError(
                        f"El run completo en {run_dir} usa otro fingerprint. "
                        "Use --force-rerun para reemplazarlo o elija otro "
                        "--output-dir."
                    )
                previous = (
                    None
                    if args.force_rerun
                    else completed_run_result(
                        run_dir,
                        fingerprint,
                        require_predictions=args.save_predictions,
                    )
                )
                if previous is not None:
                    print("    run completo y compatible; se reutiliza result.json")
                    results.append(previous)
                    pd.DataFrame(results).to_csv(
                        args.output_dir / "benchmark_physical_models.csv",
                        index=False,
                    )
                    continue
                archived = invalidate_result_sentinel(run_dir)
                if archived is not None:
                    print(f"    sentinel anterior invalidado: {archived.name}")
                atomic_write_json(
                    run_dir / "run_config.json",
                    {
                        "schema_version": 1,
                        "fingerprint": fingerprint,
                        "configuration": configuration,
                    },
                )
                result, rows, metrics = run_model(
                    model_name, data, args, seed=seed, run_dir=run_dir
                )
                metrics.to_csv(run_dir / "metrics.csv", index=False)
                if args.save_predictions:
                    rows.to_parquet(run_dir / "predictions.parquet", index=False)
                artifact_paths = ["run_config.json", "metrics.csv", "history.json"]
                if args.save_predictions:
                    artifact_paths.append("predictions.parquet")
                if (
                    model_name != "Persistence"
                    and not args.validate_only
                    and not args.no_checkpoints
                ):
                    artifact_paths.append("checkpoints/best_model.pt")
                atomic_write_json(
                    run_dir / "result.json",
                    {
                        "schema_version": 1,
                        "fingerprint": fingerprint,
                        "artifacts": run_artifact_manifest(
                            run_dir, artifact_paths
                        ),
                        "result": result,
                    },
                )
                results.append(result)
                pd.DataFrame(results).to_csv(
                    args.output_dir / "benchmark_physical_models.csv", index=False
                )
    summary = pd.DataFrame(results).sort_values(
        ["Kind", "Preset", "Dataset_ID", "Seed", "test_rmse_z", "Model"]
    )
    summary.to_csv(args.output_dir / "benchmark_physical_models.csv", index=False)
    print(summary[["Kind", "Preset", "Seed", "Model", "test_rmse_z", "train_time_s"]].to_string(index=False))
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()
