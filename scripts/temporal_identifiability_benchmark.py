"""Benchmark temporal identificable para series sintéticas irregulares.

Este runner complementa (no reemplaza) ``benchmark_synthetic.py``.  Construye
consultas en *tiempo físico* a partir de ``truth.parquet`` y usa exactamente la
misma historia para varios tiempos objetivo continuos.  Los slots de consulta
se aleatorizan para que un modelo ordinal no pueda deducir el horizonte desde
la posición del token.

También incluye utilidades reutilizables para:

* corromper timestamps con ablaciones emparejadas;
* medir la pérdida de precisión al castear timestamps absolutos a float32;
* calcular métricas por bins de horizonte, gap y densidad;
* ejecutar controles ``Persistence``, ``Ordinal`` y ``ExplicitHorizon``.

Ejemplo acotado::

    conda run -n memoria python scripts/temporal_identifiability_benchmark.py \
        --kinds univariate multivariate --limit-datasets-per-kind 1 \
        --max-train-anchors 24 --max-eval-anchors 16

La API ``predict_under_timestamp_ablations`` permite aplicar las mismas
transformaciones a cualquier modelo sin acoplar este módulo a su factory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "temporal_identifiability.yaml"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "temporal_identifiability"

TIMESTAMP_ABLATIONS = (
    "real",
    "all_equal",
    "permuted_gaps",
    "regular_grid",
    "ordinal",
    "query_only",
    "history_only",
)


@dataclass(frozen=True)
class ProtocolConfig:
    train_split: str
    eval_split: str
    history_duration: float
    slope_lookback: float
    train_anchors: int
    eval_anchors: int
    queries_per_anchor: int
    horizon_min: float
    horizon_max: float
    horizon_sampling: str
    randomize_query_slots: bool
    ridge_lambda: float
    strata_bins: int
    precision_rows_per_row_group: int
    max_ablation_history_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa identificabilidad temporal usando horizontes físicos y "
            "consultas contrafactuales sobre data/univariate y data/multivariate."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("univariate", "multivariate"),
        default=None,
    )
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Filtra IDs exactos sin mezclar realizaciones (p.ej. *_gseed3031_0000).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--limit-datasets-per-kind",
        type=int,
        default=None,
        help="Limita datasets por tipo; útil para pruebas de humo.",
    )
    parser.add_argument("--max-train-anchors", type=int, default=None)
    parser.add_argument("--max-eval-anchors", type=int, default=None)
    parser.add_argument(
        "--skip-observation-scan",
        action="store_true",
        help=(
            "Omite el escaneo de observations.parquet. Aún genera consultas y "
            "controles, pero gap/densidad y auditoría float32 quedan vacíos."
        ),
    )
    return parser.parse_args()


def load_protocol_config(path: Path) -> tuple[ProtocolConfig, dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Configuración temporal inválida: {path}")
    task = raw.get("task", {})
    controls = raw.get("controls", {})
    diagnostics = raw.get("diagnostics", {})
    cfg = ProtocolConfig(
        train_split=str(task.get("train_split", "train")),
        eval_split=str(task.get("eval_split", "test")),
        history_duration=float(task["history_duration"]),
        slope_lookback=float(task["slope_lookback"]),
        train_anchors=int(task["train_anchors"]),
        eval_anchors=int(task["eval_anchors"]),
        queries_per_anchor=int(task["queries_per_anchor"]),
        horizon_min=float(task["horizon_range"][0]),
        horizon_max=float(task["horizon_range"][1]),
        horizon_sampling=str(task.get("horizon_sampling", "log_uniform")),
        randomize_query_slots=bool(task.get("randomize_query_slots", True)),
        ridge_lambda=float(controls.get("ridge_lambda", 1e-3)),
        strata_bins=int(diagnostics.get("strata_bins", 4)),
        precision_rows_per_row_group=int(
            diagnostics.get("precision_rows_per_row_group", 50_000)
        ),
        max_ablation_history_points=int(
            diagnostics.get("max_ablation_history_points", 512)
        ),
    )
    validate_protocol_config(cfg)
    return cfg, raw


def validate_protocol_config(cfg: ProtocolConfig) -> None:
    if cfg.history_duration <= 0 or cfg.slope_lookback <= 0:
        raise ValueError("history_duration y slope_lookback deben ser > 0.")
    if cfg.train_anchors <= 0 or cfg.eval_anchors <= 0:
        raise ValueError("train_anchors y eval_anchors deben ser > 0.")
    if cfg.queries_per_anchor < 2:
        raise ValueError(
            "queries_per_anchor debe ser >= 2 para construir contrafactuales."
        )
    if not (0 < cfg.horizon_min < cfg.horizon_max):
        raise ValueError("horizon_range debe cumplir 0 < mínimo < máximo.")
    if cfg.horizon_sampling not in {"uniform", "log_uniform"}:
        raise ValueError("horizon_sampling debe ser uniform o log_uniform.")
    if cfg.strata_bins < 2:
        raise ValueError("strata_bins debe ser >= 2.")
    if cfg.precision_rows_per_row_group <= 1:
        raise ValueError("precision_rows_per_row_group debe ser > 1.")


def discover_datasets(
    data_root: Path,
    kinds: Sequence[str],
    presets: Sequence[str] | None = None,
    limit_per_kind: int | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> list[tuple[str, str, Path, Path]]:
    """Descubre pares observations/truth con límite independiente por tipo."""
    selected_presets = set(presets) if presets else None
    selected_ids = set(dataset_ids) if dataset_ids else None
    discovered_ids: set[str] = set()
    discovered: list[tuple[str, str, Path, Path]] = []
    for kind in kinds:
        kind_root = data_root / kind
        if not kind_root.is_dir():
            raise FileNotFoundError(f"No existe el directorio: {kind_root}")
        candidates: list[tuple[str, str, Path, Path]] = []
        for preset_dir in sorted(path for path in kind_root.iterdir() if path.is_dir()):
            if selected_presets is not None and preset_dir.name not in selected_presets:
                continue
            for observations_path in sorted(
                preset_dir.glob("*/observations.parquet")
            ):
                if (
                    selected_ids is not None
                    and observations_path.parent.name not in selected_ids
                ):
                    continue
                truth_path = observations_path.with_name("truth.parquet")
                if not truth_path.is_file():
                    raise FileNotFoundError(
                        f"Falta truth.parquet junto a {observations_path}"
                    )
                candidates.append(
                    (kind, preset_dir.name, observations_path, truth_path)
                )
                discovered_ids.add(observations_path.parent.name)
        if limit_per_kind is not None:
            candidates = candidates[:limit_per_kind]
        discovered.extend(candidates)
    if selected_ids is not None:
        missing = sorted(selected_ids - discovered_ids)
        if missing:
            raise FileNotFoundError(
                "No se encontraron los dataset IDs solicitados: " + ", ".join(missing)
            )
    if not discovered:
        raise FileNotFoundError("No se encontraron datasets para los filtros dados.")
    return discovered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    try:
        label = resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "size": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _runtime_provenance() -> dict[str, object]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.replace("\r\n", "\n").strip()

    status = git("status", "--short", "--untracked-files=all")
    return {
        "environment": {
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pyarrow.__version__),
            "pyyaml": str(yaml.__version__),
        },
        "repository": {
            "git_commit": git("rev-parse", "HEAD"),
            "worktree_dirty": bool(status) if status is not None else None,
            "status_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest()
                if status is not None
                else None
            ),
        },
    }


def load_truth_long(truth_path: Path, kind: str) -> pd.DataFrame:
    columns = ["time", "clean_value", "split"]
    if kind == "multivariate":
        columns.extend(["channel_index", "channel"])
    frame = pd.read_parquet(truth_path, columns=columns)
    frame["time"] = frame["time"].astype(np.float64)
    frame["clean_value"] = frame["clean_value"].astype(np.float64)
    if kind == "univariate":
        frame["channel_index"] = 0
        frame["channel"] = "x00"
    frame["channel_index"] = frame["channel_index"].astype(int)
    return frame.sort_values(["channel_index", "time"], kind="stable").reset_index(
        drop=True
    )


def _sample_horizons(
    rng: np.random.Generator,
    count: int,
    minimum: float,
    maximum: float,
    sampling: str,
) -> np.ndarray:
    if sampling == "uniform":
        return rng.uniform(minimum, maximum, size=count)
    return np.exp(rng.uniform(np.log(minimum), np.log(maximum), size=count))


def _common_split_range(truth: pd.DataFrame, split: str) -> tuple[float, float]:
    per_channel: list[tuple[float, float]] = []
    for _, channel in truth.groupby("channel_index", sort=True):
        selected = channel.loc[channel["split"].astype(str) == split, "time"]
        if selected.empty:
            raise ValueError(f"El truth no contiene split '{split}' en todos los canales.")
        per_channel.append((float(selected.min()), float(selected.max())))
    return max(value[0] for value in per_channel), min(value[1] for value in per_channel)


def build_counterfactual_examples(
    truth: pd.DataFrame,
    *,
    dataset_id: str,
    kind: str,
    preset: str,
    split: str,
    n_anchors: int,
    queries_per_anchor: int,
    history_duration: float,
    slope_lookback: float,
    horizon_range: tuple[float, float],
    horizon_sampling: str,
    randomize_query_slots: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Crea grupos con una historia compartida y múltiples consultas físicas.

    Los targets se interpolan sobre la trayectoria limpia de alta resolución.
    ``query_slot`` se asigna independientemente del horizonte para impedir que
    el orden del token actúe como una etiqueta temporal encubierta.
    """
    split_start, split_end = _common_split_range(truth, split)
    minimum_horizon, maximum_horizon = horizon_range
    global_start = max(
        float(channel["time"].min())
        for _, channel in truth.groupby("channel_index", sort=True)
    )
    # Toda consulta debe contar realmente con la duración histórica declarada.
    # En train, aceptar anchors cercanos a t=0 produciría ventanas truncadas.
    anchor_low = max(split_start, global_start + history_duration)
    anchor_high = split_end - maximum_horizon
    if anchor_high <= anchor_low:
        raise ValueError(
            f"Split {split} demasiado corto para horizon_max={maximum_horizon}."
        )

    anchor_times = np.linspace(anchor_low, anchor_high, n_anchors + 2)[1:-1]
    channels = list(truth.groupby("channel_index", sort=True))
    records: list[dict[str, float | int | str]] = []

    for anchor_index, anchor_time in enumerate(anchor_times):
        horizons = _sample_horizons(
            rng,
            queries_per_anchor,
            minimum_horizon,
            maximum_horizon,
            horizon_sampling,
        )
        horizon_ranks = np.argsort(np.argsort(horizons, kind="stable"), kind="stable")
        slots = (
            rng.permutation(queries_per_anchor)
            if randomize_query_slots
            else horizon_ranks.copy()
        )
        anchor_id = f"{dataset_id}:{split}:{anchor_index:05d}"

        for channel_index, channel in channels:
            times = channel["time"].to_numpy(dtype=np.float64)
            values = channel["clean_value"].to_numpy(dtype=np.float64)
            last_value = float(np.interp(anchor_time, times, values))
            slope_start = max(float(times[0]), anchor_time - slope_lookback)
            previous_value = float(np.interp(slope_start, times, values))
            denominator = max(anchor_time - slope_start, np.finfo(np.float64).eps)
            local_slope = (last_value - previous_value) / denominator
            channel_name = str(channel["channel"].iloc[0])

            for query_index, horizon in enumerate(horizons):
                query_time = anchor_time + float(horizon)
                target = float(np.interp(query_time, times, values))
                records.append(
                    {
                        "Dataset_ID": dataset_id,
                        "Kind": kind,
                        "Preset": preset,
                        "Split": split,
                        "Anchor_ID": anchor_id,
                        "anchor_time": float(anchor_time),
                        "query_time": query_time,
                        "horizon": float(horizon),
                        "query_slot": int(slots[query_index]),
                        "horizon_rank": int(horizon_ranks[query_index]),
                        "channel_index": int(channel_index),
                        "channel": channel_name,
                        "last_value": last_value,
                        "local_slope": float(local_slope),
                        "target": target,
                        "history_duration": float(history_duration),
                    }
                )
    result = pd.DataFrame.from_records(records)
    expected = n_anchors * queries_per_anchor * len(channels)
    if len(result) != expected:
        raise RuntimeError(f"Se esperaban {expected} consultas y se generaron {len(result)}.")
    return result


def apply_timestamp_ablation(
    history_times: np.ndarray,
    query_times: np.ndarray,
    variant: str,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforma timestamps preservando el emparejamiento valores/targets.

    La salida siempre está expresada respecto del último timestamp histórico.
    ``permuted_gaps`` conserva valores y orden de eventos, pero reasigna los
    intervalos entre ellos. ``regular_grid`` conserva el span histórico.
    """
    history = np.asarray(history_times, dtype=np.float64)
    queries = np.asarray(query_times, dtype=np.float64)
    if history.ndim != 1 or queries.ndim != 1 or history.size == 0:
        raise ValueError("history_times y query_times deben ser vectores; historia no vacía.")
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(queries)):
        raise ValueError("Los timestamps deben ser finitos.")
    if np.any(np.diff(history) < 0):
        raise ValueError("history_times debe estar ordenado no decrecientemente.")
    if variant not in TIMESTAMP_ABLATIONS:
        raise ValueError(f"Ablación desconocida: {variant}")

    origin = history[-1]
    history_relative = history - origin
    query_relative = queries - origin
    if variant == "real":
        return history_relative, query_relative
    if variant == "all_equal":
        return np.zeros_like(history), np.zeros_like(queries)
    if variant == "query_only":
        return np.zeros_like(history), query_relative
    if variant == "history_only":
        return history_relative, np.zeros_like(queries)
    if variant == "ordinal":
        ordinal_history = np.arange(1 - history.size, 1, dtype=np.float64)
        ordinal_queries = np.arange(1, queries.size + 1, dtype=np.float64)
        return ordinal_history, ordinal_queries
    if variant == "regular_grid":
        if history.size == 1:
            regular = np.zeros(1, dtype=np.float64)
        else:
            regular = np.linspace(history_relative[0], 0.0, history.size)
        return regular, query_relative

    generator = rng if rng is not None else np.random.default_rng(0)
    if history.size == 1:
        return np.zeros(1, dtype=np.float64), query_relative
    permuted = generator.permutation(np.diff(history))
    rebuilt = np.concatenate(([0.0], np.cumsum(permuted)))
    rebuilt -= rebuilt[-1]
    return rebuilt, query_relative


def predict_under_timestamp_ablations(
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    history_values: np.ndarray,
    history_times: np.ndarray,
    query_times: np.ndarray,
    *,
    variants: Sequence[str] = TIMESTAMP_ABLATIONS,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Ejecuta un predictor genérico con corrupciones temporales emparejadas.

    ``predict_fn`` recibe ``(history_values, history_times_rel, query_times_rel)``.
    La interfaz deliberadamente no depende de PyTorch ni de un factory concreto.
    """
    predictions: dict[str, np.ndarray] = {}
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(variants))
    for variant, child in zip(variants, children):
        transformed_history, transformed_queries = apply_timestamp_ablation(
            history_times,
            query_times,
            variant,
            rng=np.random.default_rng(child),
        )
        prediction = np.asarray(
            predict_fn(history_values, transformed_history, transformed_queries)
        )
        predictions[variant] = prediction
    return predictions


def timestamp_sensitivity(
    predictions: Mapping[str, np.ndarray], reference: str = "real"
) -> pd.DataFrame:
    if reference not in predictions:
        raise ValueError(f"Falta la predicción de referencia '{reference}'.")
    baseline = np.asarray(predictions[reference], dtype=np.float64)
    records = []
    for variant, prediction in predictions.items():
        candidate = np.asarray(prediction, dtype=np.float64)
        if candidate.shape != baseline.shape:
            raise ValueError("Todas las predicciones deben tener la misma shape.")
        delta = candidate - baseline
        records.append(
            {
                "Ablation": variant,
                "mean_absolute_prediction_change": float(np.mean(np.abs(delta))),
                "rms_prediction_change": float(np.sqrt(np.mean(np.square(delta)))),
                "max_absolute_prediction_change": float(np.max(np.abs(delta))),
            }
        )
    return pd.DataFrame(records)


def timestamp_precision_diagnostics(
    times: np.ndarray, *, relative_window_size: int = 512
) -> dict[str, float | int]:
    """Compara float32 absoluto vs resta del origen dentro de cada ventana.

    Restar una sola vez el inicio de una trayectoria que comienza en cero no
    recupera precisión. La operación que consume el modelo ocurre por ventana;
    por eso la alternativa correcta recentra bloques locales antes del cast.
    """
    if relative_window_size < 2:
        raise ValueError("relative_window_size debe ser >= 2.")
    original = np.asarray(times, dtype=np.float64)
    original = original[np.isfinite(original)]
    original.sort()
    if original.size < 2:
        return {
            "n_timestamps": int(original.size),
            "n_positive_gaps": 0,
            "relative_window_size": int(relative_window_size),
            "original_nonpositive_fraction": math.nan,
            "absolute_float32_collapsed_fraction": math.nan,
            "relative_float32_collapsed_fraction": math.nan,
            "median_positive_gap": math.nan,
            "p99_positive_gap": math.nan,
        }
    gaps = np.diff(original)
    positive = gaps > 0
    absolute_gaps = np.diff(original.astype(np.float32).astype(np.float64))
    relative_collapsed = 0
    relative_positive = 0
    for start in range(0, original.size, relative_window_size):
        window = original[start : start + relative_window_size]
        if window.size < 2:
            continue
        window_gaps = np.diff(window)
        relative = (window - window[0]).astype(np.float32).astype(np.float64)
        cast_gaps = np.diff(relative)
        window_positive = window_gaps > 0
        relative_positive += int(np.count_nonzero(window_positive))
        relative_collapsed += int(
            np.count_nonzero(window_positive & (cast_gaps == 0))
        )
    denominator = int(np.count_nonzero(positive))
    if denominator == 0:
        absolute_fraction = relative_fraction = math.nan
    else:
        absolute_fraction = float(np.count_nonzero(positive & (absolute_gaps == 0)) / denominator)
        relative_fraction = (
            float(relative_collapsed / relative_positive)
            if relative_positive
            else math.nan
        )
    positive_values = gaps[positive]
    return {
        "n_timestamps": int(original.size),
        "n_positive_gaps": denominator,
        "relative_window_size": int(relative_window_size),
        "original_nonpositive_fraction": float(np.mean(gaps <= 0)),
        "absolute_float32_collapsed_fraction": absolute_fraction,
        "relative_float32_collapsed_fraction": relative_fraction,
        "median_positive_gap": (
            float(np.median(positive_values)) if positive_values.size else math.nan
        ),
        "p99_positive_gap": (
            float(np.quantile(positive_values, 0.99)) if positive_values.size else math.nan
        ),
    }


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals if right >= left)
    merged: list[list[float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def scan_observation_times(
    observations_path: Path,
    kind: str,
    intervals: Sequence[tuple[float, float]],
    *,
    precision_rows_per_row_group: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Escanea tiempos por row group con memoria acotada.

    Retorna dos mapas por canal: observaciones dentro de ventanas de historia y
    una muestra estratificada por row group para la auditoría float32.
    """
    parquet = pq.ParquetFile(observations_path)
    columns = ["time"] + (["channel_index"] if kind == "multivariate" else [])
    merged_intervals = _merge_intervals(intervals)
    context_chunks: dict[int, list[np.ndarray]] = {}
    precision_chunks: dict[int, list[np.ndarray]] = {}

    for row_group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_index, columns=columns)
        times = table.column("time").to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False
        )
        if times.size == 0:
            continue
        if kind == "multivariate":
            channels = table.column("channel_index").to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
        else:
            channels = np.zeros(times.size, dtype=np.int64)

        # Muestrear bloques contiguos conserva los gaps adyacentes. Un muestreo
        # sistemático con stride escondería exactamente las colisiones float32
        # que esta auditoría busca medir.
        sample_count = min(times.size, precision_rows_per_row_group)
        block_count = min(16, max(1, sample_count // 512))
        block_size = max(2, sample_count // block_count)
        starts = np.linspace(
            0, max(0, times.size - block_size), block_count, dtype=np.int64
        )
        sampled_indices = np.unique(
            np.concatenate(
                [
                    np.arange(start, min(start + block_size, times.size), dtype=np.int64)
                    for start in starts
                ]
            )
        )
        for channel_index in np.unique(channels[sampled_indices]):
            selected = sampled_indices[channels[sampled_indices] == channel_index]
            precision_chunks.setdefault(int(channel_index), []).append(times[selected])

        in_context = np.zeros(times.size, dtype=bool)
        for left, right in merged_intervals:
            in_context |= (times >= left) & (times <= right)
        if np.any(in_context):
            for channel_index in np.unique(channels[in_context]):
                selected = in_context & (channels == channel_index)
                context_chunks.setdefault(int(channel_index), []).append(times[selected])

    def combine(chunks: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for channel_index, values in chunks.items():
            combined = np.concatenate(values).astype(np.float64, copy=False)
            combined.sort()
            result[channel_index] = combined
        return result

    return combine(context_chunks), combine(precision_chunks)


def attach_history_diagnostics(
    examples: pd.DataFrame,
    observation_times: Mapping[int, np.ndarray],
    history_duration: float,
) -> pd.DataFrame:
    diagnostics: list[dict[str, float | int | str]] = []
    anchors = examples[
        ["Anchor_ID", "anchor_time", "channel_index"]
    ].drop_duplicates()
    for row in anchors.itertuples(index=False):
        times = np.asarray(observation_times.get(int(row.channel_index), []), dtype=np.float64)
        left_time = float(row.anchor_time) - history_duration
        if times.size:
            left = int(np.searchsorted(times, left_time, side="left"))
            right = int(np.searchsorted(times, float(row.anchor_time), side="right"))
            history = times[left:right]
        else:
            history = np.empty(0, dtype=np.float64)
        if history.size:
            boundaries = np.concatenate(([left_time], history, [float(row.anchor_time)]))
            gaps = np.diff(boundaries)
            max_gap = float(np.max(gaps))
            median_gap = float(np.median(gaps))
            age = float(row.anchor_time) - float(history[-1])
        else:
            max_gap = history_duration
            median_gap = history_duration
            age = history_duration
        diagnostics.append(
            {
                "Anchor_ID": row.Anchor_ID,
                "channel_index": int(row.channel_index),
                "history_observations": int(history.size),
                "density": float(history.size / history_duration),
                "max_gap": max_gap,
                "median_gap": median_gap,
                "age_since_last_observation": age,
            }
        )
    return examples.merge(
        pd.DataFrame(diagnostics),
        on=["Anchor_ID", "channel_index"],
        how="left",
        validate="many_to_one",
    )


def summarize_ablation_window(
    history_times: np.ndarray,
    query_times: np.ndarray,
    *,
    seed: int,
    max_history_points: int,
) -> pd.DataFrame:
    history = np.asarray(history_times, dtype=np.float64)
    if history.size > max_history_points:
        indices = np.linspace(0, history.size - 1, max_history_points).astype(int)
        history = history[indices]
    records = []
    children = np.random.SeedSequence(seed).spawn(len(TIMESTAMP_ABLATIONS))
    for variant, child in zip(TIMESTAMP_ABLATIONS, children):
        transformed_history, transformed_queries = apply_timestamp_ablation(
            history,
            query_times,
            variant,
            rng=np.random.default_rng(child),
        )
        gaps = np.diff(transformed_history)
        records.append(
            {
                "Ablation": variant,
                "history_points": int(transformed_history.size),
                "unique_history_timestamps": int(np.unique(transformed_history).size),
                "history_span": float(
                    transformed_history[-1] - transformed_history[0]
                ),
                "median_history_gap": float(np.median(gaps)) if gaps.size else 0.0,
                "max_history_gap": float(np.max(gaps)) if gaps.size else 0.0,
                "query_span": float(
                    np.max(transformed_queries) - np.min(transformed_queries)
                ) if transformed_queries.size else 0.0,
                "first_query_time": float(transformed_queries[0])
                if transformed_queries.size
                else math.nan,
            }
        )
    return pd.DataFrame(records)


class _StandardizedRidge:
    def __init__(self, regularization: float):
        self.regularization = float(regularization)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "_StandardizedRidge":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (x - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(x)), standardized))
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.regularization
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise RuntimeError("El regresor no ha sido ajustado.")
        x = (np.asarray(features, dtype=np.float64) - self.mean_) / self.scale_
        return np.column_stack((np.ones(len(x)), x)) @ self.coefficients_


def _ordinal_features(frame: pd.DataFrame, n_slots: int) -> np.ndarray:
    slots = frame["query_slot"].to_numpy(dtype=int)
    if np.any((slots < 0) | (slots >= n_slots)):
        raise ValueError("query_slot fuera de rango.")
    one_hot = np.eye(n_slots, dtype=np.float64)[slots]
    return np.column_stack(
        (
            frame["last_value"].to_numpy(dtype=np.float64),
            one_hot,
        )
    )


def _explicit_horizon_features(frame: pd.DataFrame) -> np.ndarray:
    horizon = frame["horizon"].to_numpy(dtype=np.float64)
    last = frame["last_value"].to_numpy(dtype=np.float64)
    slope = frame["local_slope"].to_numpy(dtype=np.float64)
    return np.column_stack(
        (
            last,
            slope,
            horizon,
            np.log1p(horizon),
            np.square(horizon),
            slope * horizon,
            last * horizon,
        )
    )


def fit_control_predictions(
    train_examples: pd.DataFrame,
    eval_examples: pd.DataFrame,
    *,
    queries_per_anchor: int,
    ridge_lambda: float,
) -> pd.DataFrame:
    """Ajusta controles por dataset/canal y retorna predicciones emparejadas."""
    # Dataset_ID se repite entre las ramas univariate/multivariate
    # (p.ej. ``bursty_0000``). Kind/Preset forman parte de la identidad para no
    # ajustar un mismo control con procesos distintos que comparten nombre.
    key_columns = ["Kind", "Preset", "Dataset_ID", "channel_index"]
    predictions: list[pd.DataFrame] = []
    train_groups = train_examples.groupby(key_columns, sort=False)
    for key, evaluation in eval_examples.groupby(key_columns, sort=False):
        try:
            training = train_groups.get_group(key)
        except KeyError as error:
            raise ValueError(f"Faltan ejemplos de entrenamiento para {key}.") from error
        delta = (
            training["target"].to_numpy(dtype=np.float64)
            - training["last_value"].to_numpy(dtype=np.float64)
        )
        ordinal = _StandardizedRidge(ridge_lambda).fit(
            _ordinal_features(training, queries_per_anchor), delta
        )
        explicit = _StandardizedRidge(ridge_lambda).fit(
            _explicit_horizon_features(training), delta
        )

        base_columns = list(evaluation.columns)
        for model, prediction in (
            ("Persistence", evaluation["last_value"].to_numpy(dtype=np.float64)),
            (
                "Ordinal",
                evaluation["last_value"].to_numpy(dtype=np.float64)
                + ordinal.predict(_ordinal_features(evaluation, queries_per_anchor)),
            ),
            (
                "ExplicitHorizon",
                evaluation["last_value"].to_numpy(dtype=np.float64)
                + explicit.predict(_explicit_horizon_features(evaluation)),
            ),
        ):
            output = evaluation[base_columns].copy()
            output["Model"] = model
            output["prediction"] = prediction
            predictions.append(output)
    return pd.concat(predictions, ignore_index=True)


def regression_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    target = frame["target"].to_numpy(dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        return {"n": 0, "rmse": math.nan, "mae": math.nan, "bias": math.nan}
    residual = prediction[valid] - target[valid]
    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouping = ["Kind", "Model"]
    for key, group in predictions.groupby(grouping, dropna=False, sort=True):
        records.append(dict(zip(grouping, key)) | regression_metrics(group))
    for model, group in predictions.groupby("Model", sort=True):
        records.append({"Kind": "all", "Model": model} | regression_metrics(group))
    return pd.DataFrame(records)


def dataset_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = ["Dataset_ID", "Kind", "Preset", "Model"]
    records = []
    for key, group in predictions.groupby(grouping, dropna=False, sort=True):
        records.append(dict(zip(grouping, key)) | regression_metrics(group))
    return pd.DataFrame(records)


def stratified_metrics(
    predictions: pd.DataFrame,
    *,
    columns: Sequence[str] = ("horizon", "max_gap", "density"),
    bins: int = 4,
) -> pd.DataFrame:
    """Calcula RMSE/MAE por cuantiles sin tratar seeds como nuevas unidades."""
    records: list[dict[str, float | int | str]] = []
    for column in columns:
        if column not in predictions or predictions[column].notna().sum() < 2:
            continue
        working = predictions.copy()
        try:
            working["_bin"] = pd.qcut(
                working[column], q=bins, duplicates="drop"
            )
        except ValueError:
            continue
        for kind_scope, scoped in [*working.groupby("Kind", sort=True), ("all", working)]:
            for (model, interval), group in scoped.groupby(
                ["Model", "_bin"], observed=True, sort=True
            ):
                records.append(
                    {
                        "Kind": str(kind_scope),
                        "Model": str(model),
                        "Stratum": column,
                        "Bin": str(interval),
                        "bin_min": float(group[column].min()),
                        "bin_max": float(group[column].max()),
                    }
                    | regression_metrics(group)
                )
    return pd.DataFrame(records)


def _history_intervals(examples: pd.DataFrame, duration: float) -> list[tuple[float, float]]:
    anchors = examples["anchor_time"].drop_duplicates().to_numpy(dtype=np.float64)
    return _merge_intervals((float(anchor - duration), float(anchor)) for anchor in anchors)


def _ablation_summary_for_dataset(
    eval_examples: pd.DataFrame,
    observation_times: Mapping[int, np.ndarray],
    *,
    cfg: ProtocolConfig,
    seed: int,
) -> pd.DataFrame:
    first_anchor = eval_examples.sort_values(["Anchor_ID", "channel_index", "query_slot"])[
        "Anchor_ID"
    ].iloc[0]
    selected_anchor = eval_examples[eval_examples["Anchor_ID"] == first_anchor]
    records = []
    for channel_index, channel in selected_anchor.groupby("channel_index", sort=True):
        anchor_time = float(channel["anchor_time"].iloc[0])
        times = np.asarray(observation_times.get(int(channel_index), []), dtype=np.float64)
        left = np.searchsorted(times, anchor_time - cfg.history_duration, side="left")
        right = np.searchsorted(times, anchor_time, side="right")
        history = times[left:right]
        if history.size == 0:
            history = np.asarray([anchor_time], dtype=np.float64)
        queries = channel.sort_values("query_slot")["query_time"].to_numpy(dtype=np.float64)
        summary = summarize_ablation_window(
            history,
            queries,
            seed=seed + int(channel_index),
            max_history_points=cfg.max_ablation_history_points,
        )
        summary["Dataset_ID"] = channel["Dataset_ID"].iloc[0]
        summary["channel_index"] = int(channel_index)
        summary["Anchor_ID"] = first_anchor
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def run_dataset(
    dataset: tuple[str, str, Path, Path],
    *,
    cfg: ProtocolConfig,
    seed: int,
    train_anchors: int,
    eval_anchors: int,
    skip_observation_scan: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kind, preset, observations_path, truth_path = dataset
    dataset_id = observations_path.parent.name
    truth = load_truth_long(truth_path, kind)
    train = build_counterfactual_examples(
        truth,
        dataset_id=dataset_id,
        kind=kind,
        preset=preset,
        split=cfg.train_split,
        n_anchors=train_anchors,
        queries_per_anchor=cfg.queries_per_anchor,
        history_duration=cfg.history_duration,
        slope_lookback=cfg.slope_lookback,
        horizon_range=(cfg.horizon_min, cfg.horizon_max),
        horizon_sampling=cfg.horizon_sampling,
        randomize_query_slots=cfg.randomize_query_slots,
        rng=np.random.default_rng(seed),
    )
    evaluation = build_counterfactual_examples(
        truth,
        dataset_id=dataset_id,
        kind=kind,
        preset=preset,
        split=cfg.eval_split,
        n_anchors=eval_anchors,
        queries_per_anchor=cfg.queries_per_anchor,
        history_duration=cfg.history_duration,
        slope_lookback=cfg.slope_lookback,
        horizon_range=(cfg.horizon_min, cfg.horizon_max),
        horizon_sampling=cfg.horizon_sampling,
        randomize_query_slots=cfg.randomize_query_slots,
        rng=np.random.default_rng(seed + 1),
    )

    precision_records: list[dict[str, float | int | str]] = []
    ablation_summary = pd.DataFrame()
    if skip_observation_scan:
        for frame in (train, evaluation):
            frame["history_observations"] = np.nan
            frame["density"] = np.nan
            frame["max_gap"] = np.nan
            frame["median_gap"] = np.nan
            frame["age_since_last_observation"] = np.nan
    else:
        intervals = _history_intervals(
            pd.concat([train, evaluation], ignore_index=True), cfg.history_duration
        )
        contexts, precision_samples = scan_observation_times(
            observations_path,
            kind,
            intervals,
            precision_rows_per_row_group=cfg.precision_rows_per_row_group,
        )
        train = attach_history_diagnostics(train, contexts, cfg.history_duration)
        evaluation = attach_history_diagnostics(
            evaluation, contexts, cfg.history_duration
        )
        for channel_index, times in sorted(precision_samples.items()):
            precision_records.append(
                {
                    "Dataset_ID": dataset_id,
                    "Kind": kind,
                    "Preset": preset,
                    "channel_index": int(channel_index),
                }
                | timestamp_precision_diagnostics(times)
            )
        ablation_summary = _ablation_summary_for_dataset(
            evaluation, contexts, cfg=cfg, seed=seed
        )

    examples = pd.concat([train, evaluation], ignore_index=True)
    return examples, pd.DataFrame(precision_records), ablation_summary


def main() -> None:
    args = parse_args()
    cfg, raw_config = load_protocol_config(args.config)
    kinds = tuple(args.kinds or raw_config.get("data", {}).get("kinds", []))
    if not kinds:
        kinds = ("univariate", "multivariate")
    seed = int(args.seed if args.seed is not None else raw_config.get("seed", 2026))
    train_anchors = min(cfg.train_anchors, args.max_train_anchors) if args.max_train_anchors else cfg.train_anchors
    eval_anchors = min(cfg.eval_anchors, args.max_eval_anchors) if args.max_eval_anchors else cfg.eval_anchors
    datasets = discover_datasets(
        args.data_root,
        kinds,
        args.presets,
        args.limit_datasets_per_kind,
        args.dataset_ids,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    example_frames = []
    precision_frames = []
    ablation_frames = []
    for dataset_index, dataset in enumerate(datasets):
        kind, preset, observations_path, _ = dataset
        print(f"[{dataset_index + 1}/{len(datasets)}] {kind}/{preset}/{observations_path.parent.name}")
        examples, precision, ablations = run_dataset(
            dataset,
            cfg=cfg,
            seed=seed + dataset_index * 10_000,
            train_anchors=train_anchors,
            eval_anchors=eval_anchors,
            skip_observation_scan=args.skip_observation_scan,
        )
        example_frames.append(examples)
        if not precision.empty:
            precision_frames.append(precision)
        if not ablations.empty:
            ablation_frames.append(ablations)

    examples = pd.concat(example_frames, ignore_index=True)
    train = examples[examples["Split"] == cfg.train_split]
    evaluation = examples[examples["Split"] == cfg.eval_split]
    predictions = fit_control_predictions(
        train,
        evaluation,
        queries_per_anchor=cfg.queries_per_anchor,
        ridge_lambda=cfg.ridge_lambda,
    )
    metrics = aggregate_metrics(predictions)
    per_dataset_metrics = dataset_level_metrics(predictions)
    strata = stratified_metrics(predictions, bins=cfg.strata_bins)

    examples.to_parquet(args.output_dir / "counterfactual_examples.parquet", index=False)
    predictions.to_parquet(args.output_dir / "control_predictions.parquet", index=False)
    metrics.to_csv(args.output_dir / "control_metrics.csv", index=False)
    per_dataset_metrics.to_csv(
        args.output_dir / "control_metrics_by_dataset.csv", index=False
    )
    strata.to_csv(args.output_dir / "metrics_by_temporal_stratum.csv", index=False)
    if precision_frames:
        pd.concat(precision_frames, ignore_index=True).to_csv(
            args.output_dir / "timestamp_precision.csv", index=False
        )
    if ablation_frames:
        pd.concat(ablation_frames, ignore_index=True).to_csv(
            args.output_dir / "timestamp_ablation_manifest.csv", index=False
        )

    artifact_paths = [
        args.output_dir / "counterfactual_examples.parquet",
        args.output_dir / "control_predictions.parquet",
        args.output_dir / "control_metrics.csv",
        args.output_dir / "control_metrics_by_dataset.csv",
        args.output_dir / "metrics_by_temporal_stratum.csv",
    ]
    artifact_paths.extend(
        path
        for path in (
            args.output_dir / "timestamp_precision.csv",
            args.output_dir / "timestamp_ablation_manifest.csv",
        )
        if path.is_file()
    )
    metadata = {
        "schema_version": 1,
        "config": _file_provenance(args.config),
        "implementation": _file_provenance(Path(__file__)),
        "runtime": _runtime_provenance(),
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": seed,
        "kinds": list(kinds),
        "dataset_id_filter": list(args.dataset_ids or []),
        "datasets": [
            {
                "kind": dataset[0],
                "preset": dataset[1],
                "dataset_id": dataset[2].parent.name,
                "sources": [
                    _file_provenance(dataset[2]),
                    _file_provenance(dataset[3]),
                ],
            }
            for dataset in datasets
        ],
        "n_dataset_units": len(datasets),
        "n_train_anchors_per_dataset": train_anchors,
        "n_eval_anchors_per_dataset": eval_anchors,
        "queries_per_anchor": cfg.queries_per_anchor,
        "query_slots_randomized": cfg.randomize_query_slots,
        "physical_horizon_range": [cfg.horizon_min, cfg.horizon_max],
        "timestamp_dtype_at_disk": "float64",
        "observation_scan_skipped": bool(args.skip_observation_scan),
        "controls": ["Persistence", "Ordinal", "ExplicitHorizon"],
        "control_history_values": "clean_truth_oracle",
        "timestamp_ablations": list(TIMESTAMP_ABLATIONS),
        "claim_boundary": (
            "Diagnóstico de identificabilidad sobre truth sintético; no es una "
            "estimación de desempeño externo en datos reales."
        ),
        "artifacts": {
            path.name: {
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"Resultados escritos en {args.output_dir}")


if __name__ == "__main__":
    main()
