"""Microbenchmark reproducible del hot path de reporting físico.

Compara la referencia escalar previa (un diccionario por celda) con el armado
vectorizado actual y compara ``qcut`` por ablación con el cache de estratos
invariantes. No entrena modelos ni modifica artefactos del benchmark.

Ejemplo::

    conda run -n memoria python scripts/microbenchmark_physical_evaluation.py \
        --examples 512 --repeats 7
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")])

from scripts.benchmark_physical_models import (  # noqa: E402
    ChannelScaler,
    _DISTRIBUTION_COLUMNS,
    _PREDICTION_ROW_COLUMNS,
    _gaussian_diagnostics,
    _prediction_frame_for_ablation,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--horizons", type=int, default=4)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--ablations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def synthetic_inputs(
    *, examples: int, horizons: int, channels: int, ablations: int, seed: int
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    tuple[int, int, int],
    list[torch.Tensor],
    list[torch.Tensor],
    list[str],
    ChannelScaler,
]:
    rng = np.random.default_rng(seed)
    shape = (examples, horizons, channels)
    cells = int(np.prod(shape))
    flat_positions = np.arange(cells, dtype=np.int64)
    sample, horizon, channel = np.unravel_index(flat_positions, shape)
    target_z = rng.normal(size=cells).astype(np.float64)
    means = np.linspace(-3.0, 3.0, channels, dtype=np.float64)
    stds = np.linspace(0.5, 2.5, channels, dtype=np.float64)
    scaler = ChannelScaler(means, stds)
    history_events = 8 + (sample * 3 + channel * 5) % 96
    template = pd.DataFrame(
        {
            "Dataset_ID": np.full(cells, "micro_0000", dtype=object),
            "Kind": np.full(cells, "multivariate", dtype=object),
            "Preset": np.full(cells, "micro", dtype=object),
            "Seed": np.full(cells, seed, dtype=np.int64),
            "Model": np.full(cells, "FixedGaussian", dtype=object),
            "Example": sample.astype(np.int64),
            "Target_Index": horizon.astype(np.int64),
            "Horizon": np.asarray([0.25, 1.0, 3.0, 8.0], dtype=np.float64)[
                horizon % 4
            ],
            "Query_Time": 1_700_000_000.0
            + sample.astype(np.float64)
            + horizon.astype(np.float64),
            "Channel": channel.astype(np.int64),
            "history_events": history_events.astype(np.int64),
            "sampled_history_events": np.minimum(history_events, 64).astype(
                np.int64
            ),
            "density": history_events.astype(np.float64) / 8.0,
            "max_gap": 1.0 + ((sample + channel) % 23).astype(np.float64),
            "median_gap": 0.25
            + ((sample * 2 + channel) % 11).astype(np.float64) / 4.0,
            "last_observation_age": ((sample + 2 * channel) % 17).astype(
                np.float64
            )
            / 3.0,
            "global_history_events": (history_events + channels).astype(np.int64),
            "global_density": (history_events + channels).astype(np.float64)
            / 8.0,
            "global_max_gap": 2.0
            + ((sample + channel) % 29).astype(np.float64),
            "global_median_gap": 0.5
            + ((sample + channel) % 13).astype(np.float64) / 4.0,
            "global_last_observation_age": ((sample + channel) % 19).astype(
                np.float64
            )
            / 3.0,
            "target_z": target_z,
            "target": target_z * stds[channel] + means[channel],
        }
    )
    predictions = []
    log_scales = []
    names = []
    for index in range(ablations):
        predictions.append(
            torch.from_numpy(
                (target_z + rng.normal(scale=0.2 + index / 100.0, size=cells))
                .astype(np.float32)
                .reshape(shape)
            )
        )
        log_scales.append(
            torch.from_numpy(
                rng.normal(loc=-0.5, scale=0.3, size=cells)
                .astype(np.float32)
                .reshape(shape)
            )
        )
        names.append(f"ablation_{index:02d}")
    return (
        template,
        flat_positions,
        shape,
        predictions,
        log_scales,
        names,
        scaler,
    )


def scalar_reference(
    template: pd.DataFrame,
    flat_positions: np.ndarray,
    predictions: list[torch.Tensor],
    log_scales: list[torch.Tensor],
    names: list[str],
    scaler: ChannelScaler,
) -> pd.DataFrame:
    """Referencia conservadora del loop escalar anterior."""

    base_records = template.to_dict(orient="records")
    records = []
    for name, prediction_tensor, log_scale_tensor in zip(
        names, predictions, log_scales
    ):
        prediction_flat = prediction_tensor.numpy().reshape(-1)
        log_scale_flat = log_scale_tensor.numpy().reshape(-1)
        for row_index, flat_position in enumerate(flat_positions):
            record = dict(base_records[row_index])
            channel = int(record["Channel"])
            prediction_z = float(prediction_flat[flat_position])
            target_z = float(record["target_z"])
            record["Ablation"] = name
            record["prediction_z"] = prediction_z
            record["prediction"] = float(
                scaler.inverse_channel(np.asarray(prediction_z), channel)
            )
            record.update(
                _gaussian_diagnostics(
                    prediction_z,
                    target_z,
                    float(log_scale_flat[flat_position]),
                    float(scaler.std[channel]),
                )
            )
            records.append(record)
    return pd.DataFrame.from_records(records).loc[:, _PREDICTION_ROW_COLUMNS]


def vectorized_current(
    template: pd.DataFrame,
    flat_positions: np.ndarray,
    shape: tuple[int, int, int],
    predictions: list[torch.Tensor],
    log_scales: list[torch.Tensor],
    names: list[str],
    scaler: ChannelScaler,
) -> pd.DataFrame:
    return pd.concat(
        [
            _prediction_frame_for_ablation(
                template,
                flat_positions,
                shape,
                prediction,
                log_scale,
                ablation=name,
                scaler=scaler,
            )
            for name, prediction, log_scale in zip(
                names, predictions, log_scales
            )
        ],
        ignore_index=True,
    )


def summarize_without_cross_ablation_cache(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            summarize_predictions(selected)
            for _, selected in rows.groupby("Ablation", sort=True)
        ],
        ignore_index=True,
    )


def measure(function: Callable[[], object], repeats: int) -> list[float]:
    durations = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        start = time.perf_counter()
        try:
            function()
        finally:
            durations.append(time.perf_counter() - start)
            gc.enable()
    return durations


def timing_summary(durations: list[float], rows: int) -> dict[str, float]:
    values = np.asarray(durations, dtype=np.float64)
    median = float(np.median(values))
    return {
        "median_s": median,
        "p10_s": float(np.percentile(values, 10)),
        "p90_s": float(np.percentile(values, 90)),
        "rows_per_s": float(rows / median),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.examples,
        args.horizons,
        args.channels,
        args.ablations,
        args.repeats,
    ) <= 0:
        raise ValueError("Todos los tamaños y repeats deben ser positivos.")
    torch.set_num_threads(1)
    inputs = synthetic_inputs(
        examples=args.examples,
        horizons=args.horizons,
        channels=args.channels,
        ablations=args.ablations,
        seed=args.seed,
    )
    template, flat_positions, shape, predictions, log_scales, names, scaler = inputs
    scalar = lambda: scalar_reference(
        template, flat_positions, predictions, log_scales, names, scaler
    )
    vectorized = lambda: vectorized_current(
        template,
        flat_positions,
        shape,
        predictions,
        log_scales,
        names,
        scaler,
    )

    scalar_rows = scalar()
    vectorized_rows = vectorized()
    pd.testing.assert_frame_equal(
        vectorized_rows,
        scalar_rows,
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-14,
    )
    uncached_summary = summarize_without_cross_ablation_cache(vectorized_rows)
    cached_summary = summarize_predictions(vectorized_rows)
    sort_columns = [
        "Dataset_ID",
        "Kind",
        "Preset",
        "Seed",
        "Model",
        "Ablation",
        "Scope",
        "Level",
    ]
    pd.testing.assert_frame_equal(
        cached_summary.sort_values(sort_columns).reset_index(drop=True),
        uncached_summary.sort_values(sort_columns).reset_index(drop=True),
        check_exact=True,
    )

    # Warm-up fuera de las muestras reportadas.
    scalar()
    vectorized()
    summarize_without_cross_ablation_cache(vectorized_rows)
    summarize_predictions(vectorized_rows)
    scalar_times = measure(scalar, args.repeats)
    vector_times = measure(vectorized, args.repeats)
    uncached_times = measure(
        lambda: summarize_without_cross_ablation_cache(vectorized_rows),
        args.repeats,
    )
    cached_times = measure(
        lambda: summarize_predictions(vectorized_rows), args.repeats
    )
    row_count = len(vectorized_rows)
    scalar_summary = timing_summary(scalar_times, row_count)
    vector_summary = timing_summary(vector_times, row_count)
    uncached_bin_summary = timing_summary(uncached_times, row_count)
    cached_bin_summary = timing_summary(cached_times, row_count)
    result = {
        "schema_version": 1,
        "workload": {
            "examples": args.examples,
            "horizons": args.horizons,
            "channels": args.channels,
            "ablations": args.ablations,
            "rows": row_count,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "row_construction": {
            "before_scalar": scalar_summary,
            "after_vectorized": vector_summary,
            "speedup": scalar_summary["median_s"] / vector_summary["median_s"],
        },
        "invariant_bins": {
            "before_per_ablation": uncached_bin_summary,
            "after_cached": cached_bin_summary,
            "speedup": uncached_bin_summary["median_s"]
            / cached_bin_summary["median_s"],
        },
        "equivalence": {
            "rows": True,
            "summaries": True,
            "rtol": 1e-12,
            "atol": 1e-14,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
