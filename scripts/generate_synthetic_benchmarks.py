"""Generate one fast univariate and multivariate dataset per benchmark preset.

Expected repository layout::

    repository/
    ├── scripts/
    │   └── generate_benchmark_presets_fast.py
    └── src/
        └── data/
            ├── __init__.py
            ├── irregular_timeseries_generator.py
            └── irregular_timeseries_generator_fast.py

For large datasets, Parquet is strongly recommended over CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.irregular_timeseries_generator import benchmark_preset
from data.irregular_timeseries_generator_fast import (
    FastGenerationOptions,
    FastIrregularTimeSeriesGenerator,
)


DEFAULT_PRESETS = (
    "regular_control",
    "renewal",
    "bursty",
    "long_gaps",
    "informative",
    "nonstationary",
    "noisy",
    "hard_mixed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one optimized univariate and asynchronous multivariate "
            "dataset for every selected benchmark preset."
        )
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=DEFAULT_PRESETS,
        default=list(DEFAULT_PRESETS),
        help="Presets to generate (default: regular control plus all irregular presets).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base seed shared by all preset configurations (default: 2026).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Independent generator seeds. When supplied, overrides --seed and "
            "stores every realization under a unique dataset id."
        ),
    )
    parser.add_argument(
        "--univariate-observations",
        type=positive_int,
        default=250,
        help="Observations in each univariate dataset (default: 250).",
    )
    parser.add_argument(
        "--multivariate-observations",
        type=positive_int,
        default=1200,
        help=(
            "Total events in each asynchronous multivariate dataset "
            "(default: 1200)."
        ),
    )
    parser.add_argument(
        "--n-channels",
        type=positive_int,
        default=6,
        help="Channels in each multivariate dataset (default: 6).",
    )
    parser.add_argument(
        "--dense-steps",
        type=positive_int,
        default=4096,
        help=(
            "Resolution of the latent continuous trajectory (default: 4096). "
            "Forecast horizons should span multiple latent steps."
        ),
    )
    parser.add_argument(
        "--min-observations-per-channel",
        type=positive_int,
        default=8,
        help="Minimum events assigned to every channel (default: 8).",
    )
    parser.add_argument(
        "--channel-rate-concentration",
        type=positive_float,
        default=3.0,
        help=(
            "Dirichlet concentration controlling channel-count imbalance; "
            "smaller values are more heterogeneous (default: 3.0)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Directory containing univariate/ and multivariate/ (default: data).",
    )
    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Storage format. Parquet is recommended for large data (default: parquet).",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=("snappy", "zstd", "gzip", "brotli", "none"),
        default="snappy",
        help="Parquet compression codec (default: snappy).",
    )
    parser.add_argument(
        "--row-group-size",
        type=positive_int,
        default=1_000_000,
        help="Rows per Parquet row group (default: 1,000,000).",
    )
    parser.add_argument(
        "--no-dense-truth",
        action="store_true",
        help="Do not save the small dense noise-free reference trajectory.",
    )
    parser.add_argument(
        "--no-clean-value",
        action="store_true",
        help="Omit clean_value from observation rows to reduce memory and disk usage.",
    )
    parser.add_argument(
        "--no-global-sort",
        action="store_true",
        help=(
            "Do not globally order multivariate events by time. Use only when "
            "the downstream pipeline consumes channel-partitioned data."
        ),
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip exact irregularity metrics during generation.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable the optional Numba implementation of bursty sampling.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("The value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("The value must be positive.")
    return parsed


def json_default(value: Any) -> Any:
    """Convert NumPy and Path objects before serializing metadata."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def metadata_summary(bundle: Any, kind: str) -> dict[str, Any]:
    """Build the collection summary without regrouping millions of rows."""

    metadata = bundle.metadata
    summary: dict[str, Any] = {
        "dataset_id": metadata["dataset_id"],
        "kind": kind,
        "n_rows": len(bundle.observations),
        "n_channels": 1 if kind == "univariate" else metadata["n_channels"],
    }

    if kind == "univariate":
        metrics = metadata.get("irregularity")
        if metrics:
            summary.update(metrics)
        return summary

    channel_metrics = metadata.get("irregularity_by_channel")
    if channel_metrics:
        metrics = list(channel_metrics.values())
        summary.update(
            {
                "mean_cv_dt": float(np.mean([item["cv_dt"] for item in metrics])),
                "max_gap_ratio": float(
                    np.max([item["max_gap_ratio"] for item in metrics])
                ),
                "mean_burstiness": float(
                    np.mean([item["burstiness"] for item in metrics])
                ),
            }
        )
    return summary


def save_frame(
    frame: Any,
    path_without_suffix: Path,
    *,
    output_format: str,
    parquet_compression: str,
    row_group_size: int,
) -> Path:
    """Write a DataFrame using settings suitable for multi-million-row data."""

    if output_format == "csv":
        output_path = path_without_suffix.with_suffix(".csv")
        frame.to_csv(output_path, index=False)
        return output_path

    output_path = path_without_suffix.with_suffix(".parquet")
    compression = None if parquet_compression == "none" else parquet_compression
    try:
        frame.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression=compression,
            row_group_size=row_group_size,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it with: pip install pyarrow"
        ) from exc
    return output_path


def save_collection_fast(
    collection: Any,
    output_dir: Path,
    *,
    output_format: str,
    parquet_compression: str,
    row_group_size: int,
) -> None:
    """Save a collection without calling SyntheticCollection.summary().

    The original save method recomputes multivariate metrics using a Pandas
    groupby. At ten million rows that creates unnecessary extra work because
    the fast generator already stored those metrics in bundle.metadata.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    collection_metadata = {
        "kind": collection.kind,
        "n_datasets": len(collection.datasets),
        "generator_seed": int(collection.config.seed),
        "config": asdict(collection.config),
    }
    # ``collection_metadata.json`` conserva compatibilidad con consumidores
    # existentes. El archivo por seed evita que una generación posterior en el
    # mismo preset borre la procedencia de realizaciones anteriores.
    write_json(output_dir / "collection_metadata.json", collection_metadata)
    write_json(
        output_dir / f"collection_metadata_gseed{int(collection.config.seed)}.json",
        collection_metadata,
    )

    summaries: list[dict[str, Any]] = []
    for bundle in collection.datasets:
        dataset_id = str(bundle.metadata["dataset_id"])
        dataset_dir = output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        save_frame(
            bundle.observations,
            dataset_dir / "observations",
            output_format=output_format,
            parquet_compression=parquet_compression,
            row_group_size=row_group_size,
        )
        if bundle.truth is not None:
            save_frame(
                bundle.truth,
                dataset_dir / "truth",
                output_format=output_format,
                parquet_compression=parquet_compression,
                row_group_size=row_group_size,
            )

        dataset_metadata = dict(bundle.metadata)
        dataset_metadata.setdefault("generator_seed", int(collection.config.seed))
        write_json(dataset_dir / "metadata.json", dataset_metadata)
        summaries.append(metadata_summary(bundle, collection.kind))

    summary_path = output_dir / "collection_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
        new_ids = {str(row["dataset_id"]) for row in summaries}
        summaries = [
            row for row in existing if str(row.get("dataset_id")) not in new_ids
        ] + summaries
    fieldnames = list(dict.fromkeys(key for row in summaries for key in row))
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def validate_args(args: argparse.Namespace) -> None:
    required = args.n_channels * args.min_observations_per_channel
    if args.multivariate_observations < required:
        raise ValueError(
            "--multivariate-observations must be at least "
            "--n-channels * --min-observations-per-channel. "
            f"Received {args.multivariate_observations}; minimum is {required}."
        )


def elapsed_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)} min {remaining:.1f} s"


def main() -> None:
    args = parse_args()
    validate_args(args)

    univariate_root = args.output_root / "univariate"
    multivariate_root = args.output_root / "multivariate"

    options = FastGenerationOptions(
        compact_dtypes=True,
        include_clean_value=not args.no_clean_value,
        global_sort=not args.no_global_sort,
        compute_metrics=not args.skip_metrics,
        categorical_labels=True,
        use_numba_for_bursty=not args.no_numba,
    )

    generation_seeds = list(args.seeds) if args.seeds is not None else [args.seed]
    if len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("--seeds no puede contener valores repetidos.")

    total_start = time.perf_counter()

    for preset_name in args.presets:
        for generation_seed in generation_seeds:
            preset_start = time.perf_counter()
            config = benchmark_preset(preset_name, seed=generation_seed)
            config.dynamics.dense_steps = int(args.dense_steps)
            config.store_dense_truth = not args.no_dense_truth
            generator = FastIrregularTimeSeriesGenerator(config, options=options)
            dataset_prefix = (
                f"{preset_name}_gseed{generation_seed}"
                if args.seeds is not None else preset_name
            )

            univariate_start = time.perf_counter()
            univariate = generator.generate_univariate_collection(
                n_datasets=1,
                n_observations=args.univariate_observations,
                dataset_prefix=dataset_prefix,
            )
            univariate_output_dir = univariate_root / preset_name
            save_collection_fast(
                univariate,
                univariate_output_dir,
                output_format=args.output_format,
                parquet_compression=args.parquet_compression,
                row_group_size=args.row_group_size,
            )
            univariate_elapsed = time.perf_counter() - univariate_start

            # Release the first collection before allocating the large multivariate one.
            del univariate
            gc.collect()

            multivariate_start = time.perf_counter()
            multivariate = generator.generate_multivariate_collection(
                n_datasets=1,
                n_observations=args.multivariate_observations,
                n_channels=args.n_channels,
                layout="asynchronous",
                min_observations_per_channel=args.min_observations_per_channel,
                channel_rate_concentration=args.channel_rate_concentration,
                dataset_prefix=dataset_prefix,
            )
            multivariate_output_dir = multivariate_root / preset_name
            save_collection_fast(
                multivariate,
                multivariate_output_dir,
                output_format=args.output_format,
                parquet_compression=args.parquet_compression,
                row_group_size=args.row_group_size,
            )
            multivariate_elapsed = time.perf_counter() - multivariate_start

            del multivariate, generator
            gc.collect()

            preset_elapsed = time.perf_counter() - preset_start
            print(
                f"[{preset_name} | generator_seed={generation_seed}] generated and saved | "
                f"univariate={elapsed_text(univariate_elapsed)} | "
                f"multivariate={elapsed_text(multivariate_elapsed)} | "
                f"total={elapsed_text(preset_elapsed)}\n"
                f"  {univariate_output_dir}\n"
                f"  {multivariate_output_dir}",
                flush=True,
            )

    print(
        f"Completed {len(args.presets) * len(generation_seeds)} preset/seed realization(s) in "
        f"{elapsed_text(time.perf_counter() - total_start)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
