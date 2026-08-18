"""Dataset-level analysis for the irregular synthetic benchmark.

Training seeds are repeated measurements, never independent experimental units.
The script first averages seeds within each (kind, preset, dataset) unit and only
then ranks or compares models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UNIT_COLUMNS = ["Kind", "Preset", "Dataset_ID"]
KEY_COLUMNS = [*UNIT_COLUMNS, "Model"]
DEFAULT_PRIMARY_MODELS = ("Custom", "Custom-Time2Vec", "EncDec-AR")
DEFAULT_COMPARATORS = (
    "Persistence",
    "LastValueTimeMLP",
    "STraTS_Adapter",
    "CoFormer",
)
DEFAULT_ABLATIONS = ("Custom-OrdinalTime", "Custom-NoRole")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a completed synthetic benchmark at dataset level."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "synthetic_benchmark" / "benchmark_synthetic.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 84, 126))
    parser.add_argument(
        "--primary-models",
        nargs="+",
        default=DEFAULT_PRIMARY_MODELS,
        help="Pre-specified co-primary proposed models.",
    )
    parser.add_argument("--comparators", nargs="+", default=DEFAULT_COMPARATORS)
    parser.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=2026)
    return parser.parse_args()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    running_max = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def bootstrap_mean_interval(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    means = finite[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def complete_balanced_rows(
    frame: pd.DataFrame, models: list[str], seeds: list[int]
) -> pd.DataFrame:
    expected = {(model, seed) for model in models for seed in seeds}
    complete_units: list[tuple[str, str, str]] = []
    for unit, group in frame.groupby(UNIT_COLUMNS, sort=False):
        observed = set(zip(group["Model"].astype(str), group["Seed"].astype(int)))
        if expected.issubset(observed):
            complete_units.append(unit)
    if not complete_units:
        raise RuntimeError(
            "No hay ningún dataset con cobertura completa para los modelos y semillas solicitados."
        )
    complete_index = pd.MultiIndex.from_tuples(complete_units, names=UNIT_COLUMNS)
    indexed = frame.set_index(UNIT_COLUMNS)
    return indexed.loc[indexed.index.isin(complete_index)].reset_index()


def numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("val_", "test_", "train_time_s", "epochs_run", "n_params_")
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and pd.api.types.is_numeric_dtype(frame[column])
    ]


def pairwise_primary_analysis(
    seed_averaged: pd.DataFrame,
    primary_models: list[str],
    comparators: list[str],
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pivot = seed_averaged.pivot(index=UNIT_COLUMNS, columns="Model", values=metric)
    records: list[dict[str, float | int | str]] = []
    for primary in primary_models:
        if primary not in pivot.columns:
            raise ValueError(f"Falta el modelo co-principal {primary} para {metric}.")
        for comparator in comparators:
            if comparator not in pivot.columns:
                raise ValueError(f"Falta el comparador {comparator} para {metric}.")
            if comparator == primary:
                continue
            paired = pivot[[primary, comparator]].dropna()
            primary_values = paired[primary].to_numpy(dtype=np.float64)
            comparator_values = paired[comparator].to_numpy(dtype=np.float64)
            delta = primary_values - comparator_values
            ci_low, ci_high = bootstrap_mean_interval(delta, bootstrap_samples, rng)
            relative = np.divide(
                comparator_values - primary_values,
                np.abs(comparator_values),
                out=np.full_like(primary_values, np.nan),
                where=np.abs(comparator_values) > 1e-12,
            )
            nonzero = delta[np.abs(delta) > 1e-12]
            p_value = (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if nonzero.size >= 2 else np.nan
            )
            records.append(
                {
                    "Primary_Model": primary,
                    "Comparator": comparator,
                    "Metric": metric,
                    "n_datasets": len(paired),
                    "primary_mean": float(primary_values.mean()),
                    "comparator_mean": float(comparator_values.mean()),
                    "mean_delta_primary_minus_comparator": float(delta.mean()),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "mean_relative_reduction": float(np.nanmean(relative)),
                    "primary_wins": int(np.sum(delta < 0)),
                    "ties": int(np.sum(np.isclose(delta, 0.0))),
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(records)
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"])
    result["confirmatory_superiority"] = (
        (result["mean_delta_primary_minus_comparator"] < 0)
        & (result["delta_ci95_high"] < 0)
        & (result["wilcoxon_p_holm"] < 0.05)
    )
    return result.sort_values(["Primary_Model", "Comparator"])


def dimension_summary(seed_averaged: pd.DataFrame, dimension: str) -> pd.DataFrame:
    marker = f"test_rmse_{dimension}_"
    columns = [column for column in seed_averaged.columns if column.startswith(marker)]
    records: list[dict[str, float | int | str]] = []
    for column in columns:
        index = int(column.removeprefix(marker))
        grouped = seed_averaged.groupby("Model")[column]
        for model, values in grouped:
            records.append(
                {
                    "Model": model,
                    "Dimension": dimension,
                    "Index": index,
                    "mean_test_rmse": float(values.mean()),
                    "std_test_rmse": float(values.std(ddof=1)),
                    "n_datasets": int(values.count()),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples debe ser > 0.")
    output_dir = args.output_dir or args.results.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.results)
    required = {*UNIT_COLUMNS, "Seed", "Model", "val_rmse", "test_rmse"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    frame = frame.drop_duplicates([*UNIT_COLUMNS, "Seed", "Model"], keep="last")
    models = sorted(args.models or frame["Model"].dropna().astype(str).unique().tolist())
    seeds = [int(seed) for seed in args.seeds]
    frame = frame[frame["Model"].isin(models) & frame["Seed"].isin(seeds)].copy()
    balanced = complete_balanced_rows(frame, models, seeds)

    metric_columns = numeric_metric_columns(balanced)
    seed_averaged = balanced.groupby(KEY_COLUMNS, as_index=False)[metric_columns].mean()
    seed_averaged.to_csv(output_dir / "dataset_seed_averaged.csv", index=False)

    available = set(seed_averaged["Model"].astype(str))
    primary_models = [str(name) for name in args.primary_models]
    comparators = [str(name) for name in args.comparators]
    ablations = [str(name) for name in args.ablations]
    requested_inference_models = set(primary_models + comparators + ablations)
    missing_inference_models = requested_inference_models.difference(available)
    if missing_inference_models:
        raise ValueError(
            f"Faltan modelos requeridos para el análisis: {sorted(missing_inference_models)}"
        )
    ranks = seed_averaged.copy()
    ranks["test_rmse_rank"] = ranks.groupby(UNIT_COLUMNS)["test_rmse"].rank(
        method="average", ascending=True
    )
    model_summary = (
        ranks.groupby("Model", as_index=False)
        .agg(
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_test_mae=("test_mae", "mean"),
            mean_rank=("test_rmse_rank", "mean"),
            mean_train_time_s=("train_time_s", "mean"),
            n_datasets=("Dataset_ID", "count"),
        )
        .sort_values(["mean_test_rmse", "Model"])
    )
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)

    rng = np.random.default_rng(args.analysis_seed)
    pairwise = pairwise_primary_analysis(
        seed_averaged,
        primary_models,
        comparators,
        "test_rmse",
        args.bootstrap_samples,
        rng,
    )
    pairwise.to_csv(output_dir / "pairwise_coprimary_test_rmse.csv", index=False)
    ablation_pairwise = pairwise_primary_analysis(
        seed_averaged,
        ["Custom"],
        ablations,
        "test_rmse",
        args.bootstrap_samples,
        rng,
    )
    ablation_pairwise.to_csv(output_dir / "pairwise_ablations_test_rmse.csv", index=False)

    for dimension in ("target", "channel"):
        summary = dimension_summary(seed_averaged, dimension)
        if not summary.empty:
            summary.to_csv(output_dir / f"summary_by_{dimension}.csv", index=False)

    metadata = {
        "results": str(args.results.resolve()),
        "models": models,
        "training_seeds_averaged_within_dataset": seeds,
        "n_complete_dataset_units": int(
            seed_averaged[UNIT_COLUMNS].drop_duplicates().shape[0]
        ),
        "co_primary_models": primary_models,
        "comparators": comparators,
        "ablation_reference": "Custom",
        "ablations": ablations,
        "primary_selection": "pre_specified_before_confirmatory_test_analysis",
        "primary_metric": "test_rmse",
        "bootstrap_samples": args.bootstrap_samples,
        "analysis_seed": args.analysis_seed,
        "multiple_comparison_correction": "Holm",
        "statistical_unit": "generated dataset realization",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Co-primary models: {', '.join(primary_models)}")
    print(f"Complete dataset units: {metadata['n_complete_dataset_units']}")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
