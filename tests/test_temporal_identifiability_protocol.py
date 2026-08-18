from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.temporal_identifiability_benchmark import (
    TIMESTAMP_ABLATIONS,
    aggregate_metrics,
    apply_timestamp_ablation,
    build_counterfactual_examples,
    discover_datasets,
    fit_control_predictions,
    load_protocol_config,
    predict_under_timestamp_ablations,
    stratified_metrics,
    timestamp_precision_diagnostics,
    timestamp_sensitivity,
)


ROOT = Path(__file__).resolve().parents[1]


def _truth_frame() -> pd.DataFrame:
    times = np.linspace(0.0, 30.0, 601)
    values = np.exp(-0.12 * times)
    split = np.where(times < 19.0, "train", np.where(times < 22.0, "validation", "test"))
    return pd.DataFrame(
        {
            "time": times,
            "clean_value": values,
            "split": split,
            "channel_index": 0,
            "channel": "x00",
        }
    )


def test_protocol_config_is_explicitly_temporal_and_identifiable() -> None:
    cfg, raw = load_protocol_config(
        ROOT / "configs" / "benchmark" / "temporal_identifiability.yaml"
    )

    assert cfg.history_duration == 8.0
    assert cfg.queries_per_anchor >= 2
    assert cfg.randomize_query_slots is True
    assert raw["data"]["timestamp_dtype"].startswith("float64")
    assert set(raw["diagnostics"]["timestamp_ablations"]) == set(TIMESTAMP_ABLATIONS)


def test_counterfactual_examples_share_history_but_not_query_time_or_slot() -> None:
    examples = build_counterfactual_examples(
        _truth_frame(),
        dataset_id="decay_0000",
        kind="univariate",
        preset="decay",
        split="train",
        n_anchors=8,
        queries_per_anchor=4,
        history_duration=4.0,
        slope_lookback=0.2,
        horizon_range=(0.2, 2.0),
        horizon_sampling="log_uniform",
        randomize_query_slots=True,
        rng=np.random.default_rng(7),
    )

    per_anchor = examples.groupby("Anchor_ID")
    assert examples["anchor_time"].min() >= 4.0
    assert (per_anchor["anchor_time"].nunique() == 1).all()
    assert (per_anchor["query_time"].nunique() == 4).all()
    assert (per_anchor["query_slot"].nunique() == 4).all()
    assert (per_anchor["target"].nunique() == 4).all()
    assert not np.array_equal(examples["query_slot"], examples["horizon_rank"])


def test_timestamp_ablations_have_the_declared_invariants() -> None:
    history = np.asarray([10.0, 10.3, 12.0, 12.5, 16.0])
    queries = np.asarray([16.5, 19.0])

    real_h, real_q = apply_timestamp_ablation(history, queries, "real")
    permuted_h, permuted_q = apply_timestamp_ablation(
        history, queries, "permuted_gaps", rng=np.random.default_rng(3)
    )
    grid_h, grid_q = apply_timestamp_ablation(history, queries, "regular_grid")
    equal_h, equal_q = apply_timestamp_ablation(history, queries, "all_equal")

    assert real_h[-1] == 0.0
    assert np.allclose(real_q, [0.5, 3.0])
    assert np.all(np.diff(permuted_h) >= 0)
    assert permuted_h[0] == pytest.approx(real_h[0])
    assert permuted_h[-1] == 0.0
    assert np.allclose(permuted_q, real_q)
    assert np.allclose(np.diff(grid_h), np.diff(grid_h)[0])
    assert np.allclose(grid_q, real_q)
    assert np.count_nonzero(equal_h) == 0
    assert np.count_nonzero(equal_q) == 0


def test_generic_predictor_can_be_audited_without_model_factory() -> None:
    history_values = np.asarray([[1.0], [0.8], [0.7]])
    history_times = np.asarray([1.0, 2.0, 5.0])
    query_times = np.asarray([6.0, 9.0])

    def predictor(values, history, queries):
        return values[-1, 0] + 0.1 * queries

    predictions = predict_under_timestamp_ablations(
        predictor,
        history_values,
        history_times,
        query_times,
        variants=("real", "all_equal", "regular_grid"),
    )
    sensitivity = timestamp_sensitivity(predictions)

    assert np.allclose(predictions["real"], [0.8, 1.1])
    assert np.allclose(predictions["all_equal"], [0.7, 0.7])
    assert sensitivity.loc[
        sensitivity["Ablation"] == "regular_grid",
        "mean_absolute_prediction_change",
    ].item() == pytest.approx(0.0)


def test_relative_float32_avoids_absolute_timestamp_collapse() -> None:
    base = 1_700_000_000.0
    times = base + np.arange(1000, dtype=np.float64) * 0.01
    result = timestamp_precision_diagnostics(times)

    assert result["absolute_float32_collapsed_fraction"] > 0.99
    assert result["relative_float32_collapsed_fraction"] == 0.0
    assert result["relative_window_size"] == 512


def test_controls_and_stratified_metrics_have_complete_contract() -> None:
    truth = _truth_frame()
    train = build_counterfactual_examples(
        truth,
        dataset_id="decay_0000",
        kind="univariate",
        preset="decay",
        split="train",
        n_anchors=48,
        queries_per_anchor=4,
        history_duration=4.0,
        slope_lookback=0.2,
        horizon_range=(0.1, 2.0),
        horizon_sampling="log_uniform",
        randomize_query_slots=True,
        rng=np.random.default_rng(9),
    )
    evaluation = build_counterfactual_examples(
        truth,
        dataset_id="decay_0000",
        kind="univariate",
        preset="decay",
        split="test",
        n_anchors=24,
        queries_per_anchor=4,
        history_duration=4.0,
        slope_lookback=0.2,
        horizon_range=(0.1, 2.0),
        horizon_sampling="log_uniform",
        randomize_query_slots=True,
        rng=np.random.default_rng(10),
    )
    for frame in (train, evaluation):
        frame["max_gap"] = 0.05 + frame["horizon"] * 0.01
        frame["density"] = 100.0 / (1.0 + frame["horizon"])

    predictions = fit_control_predictions(
        train,
        evaluation,
        queries_per_anchor=4,
        ridge_lambda=1e-5,
    )
    overall = aggregate_metrics(predictions)
    by_stratum = stratified_metrics(predictions, bins=3)

    assert set(predictions["Model"]) == {
        "Persistence",
        "Ordinal",
        "ExplicitHorizon",
    }
    assert set(overall["Model"]) == set(predictions["Model"])
    assert set(by_stratum["Stratum"]) == {"horizon", "max_gap", "density"}
    assert set(by_stratum["Kind"]) == {"univariate", "all"}
    assert (by_stratum["n"] > 0).all()
    rmse = overall.set_index(["Kind", "Model"])["rmse"]
    assert rmse.loc[("all", "ExplicitHorizon")] < rmse.loc[("all", "Ordinal")]


def test_controls_do_not_mix_equal_dataset_ids_across_kinds() -> None:
    rows = []
    for kind, preset, sign in (
        ("univariate", "decay", 1.0),
        ("multivariate", "growth", -1.0),
    ):
        for index, horizon in enumerate(np.linspace(0.1, 1.0, 12)):
            rows.append(
                {
                    "Kind": kind,
                    "Preset": preset,
                    "Dataset_ID": "shared_0000",
                    "channel_index": 0,
                    "query_slot": index % 4,
                    "last_value": 0.0,
                    "local_slope": 0.0,
                    "horizon": horizon,
                    "target": sign * horizon,
                }
            )
    train = pd.DataFrame(rows)
    evaluation = train.groupby(["Kind", "Preset"], sort=False).head(4).copy()

    predictions = fit_control_predictions(
        train,
        evaluation,
        queries_per_anchor=4,
        ridge_lambda=1e-8,
    )
    explicit = predictions[predictions["Model"] == "ExplicitHorizon"]

    assert (explicit.loc[explicit["Kind"] == "univariate", "prediction"] > 0).all()
    assert (explicit.loc[explicit["Kind"] == "multivariate", "prediction"] < 0).all()


def test_dataset_id_filter_keeps_generator_realizations_separate(
    tmp_path: Path,
) -> None:
    for kind in ("univariate", "multivariate"):
        for dataset_id in ("long_gaps_0000", "long_gaps_gseed3031_0000"):
            directory = tmp_path / kind / "long_gaps" / dataset_id
            directory.mkdir(parents=True)
            (directory / "observations.parquet").touch()
            (directory / "truth.parquet").touch()

    selected = discover_datasets(
        tmp_path,
        ("univariate", "multivariate"),
        presets=("long_gaps",),
        dataset_ids=("long_gaps_gseed3031_0000",),
    )

    assert len(selected) == 2
    assert {item[0] for item in selected} == {"univariate", "multivariate"}
    assert {item[2].parent.name for item in selected} == {
        "long_gaps_gseed3031_0000"
    }
    with pytest.raises(FileNotFoundError, match="missing_0000"):
        discover_datasets(
            tmp_path,
            ("univariate", "multivariate"),
            dataset_ids=("missing_0000",),
        )
