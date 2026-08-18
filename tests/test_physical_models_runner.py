from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from scripts.benchmark_physical_models import (
    ChannelScaler,
    PhysicalCollate,
    PreparedPhysicalData,
    ablate_batch_timestamps,
    atomic_write_json,
    basis_variant_config,
    build_model,
    build_split_dataset,
    checkpoint_selection_metric,
    completed_run_result,
    completed_run_fingerprint,
    evaluate_prediction_rows,
    implementation_provenance,
    invalidate_result_sentinel,
    load_paired_reference_parameters,
    persistence_predictions,
    query_variant_config,
    run_artifact_manifest,
    run_configuration_fingerprint,
    run_model,
    summarize_predictions,
    trainer_metrics_from_prediction_rows,
)
from ts_transformer.training import Trainer


def test_resume_sentinel_requires_matching_fingerprint_and_artifacts(
    tmp_path: Path,
) -> None:
    first_configuration = {"task": {"horizon": 1.0}}
    second_configuration = {"task": {"horizon": 2.0}}
    first = run_configuration_fingerprint(first_configuration)
    second = run_configuration_fingerprint(second_configuration)
    assert first != second

    atomic_write_json(
        tmp_path / "run_config.json",
        {
            "schema_version": 1,
            "fingerprint": first,
            "configuration": first_configuration,
        },
    )
    (tmp_path / "metrics.csv").write_text("rmse\n1.0\n", encoding="utf-8")
    (tmp_path / "history.json").write_text(
        '{"schema_version": 1, "history": {}}', encoding="utf-8"
    )
    atomic_write_json(
        tmp_path / "result.json",
        {
            "schema_version": 1,
            "fingerprint": first,
            "artifacts": run_artifact_manifest(
                tmp_path, ("run_config.json", "metrics.csv", "history.json")
            ),
            "result": {"Model": "QueryCross", "test_rmse_z": 1.0},
        },
    )

    assert completed_run_result(
        tmp_path, first, require_predictions=False
    ) == {"Model": "QueryCross", "test_rmse_z": 1.0}
    assert completed_run_result(tmp_path, second, require_predictions=False) is None
    assert completed_run_result(tmp_path, first, require_predictions=True) is None
    (tmp_path / "predictions.parquet").touch()
    atomic_write_json(
        tmp_path / "result.json",
        {
            "schema_version": 1,
            "fingerprint": first,
            "artifacts": run_artifact_manifest(
                tmp_path,
                (
                    "run_config.json",
                    "metrics.csv",
                    "history.json",
                    "predictions.parquet",
                ),
            ),
            "result": {"Model": "QueryCross", "test_rmse_z": 1.0},
        },
    )
    assert completed_run_result(tmp_path, first, require_predictions=True) is not None
    assert completed_run_fingerprint(tmp_path) == first

    # No basta con que el sentinel coincida: run_config también debe concordar
    # y su payload debe producir realmente ese fingerprint.
    atomic_write_json(
        tmp_path / "run_config.json",
        {
            "schema_version": 1,
            "fingerprint": second,
            "configuration": first_configuration,
        },
    )
    assert completed_run_result(tmp_path, first, require_predictions=False) is None


def test_checkpoint_selection_is_distribution_aware() -> None:
    assert checkpoint_selection_metric("QueryCross") == "val_rmse"
    assert checkpoint_selection_metric("BasisDecoder-CTSSM") == "val_rmse"
    assert checkpoint_selection_metric("QueryCross-Gaussian") == "val_nll"
    assert checkpoint_selection_metric("BasisDecoder-Gaussian") == "val_nll"
    assert checkpoint_selection_metric("Persistence") is None


def test_implementation_provenance_hashes_dirty_sources_and_environment(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_source.write_text("VALUE = 1\n", encoding="utf-8")
    second_source.write_text("VALUE = 2\n", encoding="utf-8")
    environment = {"python": "test", "pytorch": "test"}

    first = implementation_provenance(
        (second_source, first_source), root=tmp_path, environment=environment
    )
    reordered = implementation_provenance(
        (first_source, second_source), root=tmp_path, environment=environment
    )
    assert first == reordered
    assert [item["path"] for item in first["sources"]] == [
        "first.py",
        "second.py",
    ]

    first_source.write_text("VALUE = 3\n", encoding="utf-8")
    changed = implementation_provenance(
        (first_source, second_source), root=tmp_path, environment=environment
    )
    assert changed["fingerprint"] != first["fingerprint"]


def test_invalidate_result_sentinel_archives_it_before_rerun(tmp_path: Path) -> None:
    sentinel = tmp_path / "result.json"
    sentinel.write_text('{"complete": true}', encoding="utf-8")

    archived = invalidate_result_sentinel(tmp_path)

    assert archived is not None
    assert not sentinel.exists()
    assert archived.parent == tmp_path / "invalidated_results"
    assert archived.read_text(encoding="utf-8") == '{"complete": true}'
    assert invalidate_result_sentinel(tmp_path) is None


def _truth() -> pd.DataFrame:
    times = np.linspace(0.0, 60.0, 601)
    split = np.where(times < 25.0, "train", np.where(times < 42.0, "validation", "test"))
    return pd.DataFrame(
        {
            "time": times,
            "clean_value": np.sin(times / 4.0),
            "split": split,
            "channel_index": 0,
            "channel": "x00",
        }
    )


def _observations() -> pd.DataFrame:
    times = np.linspace(0.0, 60.0, 1201)
    return pd.DataFrame(
        {
            "time": times,
            "value": np.sin(times / 4.0) + 0.01 * np.cos(times),
            "split": np.where(times < 25.0, "train", np.where(times < 42.0, "validation", "test")),
            "event_index": np.arange(len(times)),
        }
    )


def _args() -> Namespace:
    return Namespace(
        d_model=8,
        num_heads=2,
        num_layers=1,
        cross_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )


def _prepared() -> PreparedPhysicalData:
    truth = _truth()
    observations = _observations()
    scaler = ChannelScaler(np.asarray([0.0]), np.asarray([1.0]))
    datasets = {}
    bounds = {
        "train": (0.0, 25.0),
        "validation": (17.0, 42.0),
        "test": (34.0, 60.0),
    }
    for index, split in enumerate(("train", "validation", "test")):
        left, right = bounds[split]
        frame = observations[(observations.time >= left) & (observations.time <= right)]
        datasets[split] = build_split_dataset(
            frame,
            truth,
            kind="univariate",
            split=split,
            n_channels=1,
            scaler=scaler,
            horizons=[0.5, 2.0],
            history_duration=4.0,
            max_history_events=8,
            history_subsampling="uniform_time",
            sampling_seed=10 + index,
        )
    return PreparedPhysicalData(
        train=datasets["train"],
        validation=datasets["validation"],
        test=datasets["test"],
        scaler=scaler,
        kind="univariate",
        preset="toy",
        dataset_id="toy_0000",
        n_channels=1,
        max_history_events=8,
        row_counts={key: len(value) for key, value in datasets.items()},
    )


def test_physical_dataset_uses_observations_for_history_and_truth_for_queries() -> None:
    data = _prepared()
    sample = data.test[0]

    assert sample["input_values"].shape[0] <= 8 + 2
    assert sample["input_observation_counts"].shape == sample["input_timestamps"].shape
    assert sample["absolute_target_timestamps"].dtype == torch.float64
    assert sorted(sample["requested_target_horizons"].tolist()) == [0.5, 2.0]
    assert torch.all(sample["target_loss_mask"] == 1)


def test_train_queries_are_continuous_while_evaluation_horizons_are_fixed() -> None:
    truth = _truth()
    observations = _observations()
    scaler = ChannelScaler(np.asarray([0.0]), np.asarray([1.0]))
    train = build_split_dataset(
        observations[observations.time <= 25.0],
        truth,
        kind="univariate",
        split="train",
        n_channels=1,
        scaler=scaler,
        horizons=[0.5, 2.0],
        history_duration=4.0,
        max_history_events=8,
        history_subsampling="uniform_time",
        sampling_seed=71,
        train_horizon_range=(0.5, 2.0),
        train_horizon_sampling="log_uniform",
        queries_per_sample=2,
    )
    evaluation = build_split_dataset(
        observations[(observations.time >= 34.0) & (observations.time <= 60.0)],
        truth,
        kind="univariate",
        split="test",
        n_channels=1,
        scaler=scaler,
        horizons=[0.5, 2.0],
        history_duration=4.0,
        max_history_events=8,
        history_subsampling="uniform_time",
        sampling_seed=72,
    )

    train_horizons = train[0]["requested_target_horizons"]
    eval_horizons = evaluation[0]["requested_target_horizons"]
    assert torch.all((train_horizons >= 0.5) & (train_horizons <= 2.0))
    assert set(train_horizons.tolist()) != {0.5, 2.0}
    assert sorted(eval_horizons.tolist()) == [0.5, 2.0]


def test_required_model_ablations_have_scientific_flags() -> None:
    full = query_variant_config("QueryCross")
    no_time = query_variant_config("NoTime")
    query_only = query_variant_config("QueryOnly")
    ctssm = query_variant_config("CTSSM")
    gaussian = query_variant_config("QueryCross-Gaussian")

    assert full.use_relative_time_bias and full.use_query_horizon
    assert full.temporal_feature_dim == 1
    assert not no_time.use_relative_time_bias
    assert not no_time.use_query_horizon
    assert not no_time.use_history_time_encoding
    assert query_only.use_query_horizon
    assert not query_only.use_history_time_encoding
    assert ctssm.use_ctssm
    assert gaussian.use_relative_time_bias
    assert not basis_variant_config("BasisDecoder").use_ctssm
    assert basis_variant_config("BasisDecoder-CTSSM").use_ctssm


def test_common_parameters_are_explicitly_paired_across_ablations() -> None:
    data = _prepared()
    torch.manual_seed(1)
    reference = build_model("QueryCross", data, _args())
    torch.manual_seed(999)
    ablation = build_model("NoTime", data, _args())
    assert reference is not None and ablation is not None

    loaded = load_paired_reference_parameters(ablation, reference.state_dict())
    reference_state = reference.state_dict()
    ablation_state = ablation.state_dict()
    common = [
        name
        for name in ablation_state
        if name in reference_state
        and ablation_state[name].shape == reference_state[name].shape
    ]

    assert loaded == len(common)
    assert common
    assert all(torch.equal(ablation_state[name], reference_state[name]) for name in common)


def test_all_neural_variants_run_one_physical_batch() -> None:
    data = _prepared()
    loader = DataLoader(data.test, batch_size=2, collate_fn=PhysicalCollate())
    batch = next(iter(loader))

    for model_name in (
        "QueryCross",
        "BasisDecoder",
        "BasisDecoder-CTSSM",
        "NoTime",
        "QueryOnly",
        "CTSSM",
    ):
        model = build_model(model_name, data, _args()).eval()
        prediction = model(
            input_values=batch["input_values"],
            input_timestamps=batch["input_timestamps"],
            is_target_mask=batch["is_target_mask"],
            padding_mask=batch["padding_mask"],
            lengths=batch["lengths"],
            temporal_features=batch["temporal_features"],
        )
        assert prediction.shape == batch["target_values"].shape
        assert torch.isfinite(prediction).all()

    gaussian = build_model("QueryCross-Gaussian", data, _args()).eval()
    distribution = gaussian(
        input_values=batch["input_values"],
        input_timestamps=batch["input_timestamps"],
        is_target_mask=batch["is_target_mask"],
        padding_mask=batch["padding_mask"],
        lengths=batch["lengths"],
        temporal_features=batch["temporal_features"],
        return_dict=True,
    )
    assert distribution["preds"].shape == batch["target_values"].shape
    assert distribution["log_scale"].shape == batch["target_values"].shape
    assert torch.isfinite(distribution["log_scale"]).all()

    basis_gaussian = build_model("BasisDecoder-Gaussian", data, _args()).eval()
    basis_distribution = basis_gaussian(
        input_values=batch["input_values"],
        input_timestamps=batch["input_timestamps"],
        is_target_mask=batch["is_target_mask"],
        padding_mask=batch["padding_mask"],
        lengths=batch["lengths"],
        temporal_features=batch["temporal_features"],
        return_dict=True,
    )
    assert basis_distribution["preds"].shape == batch["target_values"].shape
    assert basis_distribution["log_scale"].shape == batch["target_values"].shape


def test_timestamp_corruptions_and_persistence_are_shape_safe() -> None:
    data = _prepared()
    batch = next(iter(DataLoader(data.test, batch_size=2, collate_fn=PhysicalCollate())))
    original = batch["input_timestamps"]
    regular = ablate_batch_timestamps(batch, "regular_grid", output_dim=1, seed=3)
    permuted = ablate_batch_timestamps(batch, "permuted_gaps", output_dim=1, seed=3)
    equal = ablate_batch_timestamps(batch, "all_equal", output_dim=1, seed=3)
    no_count_feature = ablate_batch_timestamps(
        batch, "real_no_count_feature", output_dim=1, seed=3
    )
    prediction = persistence_predictions(batch, output_dim=1)

    assert regular.shape == original.shape
    assert permuted.shape == original.shape
    assert torch.all(equal[~batch["padding_mask"]] == 0)
    assert torch.equal(no_count_feature, original)
    assert prediction.shape == batch["target_values"].shape
    assert torch.allclose(prediction[:, 0], prediction[:, 1])


def test_smoke_evaluation_produces_real_and_corrupted_strata() -> None:
    data = _prepared()
    loader = DataLoader(data.test, batch_size=3, collate_fn=PhysicalCollate())
    rows = evaluate_prediction_rows(
        None,
        loader,
        data,
        model_name="Persistence",
        seed=42,
        device="cpu",
        timestamp_ablations=("real", "all_equal"),
        history_duration=4.0,
    )
    metrics = summarize_predictions(rows, bins=2)

    assert set(rows["Ablation"]) == {"real", "all_equal"}
    assert {
        "overall",
        "horizon",
        "channel",
        "density_bin",
        "max_gap_bin",
        "last_observation_age_bin",
        "channel_density_bin",
        "channel_max_gap_bin",
        "channel_last_age_bin",
    }.issubset(
        set(metrics["Scope"])
    )
    real = metrics[(metrics.Ablation == "real") & (metrics.Scope == "overall")]
    equal = metrics[(metrics.Ablation == "all_equal") & (metrics.Scope == "overall")]
    assert real["rmse_z"].item() == equal["rmse_z"].item()


def test_channel_temporal_strata_use_quantiles_within_each_channel() -> None:
    records = []
    for channel, offset in ((0, 0.0), (1, 100.0)):
        for index in range(4):
            records.append(
                {
                    "Dataset_ID": "toy_0000",
                    "Kind": "multivariate",
                    "Preset": "toy",
                    "Seed": 1,
                    "Model": "Persistence",
                    "Ablation": "real",
                    "Horizon": 1.0,
                    "Channel": channel,
                    "density": offset + index + 1.0,
                    "max_gap": offset + 4.0 - index,
                    "last_observation_age": offset + index,
                    "prediction_z": float(index),
                    "target_z": 0.0,
                    "prediction": float(index),
                    "target": 0.0,
                    "nll_z": np.nan,
                    "crps_z": np.nan,
                    "coverage_90": np.nan,
                    "coverage_95": np.nan,
                    "scale_z": np.nan,
                    "nll": np.nan,
                    "crps": np.nan,
                    "scale": np.nan,
                }
            )

    metrics = summarize_predictions(pd.DataFrame.from_records(records), bins=2)
    for scope in (
        "channel_density_bin",
        "channel_max_gap_bin",
        "channel_last_age_bin",
    ):
        selected = metrics[metrics["Scope"] == scope]
        assert len(selected) == 4
        assert set(selected["n"]) == {2}
        assert selected["Level"].str.startswith("channel=0|").sum() == 2
        assert selected["Level"].str.startswith("channel=1|").sum() == 2


def test_evaluation_uses_exact_pre_subsampling_diagnostics_per_channel() -> None:
    observations = pd.DataFrame(
        {
            "time": [6.0, 6.0, 8.0, 10.0],
            "value": [6.0, 60.0, 8.0, 10.0],
            "split": ["test"] * 4,
            "event_index": np.arange(4),
            "channel_index": [0, 1, 0, 0],
        }
    )
    times = np.arange(13, dtype=np.float64)
    truth = pd.DataFrame(
        {
            "time": np.repeat(times, 2),
            "clean_value": np.column_stack([times, 10.0 * times]).reshape(-1),
            "split": "test",
            "channel_index": np.tile([0, 1], len(times)),
            "channel": np.tile(["x00", "x01"], len(times)),
        }
    )
    scaler = ChannelScaler(np.zeros(2), np.ones(2))
    dataset = build_split_dataset(
        observations,
        truth,
        kind="multivariate",
        split="test",
        n_channels=2,
        scaler=scaler,
        horizons=[1.0],
        history_duration=4.0,
        max_history_events=3,
        history_subsampling="uniform_time",
        sampling_seed=19,
    )
    data = PreparedPhysicalData(
        train=dataset,
        validation=dataset,
        test=dataset,
        scaler=scaler,
        kind="multivariate",
        preset="diagnostics",
        dataset_id="diagnostics_0000",
        n_channels=2,
        max_history_events=3,
        row_counts={"train": 4, "validation": 4, "test": 4},
    )

    rows = evaluate_prediction_rows(
        None,
        DataLoader(dataset, batch_size=1, collate_fn=PhysicalCollate()),
        data,
        model_name="Persistence",
        seed=2,
        device="cpu",
        timestamp_ablations=("real", "real_no_density"),
        history_duration=4.0,
    )

    # El alias histórico se acepta, pero los resultados usan el nombre honesto.
    assert set(rows["Ablation"]) == {"real", "real_no_count_feature"}
    real = rows[rows.Ablation == "real"].set_index("Channel")
    assert real.loc[0, "history_events"] == 3
    assert real.loc[0, "density"] == 0.75
    assert real.loc[0, "max_gap"] == 2.0
    assert real.loc[0, "median_gap"] == 2.0
    assert real.loc[0, "last_observation_age"] == 0.0
    assert real.loc[1, "history_events"] == 1
    assert real.loc[1, "density"] == 0.25
    assert real.loc[1, "max_gap"] == 0.0
    assert real.loc[1, "median_gap"] == 0.0
    assert real.loc[1, "last_observation_age"] == 4.0
    assert set(real["global_history_events"]) == {4}


def test_gaussian_evaluation_reports_nll_crps_and_coverage() -> None:
    data = _prepared()
    loader = DataLoader(
        Subset(data.test, list(range(3))),
        batch_size=3,
        collate_fn=PhysicalCollate(),
    )
    model = build_model("QueryCross-Gaussian", data, _args()).eval()
    rows = evaluate_prediction_rows(
        model,
        loader,
        data,
        model_name="QueryCross-Gaussian",
        seed=4,
        device="cpu",
        timestamp_ablations=("real",),
        history_duration=4.0,
    )
    overall = summarize_predictions(rows)
    overall = overall[overall.Scope == "overall"].iloc[0]

    assert np.isfinite(overall["nll_z"])
    assert np.isfinite(overall["crps_z"])
    assert 0.0 <= overall["coverage_90"] <= 1.0
    assert 0.0 <= overall["coverage_95"] <= 1.0

    trainer_metrics = trainer_metrics_from_prediction_rows(rows)
    assert np.isfinite(trainer_metrics["test_nll"])
    assert trainer_metrics["test_loss"] == trainer_metrics["test_nll"]
    assert 0.0 <= trainer_metrics["test_coverage_50"] <= 1.0


def test_runner_trains_one_tiny_epoch_with_single_test_forward_loop(
    tmp_path, monkeypatch
) -> None:
    data = _prepared()
    data.train = Subset(data.train, list(range(4)))
    data.validation = Subset(data.validation, list(range(4)))
    data.test = Subset(data.test, list(range(4)))
    args = _args()
    args.batch_size = 2
    args.num_workers = 0
    args.device = "cpu"
    args.learning_rate = 1e-3
    args.weight_decay = 0.0
    args.epochs = 1
    args.early_stopping_patience = 0
    args.no_checkpoints = True
    args.validate_only = False
    args.timestamp_ablations = ("real",)
    args.history_duration = 4.0

    def fail_on_second_evaluation(*_args, **_kwargs):
        raise AssertionError("run_model no debe invocar Trainer.evaluate_on_loader")

    monkeypatch.setattr(Trainer, "evaluate_on_loader", fail_on_second_evaluation)

    result, rows, metrics = run_model(
        "QueryOnly", data, args, seed=7, run_dir=tmp_path
    )

    assert result["epochs_run"] == 1
    assert result["n_params"] > 0
    assert result["train_time_s"] >= 0.0
    assert result["eval_time_s"] >= 0.0
    assert result["metric_derivation_time_s"] >= 0.0
    assert np.isfinite(result["test_rmse_z"])
    assert np.isclose(result["trainer_test_rmse"], result["test_rmse_z"])
    assert "trainer_metric_eval_time_s" not in result
    assert not rows.empty
    assert not metrics.empty
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["checkpoint_selection"] == "val_rmse"
    assert history["history"]["train_loss"]
