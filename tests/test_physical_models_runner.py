from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset

import scripts.benchmark_physical_models as physical_runner
import ts_transformer.training.train_loop as train_loop
from scripts.benchmark_physical_models import (
    ChannelScaler,
    ExactDatasetUnit,
    PhysicalCollate,
    PreparedPhysicalData,
    _gaussian_diagnostics,
    _gaussian_diagnostics_arrays,
    _prediction_frame_for_ablation,
    ablate_batch_timestamps,
    atomic_write_json,
    basis_variant_config,
    build_model,
    build_split_dataset,
    checkpoint_selection_metric,
    completed_dataset_results_without_preparation,
    completed_run_result,
    completed_run_fingerprint,
    discover_exact_datasets,
    evaluate_prediction_rows,
    implementation_provenance,
    invalidate_result_sentinel,
    load_paired_reference_parameters,
    persistence_predictions,
    normalize_exact_dataset_units,
    query_variant_config,
    read_observation_interval,
    read_observation_intervals,
    run_artifact_manifest,
    run_configuration,
    run_configuration_fingerprint,
    run_model,
    summarize_predictions,
    trainer_metrics_from_prediction_rows,
)
from ts_transformer.training import Trainer
from ts_transformer.training.train_loop import TrainingConfig


def test_exact_dataset_units_preserve_nonlocal_protocol_indices(
    tmp_path: Path,
) -> None:
    first = tmp_path / "multivariate" / "hard_mixed" / "hard_mixed_0000"
    second = tmp_path / "univariate" / "bursty" / "bursty_0000"
    for directory in (first, second):
        directory.mkdir(parents=True)
        (directory / "observations.parquet").touch()
        (directory / "truth.parquet").touch()

    units = normalize_exact_dataset_units(
        (
            "univariate:bursty:bursty_0000:8",
            "multivariate:hard_mixed:hard_mixed_0000:1",
        )
    )
    discovered = discover_exact_datasets(tmp_path, units)

    assert [unit.protocol_index for unit, _ in discovered] == [1, 8]
    assert [spec[0:2] for _, spec in discovered] == [
        ("multivariate", "hard_mixed"),
        ("univariate", "bursty"),
    ]
    assert all(isinstance(unit, ExactDatasetUnit) for unit, _ in discovered)
    with pytest.raises(ValueError, match="protocol_index duplicados"):
        normalize_exact_dataset_units(
            (
                "univariate:bursty:bursty_0000:1",
                "multivariate:hard_mixed:hard_mixed_0000:1",
            )
        )


def test_exact_dataset_unit_cli_rejects_cartesian_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_physical_models.py",
            "--dataset-unit",
            "multivariate:hard_mixed:hard_mixed_0000:1",
            "--kinds",
            "multivariate",
        ],
    )
    parsed = physical_runner.parse_args()
    raw = physical_runner.load_config(parsed.config)

    with pytest.raises(ValueError, match="filtros cartesianos"):
        physical_runner.resolve_options(parsed, raw)


def test_torch_compile_cli_configures_training_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_physical_models.py",
            "--torch-compile",
            "--torch-compile-mode",
            "max-autotune",
            "--torch-compile-fullgraph",
        ],
    )
    parsed = physical_runner.parse_args()
    resolved = physical_runner.resolve_options(
        parsed, physical_runner.load_config(parsed.config)
    )
    config = physical_runner.training_config(resolved, checkpoint_dir=None)

    assert config.use_torch_compile is True
    assert config.torch_compile_mode == "max-autotune"
    assert config.torch_compile_fullgraph is True


def test_torch_compile_missing_triton_skips_without_scoping_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainingConfig(use_torch_compile=True)
    trainer.device = torch.device("cuda")
    trainer.model = torch.nn.Identity()
    monkeypatch.setattr(train_loop.importlib.util, "find_spec", lambda _name: None)

    trainer._compile_model_if_requested()

    assert "Triton no está disponible" in capsys.readouterr().out


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


def test_shared_parquet_scan_matches_independent_interval_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.parquet"
    times = np.linspace(0.0, 10.0, 101)
    pd.DataFrame(
        {
            "time": times,
            "value": np.sin(times),
            "split": np.where(
                times <= 6.0,
                "train",
                np.where(times <= 8.0, "validation", "test"),
            ),
            "event_index": np.arange(len(times), dtype=np.int64),
        }
    ).to_parquet(path, index=False, row_group_size=17)
    bounds = {
        "train": (0.0, 6.0),
        "validation": (4.0, 8.0),
        "test": (7.0, 10.0),
    }
    seeds = {"train": 11, "validation": 12, "test": 13}

    for cap in (None, 19):
        independent = {
            split: read_observation_interval(
                path,
                "univariate",
                interval,
                max_rows=cap,
                seed=seeds[split],
                batch_size=13,
            )
            for split, interval in bounds.items()
        }
        shared = read_observation_intervals(
            path,
            "univariate",
            bounds,
            max_rows=cap,
            seeds=seeds,
            batch_size=13,
        )
        for split in bounds:
            pd.testing.assert_frame_equal(shared[split], independent[split])


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


def test_complete_dataset_can_resume_without_reconstructing_parquet(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "physical.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    dataset_source = tmp_path / "source" / "toy_0000"
    dataset_source.mkdir(parents=True)
    observations_path = dataset_source / "observations.parquet"
    truth_path = dataset_source / "truth.parquet"
    observations_path.write_bytes(b"observations")
    truth_path.write_bytes(b"truth")
    output = tmp_path / "output"
    dataset_dir = output / "univariate" / "toy" / "toy_0000"
    dataset_dir.mkdir(parents=True)
    args = Namespace(
        config=config_path,
        horizons=(0.25, 1.0),
        train_horizon_min=0.25,
        train_horizon_max=1.0,
        train_horizon_sampling="log_uniform",
        queries_per_sample=2,
        history_duration=4.0,
        history_subsampling="uniform_time",
        cache_deterministic_history=True,
        max_observation_rows_per_split=None,
        max_train_samples=8,
        max_val_samples=4,
        max_test_samples=4,
        batch_size=2,
        num_workers=0,
        models=("Persistence",),
        seeds=(42,),
        d_model=8,
        num_heads=2,
        num_layers=1,
        cross_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=1,
        early_stopping_patience=1,
        device="cpu",
        timestamp_ablations=("real",),
        save_predictions=False,
        validate_only=False,
        no_checkpoints=False,
        deterministic=True,
    )
    scaler = ChannelScaler(np.asarray([0.0]), np.asarray([1.0]))
    data = PreparedPhysicalData(
        train=range(8),
        validation=range(4),
        test=range(4),
        scaler=scaler,
        kind="univariate",
        preset="toy",
        dataset_id="toy_0000",
        n_channels=1,
        max_history_events=8,
        row_counts={"train": 100, "validation": 50, "test": 50},
    )
    configuration = run_configuration(
        data,
        args,
        model_name="Persistence",
        seed=42,
        protocol_seed=7,
        protocol_index=5,
        source_paths=(observations_path, truth_path),
    )
    assert configuration["task"]["protocol_index"] == 5
    metadata = {
        "dataset_id": "toy_0000",
        "kind": "univariate",
        "preset": "toy",
        "horizons": [0.25, 1.0],
        "train_horizon_range": [0.25, 1.0],
        "train_horizon_sampling": "log_uniform",
        "queries_per_train_sample": 2,
        "history_duration": 4.0,
        "max_history_events": 8,
        "history_subsampling": "uniform_time",
        "cache_deterministic_history": True,
        "observation_rows": data.row_counts,
        "observation_row_cap": None,
        "forecast_origins": {
            "train": {"selected_examples_after_cap": 8},
            "validation": {"selected_examples_after_cap": 4},
            "test": {"selected_examples_after_cap": 4},
        },
        "scaler_mean": [0.0],
        "scaler_std": [1.0],
        "source_provenance": configuration["dataset"]["sources"],
        "protocol_config": configuration["protocol_config"],
    }
    atomic_write_json(dataset_dir / "data_metadata.json", metadata)
    fingerprint = run_configuration_fingerprint(configuration)
    run_dir = dataset_dir / "seed_42" / "Persistence"
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run_config.json",
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "configuration": configuration,
        },
    )
    (run_dir / "metrics.csv").write_text("rmse\n0.0\n", encoding="utf-8")
    (run_dir / "history.json").write_text("{}", encoding="utf-8")
    expected_result = {"Model": "Persistence", "test_rmse_z": 0.0}
    atomic_write_json(
        run_dir / "result.json",
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "artifacts": run_artifact_manifest(
                run_dir, ("run_config.json", "metrics.csv", "history.json")
            ),
            "result": expected_result,
        },
    )

    resumed = completed_dataset_results_without_preparation(
        ("univariate", "toy", observations_path, truth_path),
        args,
        dataset_dir=dataset_dir,
        protocol_seed=7,
        protocol_index=5,
        max_history_events=8,
    )
    assert resumed == [expected_result]

    metadata["history_duration"] = 5.0
    atomic_write_json(dataset_dir / "data_metadata.json", metadata)
    assert completed_dataset_results_without_preparation(
        ("univariate", "toy", observations_path, truth_path),
        args,
        dataset_dir=dataset_dir,
        protocol_seed=7,
        protocol_index=5,
        max_history_events=8,
    ) is None


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


def test_vectorized_gaussian_diagnostics_match_scalar_reference() -> None:
    prediction = np.asarray([0.0, -1.0, 2.0, 3.0, -5.0, 1.0, 8.0])
    target = np.asarray([0.0, 1.0, -2.0, 4.0, 2.0, -3.0, 7.0])
    log_scale = np.asarray([-25.0, -20.0, -12.0, 0.0, 8.0, 20.0, 25.0])
    channel_std = np.asarray([0.5, 1.0, 2.0, 3.0, 0.75, 4.0, 1.5])

    vectorized = _gaussian_diagnostics_arrays(
        prediction, target, log_scale, channel_std
    )
    scalar = [
        _gaussian_diagnostics(pred, truth, scale, std)
        for pred, truth, scale, std in zip(
            prediction, target, log_scale, channel_std
        )
    ]
    for column in vectorized:
        expected = np.asarray([row[column] for row in scalar], dtype=np.float64)
        if column in {"log_scale_z", "coverage_90", "coverage_95"}:
            np.testing.assert_array_equal(vectorized[column], expected)
        else:
            np.testing.assert_allclose(
                vectorized[column], expected, rtol=1e-14, atol=1e-14
            )


def _prediction_template_for_test(
    *, target_z: np.ndarray, channels: np.ndarray
) -> pd.DataFrame:
    count = len(target_z)
    return pd.DataFrame(
        {
            "Dataset_ID": ["toy_0000"] * count,
            "Kind": ["multivariate"] * count,
            "Preset": ["toy"] * count,
            "Seed": np.full(count, 42, dtype=np.int64),
            "Model": ["FixedGaussian"] * count,
            "Example": np.arange(count, dtype=np.int64) // 2,
            "Target_Index": np.arange(count, dtype=np.int64) % 2,
            "Horizon": np.linspace(0.25, 2.0, count),
            "Query_Time": np.linspace(10.25, 12.0, count),
            "Channel": channels.astype(np.int64),
            "history_events": np.arange(1, count + 1, dtype=np.int64),
            "sampled_history_events": np.arange(
                1, count + 1, dtype=np.int64
            ),
            "density": np.linspace(0.25, 1.0, count),
            "max_gap": np.linspace(4.0, 1.0, count),
            "median_gap": np.linspace(2.0, 0.5, count),
            "last_observation_age": np.linspace(0.0, 1.5, count),
            "global_history_events": np.full(count, 9, dtype=np.int64),
            "global_density": np.full(count, 2.25),
            "global_max_gap": np.full(count, 4.0),
            "global_median_gap": np.full(count, 1.0),
            "global_last_observation_age": np.full(count, 0.5),
            "target_z": target_z,
            "target": target_z,
        }
    )


def test_vectorized_prediction_frame_matches_scalar_schema_and_values() -> None:
    shape = (2, 2, 2)
    flat_positions = np.asarray([0, 2, 3, 5, 7], dtype=np.int64)
    channels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    target_z = np.asarray([0.5, -1.0, 2.0, 1.5, -0.25], dtype=np.float64)
    scaler = ChannelScaler(
        mean=np.asarray([10.0, -3.0]), std=np.asarray([2.0, 0.5])
    )
    template = _prediction_template_for_test(
        target_z=target_z, channels=channels
    )
    template["target"] = (
        target_z * scaler.std[channels] + scaler.mean[channels]
    )
    predictions = torch.tensor(
        [[[0.25, 9.0], [-0.5, 1.0]], [[8.0, 2.5], [7.0, -0.75]]],
        dtype=torch.float32,
    )
    log_scales = torch.tensor(
        [[[-25.0, 0.0], [-12.0, 8.0]], [[0.0, 20.0], [1.0, 25.0]]],
        dtype=torch.float32,
    )

    actual = _prediction_frame_for_ablation(
        template,
        flat_positions,
        shape,
        predictions,
        log_scales,
        ablation="real",
        scaler=scaler,
    )
    expected_records = []
    prediction_flat = predictions.numpy().reshape(-1)
    log_scale_flat = log_scales.numpy().reshape(-1)
    for row_index, flat_position in enumerate(flat_positions):
        record = template.iloc[row_index].to_dict()
        channel = int(record["Channel"])
        prediction_z = float(prediction_flat[flat_position])
        record.update(
            {
                "Ablation": "real",
                "prediction_z": prediction_z,
                "prediction": float(
                    scaler.inverse_channel(np.asarray(prediction_z), channel)
                ),
            }
        )
        record.update(
            _gaussian_diagnostics(
                prediction_z,
                float(record["target_z"]),
                float(log_scale_flat[flat_position]),
                float(scaler.std[channel]),
            )
        )
        expected_records.append(record)
    expected = pd.DataFrame.from_records(expected_records).loc[:, actual.columns]

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=True,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )

    point = _prediction_frame_for_ablation(
        template,
        flat_positions,
        shape,
        predictions,
        None,
        ablation="real",
        scaler=scaler,
    )
    assert point.loc[:, list(physical_runner._DISTRIBUTION_COLUMNS)].isna().all().all()


def test_evaluation_transfers_static_batch_once_per_loader_batch(monkeypatch) -> None:
    data = _prepared()
    loader = DataLoader(
        Subset(data.test, list(range(3))),
        batch_size=3,
        collate_fn=PhysicalCollate(),
    )
    model = build_model("QueryOnly", data, _args()).eval()
    transfer_calls = 0
    forward_calls = 0
    original_transfer = physical_runner._static_model_batch_to_device
    original_forward = physical_runner._forward_prepared_ablations

    def counted_transfer(*args, **kwargs):
        nonlocal transfer_calls
        transfer_calls += 1
        return original_transfer(*args, **kwargs)

    def counted_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(
        physical_runner, "_static_model_batch_to_device", counted_transfer
    )
    monkeypatch.setattr(physical_runner, "_forward_prepared_ablations", counted_forward)
    ablations = ("real", "all_equal", "regular_grid", "query_only")
    rows = evaluate_prediction_rows(
        model,
        loader,
        data,
        model_name="QueryOnly",
        seed=42,
        device="cpu",
        timestamp_ablations=ablations,
        history_duration=4.0,
    )

    assert transfer_calls == 1
    assert forward_calls == 1
    assert set(rows["Ablation"]) == set(ablations)


@torch.inference_mode()
def test_fused_ablations_match_individual_prepared_forwards() -> None:
    data = _prepared()
    loader = DataLoader(
        Subset(data.test, list(range(3))),
        batch_size=3,
        collate_fn=PhysicalCollate(),
    )
    batch = next(iter(loader))
    model = build_model("QueryOnly", data, _args()).eval()
    prepared = physical_runner._static_model_batch_to_device(batch, torch.device("cpu"))
    ablations = (
        "real",
        "real_no_count_feature",
        "all_equal",
        "all_equal_no_count_feature",
    )
    timestamps = torch.stack(
        [
            ablate_batch_timestamps(
                batch,
                ablation,
                output_dim=data.n_channels,
                seed=42 + index,
            )
            for index, ablation in enumerate(ablations)
        ],
        dim=0,
    )
    zero_temporal_features = tuple(
        ablation.endswith("_no_count_feature") for ablation in ablations
    )

    expected = torch.stack(
        [
            physical_runner._forward_prepared_model(
                model,
                prepared,
                timestamps[index],
                target_ndim=batch["target_values"].ndim,
                zero_temporal_features=zero_temporal_features[index],
            )[0]
            for index in range(len(ablations))
        ],
        dim=0,
    )
    actual, log_scales = physical_runner._forward_prepared_ablations(
        model,
        prepared,
        timestamps,
        target_ndim=batch["target_values"].ndim,
        zero_temporal_features=zero_temporal_features,
    )

    assert log_scales is None
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@torch.inference_mode()
def test_fused_evaluation_matches_single_batch_across_loader_batches() -> None:
    data = _prepared()
    model = build_model("QueryOnly", data, _args()).eval()
    ablations = ("real", "real_no_count_feature", "all_equal", "query_only")
    single_batch = DataLoader(
        data.test,
        batch_size=len(data.test),
        collate_fn=PhysicalCollate(),
    )
    multiple_batches = DataLoader(
        data.test,
        batch_size=2,
        collate_fn=PhysicalCollate(),
    )

    expected = evaluate_prediction_rows(
        model,
        single_batch,
        data,
        model_name="QueryOnly",
        seed=42,
        device="cpu",
        timestamp_ablations=ablations,
        history_duration=4.0,
    )
    actual = evaluate_prediction_rows(
        model,
        multiple_batches,
        data,
        model_name="QueryOnly",
        seed=42,
        device="cpu",
        timestamp_ablations=ablations,
        history_duration=4.0,
    )
    sort_columns = ["Ablation", "Example", "Target_Index", "Channel"]

    pd.testing.assert_frame_equal(
        actual.sort_values(sort_columns).reset_index(drop=True),
        expected.sort_values(sort_columns).reset_index(drop=True),
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_epoch_accumulates_detached_losses_on_device(monkeypatch) -> None:
    data = _prepared()
    model = build_model("QueryOnly", data, _args())
    trainer = Trainer(
        model,
        [{}, {}, {}],
        config=TrainingConfig(device="cpu", num_epochs=1),
    )
    synchronize_values: list[bool] = []

    def fake_train_step(_batch, *, synchronize: bool = True):
        synchronize_values.append(synchronize)
        return torch.tensor(2.5, device=trainer.device)

    monkeypatch.setattr(trainer, "_train_step", fake_train_step)

    assert trainer._train_one_epoch(epoch=1) == pytest.approx(2.5)
    assert synchronize_values == [False, False, False]


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


def test_summarize_reuses_invariant_bins_without_changing_results(
    monkeypatch,
) -> None:
    records = []
    for ablation in ("all_equal", "real"):
        for example in range(8):
            for channel in (0, 1):
                value = float(example + channel / 4.0)
                records.append(
                    {
                        "Dataset_ID": "toy_0000",
                        "Kind": "multivariate",
                        "Preset": "toy",
                        "Seed": 42,
                        "Model": "QueryCross",
                        "Ablation": ablation,
                        "Example": example,
                        "Target_Index": example % 2,
                        "Horizon": float(1 + example % 2),
                        "Channel": channel,
                        "density": value + 1.0,
                        "max_gap": 20.0 - value,
                        "last_observation_age": value / 2.0,
                        "prediction_z": value,
                        "target_z": value + 0.5,
                        "prediction": value * 2.0,
                        "target": (value + 0.5) * 2.0,
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
    rows = pd.DataFrame.from_records(records)
    expected = pd.concat(
        [
            summarize_predictions(rows[rows.Ablation == ablation], bins=2)
            for ablation in ("all_equal", "real")
        ],
        ignore_index=True,
    )

    original_qcut = physical_runner.pd.qcut
    qcut_calls = 0

    def counted_qcut(*args, **kwargs):
        nonlocal qcut_calls
        qcut_calls += 1
        return original_qcut(*args, **kwargs)

    monkeypatch.setattr(physical_runner.pd, "qcut", counted_qcut)
    actual = summarize_predictions(rows, bins=2)
    sort_columns = ["Ablation", "Scope", "Level"]
    actual = actual.sort_values(sort_columns).reset_index(drop=True)
    expected = expected.sort_values(sort_columns).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert qcut_calls == 9


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
