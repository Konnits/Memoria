from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from ts_transformer.data.collate import build_collate_fn
from ts_transformer.data.sequence_builder import SequenceBuilder
from ts_transformer.data.timeseries_dataset import (
    EventTimeSeriesDataset,
    TimeSeriesDataset,
    TimeSeriesDatasetConfig,
)
from ts_transformer.training.train_loop import Trainer


def _column(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def test_absolute_time_stays_float64_until_window_is_relative() -> None:
    base = 1_700_000_000.0
    relative = np.asarray([0.0, 0.001, 0.003, 0.007, 0.012], dtype=np.float64)
    timestamps = base + relative
    values = _column(relative)
    config = TimeSeriesDatasetConfig(history_length=3, target_offset=0)
    dataset = TimeSeriesDataset(
        values,
        timestamps,
        config,
        input_dim=1,
        output_dim=1,
        targets=values,
        sequence_builder=SequenceBuilder(input_dim=1),
    )

    assert dataset.timestamps.dtype == torch.float64
    sample = dataset[0]
    assert sample["input_timestamps"].dtype == torch.float32
    assert sample["absolute_target_timestamps"].dtype == torch.float64
    assert torch.all(sample["input_timestamps"][1:] > sample["input_timestamps"][:-1])
    assert torch.allclose(
        sample["input_timestamps"],
        torch.tensor([0.0, 0.001, 0.003, 0.007]),
        atol=2e-7,
        rtol=0.0,
    )


def test_physical_horizon_next_uses_a_real_observation_and_its_time() -> None:
    timestamps = np.asarray([0.0, 0.4, 1.1, 1.9, 3.0], dtype=np.float64)
    targets = _column([0.0, 4.0, 11.0, 19.0, 30.0])
    config = TimeSeriesDatasetConfig(
        history_length=2,
        target_horizon_choices=[0.5, 1.0],
        num_targets=2,
        target_match_mode="next",
    )
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        config,
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    sample = dataset[0]
    assert torch.allclose(
        sample["target_timestamp"], torch.tensor([1.1, 1.9], dtype=torch.float64)
    )
    assert torch.equal(sample["target_values"], torch.tensor([[11.0], [19.0]]))
    assert torch.allclose(
        sample["requested_target_horizons"],
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    assert torch.allclose(
        sample["target_horizons"],
        torch.tensor([0.7, 1.5], dtype=torch.float64),
    )


def test_continuous_queries_are_identifiable_and_linearly_interpolated() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    targets = _column(timestamps)
    config = TimeSeriesDatasetConfig(
        history_length=2,
        target_horizon_choices=[0.5, 1.5],
        num_targets=2,
        target_match_mode="linear",
    )
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        config,
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    sample = dataset[0]
    # La historia es idéntica para ambas consultas; sólo cambia t_query y, por
    # tanto, la verdad. Un modelo que ignore el tiempo no puede resolver ambas.
    assert torch.equal(sample["past_values"], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(sample["target_timestamp"], torch.tensor([1.5, 2.5]))
    assert torch.allclose(sample["target_values"][:, 0], torch.tensor([1.5, 2.5]))
    assert torch.equal(sample["target_loss_mask"], torch.ones(2, 1))


def test_event_linear_interpolation_uses_each_sensors_valid_observations() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    values = np.asarray(
        [[0.0, np.nan], [np.nan, 10.0], [2.0, np.nan], [np.nan, 30.0], [4.0, np.nan]],
        dtype=np.float32,
    )
    config = TimeSeriesDatasetConfig(
        history_length=2,
        target_horizon_choices=[0.5],
        target_match_mode="linear",
    )
    dataset = EventTimeSeriesDataset(
        values,
        timestamps,
        values,
        config,
        input_dim=2,
        output_dim=2,
    )

    sample = dataset[0]
    assert torch.equal(sample["target_timestamp"], torch.tensor([1.5]))
    assert torch.allclose(sample["target_values"], torch.tensor([[1.5, 15.0]]))
    assert torch.equal(sample["target_loss_mask"], torch.ones(1, 2))
    assert torch.equal(
        sample["target_source_timestamps"][0],
        torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float64),
    )


def test_physical_queries_can_use_an_independent_clean_truth_grid() -> None:
    observation_timestamps = np.asarray([0.0, 0.9, 2.2, 4.0], dtype=np.float64)
    noisy_observations = _column([100.0, -100.0, 100.0, -100.0])
    truth_timestamps = np.arange(0.0, 5.0, 0.5, dtype=np.float64)
    clean_truth = _column(2.0 * truth_timestamps + 1.0)
    dataset = TimeSeriesDataset(
        noisy_observations,
        observation_timestamps,
        TimeSeriesDatasetConfig(
            history_length=2,
            target_horizon_choices=[0.6],
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=clean_truth,
        target_timestamps=truth_timestamps,
    )

    sample = dataset[0]
    assert torch.allclose(
        sample["target_timestamp"], torch.tensor([1.5], dtype=torch.float64)
    )
    assert torch.allclose(sample["target_values"], torch.tensor([[4.0]]))
    assert torch.equal(
        sample["target_source_timestamps"],
        torch.tensor([[[1.5, 1.5]]], dtype=torch.float64),
    )


def test_sampled_horizons_and_query_permutation_are_reproducible_and_aligned() -> None:
    timestamps = np.arange(12, dtype=np.float64)
    targets = _column(timestamps)
    config = TimeSeriesDatasetConfig(
        history_length=2,
        target_horizon_min=0.25,
        target_horizon_max=4.0,
        target_horizon_sampling="log_uniform",
        target_match_mode="linear",
        num_targets=4,
        randomize_query_order=True,
        sampling_seed=91,
    )
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        config,
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    first = dataset[0]
    repeated = dataset[0]
    assert torch.equal(first["target_timestamp"], repeated["target_timestamp"])
    assert torch.equal(first["target_values"], repeated["target_values"])
    assert torch.allclose(first["target_values"][:, 0], first["target_timestamp"].float())

    dataset.set_epoch(1)
    next_epoch = dataset[0]
    assert not torch.equal(first["target_timestamp"], next_epoch["target_timestamp"])


def test_event_offsets_remain_backward_compatible() -> None:
    timestamps = np.asarray([0.0, 0.2, 0.5, 1.4, 2.0, 4.0], dtype=np.float64)
    targets = _column([0, 1, 2, 3, 4, 5])
    config = TimeSeriesDatasetConfig(
        history_length=2,
        target_offset_choices=[0, 2],
        num_targets=2,
    )
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        config,
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    sample = dataset[0]
    assert torch.equal(sample["target_timestamp"], torch.tensor([0.5, 2.0]))
    assert torch.equal(sample["target_values"], torch.tensor([[2.0], [4.0]]))


def test_collate_keeps_absolute_audit_timestamps_in_float64() -> None:
    timestamps = np.arange(8, dtype=np.float64) + 1_700_000_000.0
    targets = _column(np.arange(8))
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=2,
            target_horizon_choices=[1.0, 2.0],
            num_targets=2,
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
        sequence_builder=SequenceBuilder(input_dim=1),
    )
    batch = build_collate_fn()([dataset[0], dataset[1]])

    assert batch["input_timestamps"].dtype == torch.float32
    assert batch["absolute_target_timestamps"].dtype == torch.float64
    assert batch["requested_target_timestamps"].dtype == torch.float64
    assert batch["target_source_timestamps"].dtype == torch.float64
    assert batch["target_horizons"].shape == (2, 2)


def test_physical_history_duration_is_covered_with_a_bounded_number_of_events() -> None:
    timestamps = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 1.0, 2.0, 4.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        dtype=np.float64,
    )
    targets = _column(timestamps)
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=512,
            history_duration=8.0,
            max_history_events=4,
            history_subsampling="uniform_time",
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    assert isinstance(dataset._example_indices, range)
    sample = dataset[0]
    assert sample["past_timestamps"].numel() == 4
    assert sample["past_timestamps"][0] == 0.0
    assert sample["past_timestamps"][-1] == 8.0
    assert sample["past_timestamps"][-1] - sample["past_timestamps"][0] == 8.0
    # Los cuatro tokens representan las nueve filas originales del intervalo.
    assert sample["past_observation_counts"].sum() == 9.0


def test_random_physical_history_subsampling_is_reproducible_by_epoch() -> None:
    timestamps = np.linspace(0.0, 20.0, 201, dtype=np.float64)
    targets = _column(timestamps)
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=32,
            history_duration=10.0,
            max_history_events=8,
            history_subsampling="random",
            target_horizon_choices=[1.0],
            target_match_mode="linear",
            sampling_seed=123,
            cache_deterministic_history=True,
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
    )

    first = dataset[0]["past_timestamps"]
    assert not dataset._history_cache_enabled
    assert torch.equal(first, dataset[0]["past_timestamps"])
    assert first[-1] - first[0] == 10.0
    dataset.set_epoch(1)
    assert not torch.equal(first, dataset[0]["past_timestamps"])


def test_dense_history_cache_reuses_only_history_across_epochs(monkeypatch) -> None:
    timestamps = np.linspace(0.0, 30.0, 601, dtype=np.float64)
    targets = _column(timestamps)
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=64,
            history_duration=10.0,
            max_history_events=12,
            history_subsampling="uniform_time",
            target_horizon_min=0.25,
            target_horizon_max=4.0,
            target_horizon_sampling="log_uniform",
            num_targets=4,
            target_match_mode="linear",
            randomize_query_order=True,
            sampling_seed=321,
            cache_deterministic_history=True,
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
    )
    calls = 0
    original = dataset._subsample_history_positions

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_subsample_history_positions", counted)
    first = dataset[0]
    repeated = dataset[0]
    dataset.set_epoch(1)
    next_epoch = dataset[0]

    assert dataset._history_cache_enabled
    assert len(dataset._dense_history_cache) == 1
    assert calls == 1
    assert torch.equal(first["past_timestamps"], repeated["past_timestamps"])
    assert torch.equal(first["past_timestamps"], next_epoch["past_timestamps"])
    assert torch.equal(
        first["past_observation_counts"], next_epoch["past_observation_counts"]
    )
    assert not torch.equal(first["target_timestamp"], next_epoch["target_timestamp"])


def test_long_event_history_cache_keeps_queries_epoch_dependent(monkeypatch) -> None:
    observation_times = np.linspace(0.0, 30.0, 1201, dtype=np.float64)
    sensor_ids = np.arange(observation_times.size, dtype=np.int64) % 3
    values = _column(np.sin(observation_times))
    truth_times = np.linspace(0.0, 35.0, 1401, dtype=np.float64)
    truth = np.stack(
        [np.sin(truth_times + phase) for phase in (0.0, 0.2, 0.4)], axis=1
    ).astype(np.float32)
    dataset = EventTimeSeriesDataset(
        values,
        observation_times,
        truth,
        TimeSeriesDatasetConfig(
            history_length=64,
            history_duration=10.0,
            max_history_events=15,
            history_subsampling="uniform_index",
            target_horizon_min=0.25,
            target_horizon_max=4.0,
            target_horizon_sampling="log_uniform",
            num_targets=4,
            target_match_mode="linear",
            randomize_query_order=True,
            sampling_seed=654,
            cache_deterministic_history=True,
        ),
        input_dim=3,
        output_dim=3,
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([20.0], dtype=np.float64),
        event_sensor_ids=sensor_ids,
    )
    calls = 0
    original = dataset._subsample_event_positions

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_subsample_event_positions", counted)
    first = dataset[0]
    dataset.set_epoch(1)
    next_epoch = dataset[0]

    assert dataset._event_history_cache_enabled
    assert len(dataset._event_history_cache) == 1
    assert calls == 1
    for key in (
        "past_values",
        "past_timestamps",
        "past_sensor_ids",
        "past_observation_counts",
        "past_sensor_observation_count",
        "past_sensor_max_gap",
        "sensor_last_observation_age",
    ):
        assert torch.equal(first[key], next_epoch[key]), key
    assert not torch.equal(first["target_timestamp"], next_epoch["target_timestamp"])


def test_trainer_propagates_epoch_through_subset_wrappers() -> None:
    timestamps = np.linspace(0.0, 20.0, 201, dtype=np.float64)
    targets = _column(timestamps)
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=32,
            history_duration=10.0,
            max_history_events=8,
            history_subsampling="random",
            target_horizon_min=0.25,
            target_horizon_max=4.0,
            target_horizon_sampling="log_uniform",
            sampling_seed=123,
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
    )
    loader = DataLoader(Subset(dataset, [0]), batch_size=1)

    Trainer._set_loader_epoch(loader, 7)

    assert dataset._epoch == 7


def test_collate_exposes_density_mass_for_subsampled_history() -> None:
    timestamps = np.linspace(0.0, 12.0, 121, dtype=np.float64)
    targets = _column(timestamps)
    dataset = TimeSeriesDataset(
        targets,
        timestamps,
        TimeSeriesDatasetConfig(
            history_length=16,
            history_duration=4.0,
            max_history_events=5,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=targets,
        sequence_builder=SequenceBuilder(input_dim=1),
    )
    batch = build_collate_fn()([dataset[0], dataset[1]])

    counts = batch["input_observation_counts"]
    assert counts.shape == batch["input_timestamps"].shape
    assert torch.all(counts[:, :-1] > 0.0)
    assert torch.all(counts[:, -1] == 0.0)  # token de consulta
    assert torch.all(batch["past_original_observation_count"] > 5)
    assert torch.allclose(
        batch["past_original_max_gap"],
        torch.full((2,), 0.1, dtype=torch.float64),
        atol=1e-12,
    )


def test_event_history_limit_caps_tokens_not_only_timestamp_rows() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    values = np.tile(np.arange(3, dtype=np.float32), (8, 1))
    dataset = EventTimeSeriesDataset(
        values,
        timestamps,
        values,
        TimeSeriesDatasetConfig(
            history_length=16,
            history_duration=4.0,
            max_history_events=5,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=3,
        output_dim=3,
    )

    sample = dataset[0]
    assert sample["past_values"].shape[0] == 5
    assert sample["past_sensor_ids"].shape[0] == 5
    assert sample["past_timestamps"][-1] == 4.0


def test_event_subsampling_preserves_last_observation_of_each_sensor() -> None:
    timestamps = np.arange(21, dtype=np.float64)
    values = np.full((21, 3), np.nan, dtype=np.float32)
    values[:, 0] = np.arange(21, dtype=np.float32)
    values[9, 1] = 90.0
    values[1, 2] = 20.0
    targets = np.tile(np.arange(21, dtype=np.float32)[:, None], (1, 3))
    dataset = EventTimeSeriesDataset(
        values,
        timestamps,
        targets,
        TimeSeriesDatasetConfig(
            history_length=16,
            history_duration=10.0,
            max_history_events=5,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=3,
        output_dim=3,
    )

    sample = dataset[0]

    assert set(sample["past_sensor_ids"].tolist()) == {0, 1, 2}
    sensor_two = sample["past_sensor_ids"] == 2
    assert sample["past_timestamps"][sensor_two].item() == 1.0
    assert sample["past_observation_counts"].sum().item() == 13.0


def test_long_event_stream_matches_wide_nan_representation() -> None:
    wide_timestamps = np.asarray([6.0, 8.0, 10.0], dtype=np.float64)
    wide_values = np.asarray(
        [[60.0, 600.0], [80.0, np.nan], [100.0, np.nan]],
        dtype=np.float32,
    )
    long_timestamps = np.asarray([6.0, 6.0, 8.0, 10.0], dtype=np.float64)
    long_values = _column([60.0, 600.0, 80.0, 100.0])
    long_sensor_ids = np.asarray([0, 1, 0, 0], dtype=np.int64)
    truth_timestamps = np.arange(0.0, 13.0, dtype=np.float64)
    truth_values = np.stack(
        [10.0 * truth_timestamps, 100.0 * truth_timestamps], axis=1
    ).astype(np.float32)
    config = TimeSeriesDatasetConfig(
        history_length=8,
        history_duration=4.0,
        max_history_events=8,
        target_horizon_choices=[1.0],
        target_match_mode="linear",
    )
    common = dict(
        targets=truth_values,
        config=config,
        input_dim=2,
        output_dim=2,
        target_timestamps=truth_timestamps,
        forecast_origin_timestamps=np.asarray([10.0], dtype=np.float64),
    )
    wide = EventTimeSeriesDataset(
        wide_values,
        wide_timestamps,
        **common,
    )
    long = EventTimeSeriesDataset(
        long_values,
        long_timestamps,
        event_sensor_ids=long_sensor_ids,
        **common,
    )

    wide_sample = wide[0]
    long_sample = long[0]
    assert long.values.shape == (4, 1)
    assert long.get_approx_lengths() == wide.get_approx_lengths() == [5]
    for key in (
        "past_values",
        "past_timestamps",
        "past_sensor_ids",
        "past_observation_counts",
        "target_timestamp",
        "target_values",
        "past_sensor_observation_count",
        "past_sensor_max_gap",
        "past_sensor_median_gap",
        "sensor_last_observation_age",
    ):
        assert torch.equal(long_sample[key], wide_sample[key]), key


def test_sensor_diagnostics_are_exact_before_event_subsampling_and_collate() -> None:
    timestamps = np.asarray([6.0, 6.0, 8.0, 10.0], dtype=np.float64)
    sensor_ids = np.asarray([0, 1, 0, 0], dtype=np.int64)
    truth_timestamps = np.arange(0.0, 13.0, dtype=np.float64)
    truth_values = np.stack(
        [truth_timestamps, 10.0 * truth_timestamps], axis=1
    ).astype(np.float32)
    dataset = EventTimeSeriesDataset(
        _column([6.0, 60.0, 8.0, 10.0]),
        timestamps,
        truth_values,
        TimeSeriesDatasetConfig(
            history_length=8,
            history_duration=4.0,
            max_history_events=3,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=2,
        output_dim=2,
        event_sensor_ids=sensor_ids,
        target_timestamps=truth_timestamps,
        forecast_origin_timestamps=np.asarray([10.0], dtype=np.float64),
        sequence_builder=SequenceBuilder(
            input_dim=1,
            use_sensor_ids=True,
            num_sensors=2,
            num_target_tokens=2,
            target_sensor_ids=[0, 1],
        ),
    )

    sample = dataset[0]
    assert int((~sample["is_target_mask"]).sum()) == 3
    assert torch.equal(
        sample["past_sensor_observation_count"],
        torch.tensor([3.0, 1.0], dtype=torch.float64),
    )
    assert torch.equal(
        sample["past_sensor_max_gap"],
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        sample["past_sensor_median_gap"],
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        sample["sensor_last_observation_age"],
        torch.tensor([0.0, 4.0], dtype=torch.float64),
    )

    batch = build_collate_fn()([sample])
    assert batch["past_sensor_observation_count"].shape == (1, 2)
    assert batch["sensor_last_observation_age"][0, 1] == 4.0


def test_preindexed_sensor_diagnostics_match_brute_force_windows() -> None:
    timestamps = np.asarray(
        [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.0, 5.0],
        dtype=np.float64,
    )
    sensor_ids = np.asarray([0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0], dtype=np.int64)
    truth_timestamps = np.arange(0.0, 8.5, 0.5, dtype=np.float64)
    dataset = EventTimeSeriesDataset(
        _column(np.linspace(-1.0, 1.0, len(timestamps))),
        timestamps,
        np.stack([truth_timestamps] * 3, axis=1).astype(np.float32),
        TimeSeriesDatasetConfig(
            history_length=16,
            history_duration=3.0,
            max_history_events=6,
            target_horizon_choices=[0.5],
            target_match_mode="linear",
            compute_history_diagnostics=True,
        ),
        input_dim=3,
        output_dim=3,
        event_sensor_ids=sensor_ids,
        target_timestamps=truth_timestamps,
        forecast_origin_timestamps=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
    )

    for idx, anchor in enumerate(dataset._example_indices):
        origin = dataset._forecast_origin(idx, int(anchor))
        start, stop = dataset._history_bounds(idx, int(anchor), origin)
        event_times = dataset.timestamps[start:stop]
        event_sensors = dataset.event_sensor_ids[start:stop]
        brute_force = dataset._sensor_diagnostics(
            event_times, event_sensors, origin
        )
        preindexed = dataset._sensor_diagnostics(
            event_times,
            event_sensors,
            origin,
            history_start=start,
            history_stop=stop,
        )
        assert brute_force.keys() == preindexed.keys()
        for key in brute_force:
            assert torch.equal(brute_force[key], preindexed[key]), (idx, key)


def test_event_diagnostics_can_be_disabled_for_train_and_validation() -> None:
    dataset = EventTimeSeriesDataset(
        _column([6.0, 60.0, 8.0, 10.0]),
        np.asarray([6.0, 6.0, 8.0, 10.0], dtype=np.float64),
        np.stack([np.arange(13), np.arange(13)], axis=1).astype(np.float32),
        TimeSeriesDatasetConfig(
            history_length=8,
            history_duration=4.0,
            max_history_events=3,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
            compute_history_diagnostics=False,
        ),
        input_dim=2,
        output_dim=2,
        event_sensor_ids=np.asarray([0, 1, 0, 0], dtype=np.int64),
        target_timestamps=np.arange(13, dtype=np.float64),
        forecast_origin_timestamps=np.asarray([10.0], dtype=np.float64),
    )

    sample = dataset[0]
    assert "past_original_median_gap" not in sample
    assert "past_sensor_observation_count" not in sample
    assert "sensor_last_observation_age" not in sample


def test_explicit_forecast_origin_inside_gap_uses_history_ending_at_origin() -> None:
    observation_times = np.asarray([0.0, 1.0, 2.0, 8.0, 9.0], dtype=np.float64)
    observations = _column([0.0, 1.0, 2.0, 8.0, 9.0])
    truth_times = np.arange(0.0, 10.5, 0.5, dtype=np.float64)
    truth_values = _column(2.0 * truth_times)
    dataset = TimeSeriesDataset(
        observations,
        observation_times,
        TimeSeriesDatasetConfig(
            history_length=16,
            history_duration=4.0,
            max_history_events=8,
            target_horizon_choices=[1.0, 2.0],
            num_targets=2,
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=truth_values,
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([5.0], dtype=np.float64),
    )

    sample = dataset[0]
    assert torch.equal(
        sample["past_timestamps"], torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    assert sample["forecast_origin"] == 5.0
    assert sample["last_observation_age"] == 3.0
    assert torch.equal(
        sample["target_timestamp"], torch.tensor([6.0, 7.0], dtype=torch.float64)
    )
    assert torch.equal(sample["target_horizons"], torch.tensor([1.0, 2.0]))
    assert torch.equal(sample["target_values"], torch.tensor([[12.0], [14.0]]))


def test_invalid_explicit_origins_are_filtered_by_history_and_target_coverage() -> None:
    observation_times = np.asarray([0.0, 1.0, 2.0, 8.0, 9.0], dtype=np.float64)
    observations = _column(observation_times)
    truth_times = np.arange(0.0, 10.5, 0.5, dtype=np.float64)
    dataset = TimeSeriesDataset(
        observations,
        observation_times,
        TimeSeriesDatasetConfig(
            history_length=8,
            history_duration=4.0,
            target_horizon_choices=[1.0, 2.0],
            num_targets=2,
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=_column(truth_times),
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([1.0, 5.0, 8.0, 9.0]),
    )

    assert len(dataset) == 2
    assert dataset.num_discarded_forecast_origins == 2
    assert dataset.forecast_origin_audit["candidate_count"] == 4
    assert dataset.forecast_origin_audit["accepted_count"] == 2
    assert dataset.forecast_origin_audit["discarded_count"] == 2
    assert dataset.forecast_origin_audit["discarded_by_cause"] == {
        "insufficient_history_coverage": 1,
        "origin_after_last_observation": 0,
        "empty_history": 0,
        "target_before_available_range": 0,
        "target_after_available_range": 1,
    }
    assert (
        dataset.forecast_origin_audit["empty_history_policy"]
        == "discard_origin_without_synthetic_observation"
    )
    assert torch.equal(
        dataset._example_forecast_origins,
        torch.tensor([5.0, 8.0], dtype=torch.float64),
    )


def test_event_dataset_supports_forecast_origin_inside_async_gap() -> None:
    observation_times = np.asarray([0.0, 1.0, 2.0, 8.0, 9.0], dtype=np.float64)
    observations = np.asarray(
        [[0.0, np.nan], [np.nan, 10.0], [2.0, np.nan], [8.0, np.nan], [np.nan, 90.0]],
        dtype=np.float32,
    )
    truth_times = np.arange(0.0, 10.5, 0.5, dtype=np.float64)
    truth_values = np.stack([truth_times, 10.0 * truth_times], axis=1).astype(
        np.float32
    )
    builder = SequenceBuilder(
        input_dim=1,
        use_sensor_ids=True,
        num_sensors=2,
        num_target_tokens=2,
        target_sensor_ids=[0, 1],
    )
    dataset = EventTimeSeriesDataset(
        observations,
        observation_times,
        truth_values,
        TimeSeriesDatasetConfig(
            history_length=8,
            history_duration=4.0,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=2,
        output_dim=2,
        sequence_builder=builder,
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([5.0], dtype=np.float64),
    )

    sample = dataset[0]
    assert torch.equal(
        sample["absolute_target_timestamps"],
        torch.tensor([6.0], dtype=torch.float64),
    )
    assert sample["forecast_origin"] == 5.0
    assert sample["last_observation_age"] == 3.0
    assert torch.equal(sample["input_timestamps"], torch.tensor([0.0, 1.0, 5.0, 5.0]))
    assert torch.equal(sample["input_sensor_ids"], torch.tensor([1, 0, 0, 1]))


def test_event_dataset_discards_empty_history_without_fabricating_an_event() -> None:
    observation_times = np.arange(6, dtype=np.float64)
    empty_events = np.full((6, 2), np.nan, dtype=np.float32)
    empty_events[4, 0] = 42.0
    truth_times = np.arange(7, dtype=np.float64)
    truth = np.stack((truth_times, truth_times), axis=1).astype(np.float32)
    dataset = EventTimeSeriesDataset(
        empty_events,
        observation_times,
        truth,
        TimeSeriesDatasetConfig(
            history_length=4,
            history_duration=2.0,
            max_history_events=2,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=2,
        output_dim=2,
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([3.0, 5.0]),
    )

    assert len(dataset) == 1
    assert torch.equal(
        dataset._example_forecast_origins,
        torch.tensor([5.0], dtype=torch.float64),
    )
    assert dataset.forecast_origin_audit["candidate_count"] == 2
    assert dataset.forecast_origin_audit["accepted_count"] == 1
    assert dataset.forecast_origin_audit["discarded_by_cause"]["empty_history"] == 1
    sample = dataset[0]
    assert torch.equal(
        sample["past_sensor_observation_count"],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(sample["past_values"], torch.tensor([[42.0]]))
    assert torch.equal(sample["past_sensor_ids"], torch.tensor([0]))


def test_explicit_forecast_origins_require_physical_horizons_and_duration() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    values = _column(timestamps)
    with pytest.raises(ValueError, match="sólo se admite con horizontes físicos"):
        TimeSeriesDataset(
            values,
            timestamps,
            TimeSeriesDatasetConfig(history_length=2),
            input_dim=1,
            output_dim=1,
            targets=values,
            forecast_origin_timestamps=np.asarray([4.0]),
        )

    with pytest.raises(ValueError, match="requiere history_duration"):
        TimeSeriesDataset(
            values,
            timestamps,
            TimeSeriesDatasetConfig(
                history_length=2,
                target_horizon_choices=[1.0],
            ),
            input_dim=1,
            output_dim=1,
            targets=values,
            forecast_origin_timestamps=np.asarray([4.0]),
        )


def test_collate_keeps_explicit_origins_ages_and_queries_aligned() -> None:
    observation_times = np.asarray([0.0, 1.0, 2.0, 8.0, 9.0], dtype=np.float64)
    values = _column(observation_times)
    truth_times = np.arange(0.0, 11.0, 0.5, dtype=np.float64)
    dataset = TimeSeriesDataset(
        values,
        observation_times,
        TimeSeriesDatasetConfig(
            history_length=8,
            history_duration=4.0,
            target_horizon_choices=[1.0],
            target_match_mode="linear",
        ),
        input_dim=1,
        output_dim=1,
        targets=_column(truth_times),
        sequence_builder=SequenceBuilder(input_dim=1),
        target_timestamps=truth_times,
        forecast_origin_timestamps=np.asarray([5.0, 8.0]),
    )

    batch = build_collate_fn()([dataset[0], dataset[1]])
    assert torch.equal(
        batch["forecast_origin"], torch.tensor([5.0, 8.0], dtype=torch.float64)
    )
    assert torch.equal(
        batch["last_observation_age"], torch.tensor([3.0, 0.0], dtype=torch.float64)
    )
    assert torch.equal(
        batch["absolute_target_timestamps"],
        torch.tensor([[6.0], [9.0]], dtype=torch.float64),
    )
    assert torch.equal(batch["target_horizons"], torch.ones(2, 1))
