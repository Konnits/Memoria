"""High-performance extension for irregular_timeseries_generator.py.

Designed for multi-million-row synthetic irregular time-series datasets.
It keeps the same statistical model/configuration but replaces the expensive
observation path with:

* O(n) vectorized gap mapping.
* O(n) vectorized strict monotonicity enforcement.
* CDF-warp informative sampling (no k*n oversampling arrays).
* Optional Numba scan for exact two-state burst sampling.
* Preallocated NumPy arrays instead of one DataFrame per channel.
* Numeric global sorting and categorical/string-light output columns.

Place this file next to irregular_timeseries_generator.py.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .irregular_timeseries_generator import (
    DatasetBundle,
    GeneratorConfig,
    IrregularTimeSeriesGenerator,
    MultivariateLayout,
    SamplingConfig,
    SyntheticCollection,
    _allocate_channel_counts,
    _delta_t,
    _generate_gaps,
    _json_default,
    _robust_standardize,
    irregularity_metrics,
)

try:
    from numba import njit

    @njit(cache=True, nogil=True)
    def _bursty_intervals_scan(
        switch_uniforms: np.ndarray,
        exponential_unit: np.ndarray,
        p_enter: float,
        p_leave: float,
        burst_rate_ratio: float,
    ) -> np.ndarray:
        n = switch_uniforms.size
        result = np.empty(n, dtype=np.float64)
        state = 0
        for i in range(n):
            u = switch_uniforms[i]
            if state == 0:
                if u < p_enter:
                    state = 1
            else:
                if u < p_leave:
                    state = 0
            rate = burst_rate_ratio if state == 1 else 1.0
            result[i] = exponential_unit[i] / rate
        return result

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False
    _bursty_intervals_scan = None


@dataclass
class FastGenerationOptions:
    """Performance/output trade-offs.

    compact_dtypes:
        value, clean_value and deltas use float32; time remains float64.
    include_clean_value:
        Disable for training corpora when dense truth is stored separately.
    global_sort:
        Sort the asynchronous event stream by (time, channel_index).
        Disable when writing/consuming channel-partitioned data.
    compute_metrics:
        Compute exact irregularity metrics from each already-sorted channel.
    categorical_labels:
        Store dataset_id, channel and split as pandas Categoricals.
    use_numba_for_bursty:
        Exact state scan when Numba is installed. Fallback is a fast vectorized
        approximation based on overlapping burst intervals.
    """

    compact_dtypes: bool = True
    include_clean_value: bool = True
    global_sort: bool = True
    compute_metrics: bool = True
    categorical_labels: bool = True
    use_numba_for_bursty: bool = True


class FastIrregularTimeSeriesGenerator(IrregularTimeSeriesGenerator):
    """Drop-in high-performance generator for large corpora."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        *,
        options: Optional[FastGenerationOptions] = None,
    ) -> None:
        super().__init__(config)
        self.options = copy.deepcopy(options) if options is not None else FastGenerationOptions()

    def _sample_times(
        self,
        n: int,
        dense_time: np.ndarray,
        reference_signal: np.ndarray,
        sampling: SamplingConfig,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fast exact-size sampler.

        Informative sampling uses monotone inverse-CDF warping on the dense
        signal grid. Complexity is O(n + dense_steps), instead of constructing
        informative_oversampling*n candidates and doing weighted sampling
        without replacement.
        """
        sampling.validate()
        if sampling.ensure_split_coverage and n < 3 * sampling.min_observations_per_split:
            raise ValueError("n is too small for the requested min_observations_per_split.")

        base_times, selected_mode = _generate_normalized_times_fast(
            n=n,
            cfg=sampling,
            rng=rng,
            use_numba=self.options.use_numba_for_bursty,
        )
        gaps = _generate_gaps(sampling, rng, split_config=self.config.split)
        informative = (
            sampling.informative_value_strength > 0
            or sampling.informative_derivative_strength > 0
        )

        if informative:
            normalized_times = _informative_cdf_warp(
                base_quantiles=base_times,
                dense_time=dense_time,
                reference_signal=reference_signal,
                gaps=gaps,
                value_strength=sampling.informative_value_strength,
                derivative_strength=sampling.informative_derivative_strength,
            )
        else:
            normalized_times = _map_around_gaps_fast(base_times, gaps)

        horizon = sampling.time_end - sampling.time_start
        observed_times = sampling.time_start + normalized_times * horizon
        observed_times = _ensure_strictly_increasing_fast(
            observed_times,
            sampling.time_start,
            sampling.time_end,
            sampling.min_separation_fraction * horizon,
        )

        labels = self._split_codes(observed_times)
        split_counts = {
            "train": int(np.count_nonzero(labels == 0)),
            "validation": int(np.count_nonzero(labels == 1)),
            "test": int(np.count_nonzero(labels == 2)),
        }
        repaired = False
        if (
            sampling.ensure_split_coverage
            and min(split_counts.values()) < sampling.min_observations_per_split
        ):
            observed_times = self._repair_split_coverage(
                observed_times,
                gaps=gaps,
                minimum=sampling.min_observations_per_split,
                rng=rng,
            )
            labels = self._split_codes(observed_times)
            split_counts = {
                "train": int(np.count_nonzero(labels == 0)),
                "validation": int(np.count_nonzero(labels == 1)),
                "test": int(np.count_nonzero(labels == 2)),
            }
            repaired = True

        metadata = {
            "requested_mode": sampling.mode,
            "selected_mode": selected_mode,
            "informative": informative,
            "informative_method": "dense_inverse_cdf_warp" if informative else None,
            "informative_value_strength": sampling.informative_value_strength,
            "informative_derivative_strength": sampling.informative_derivative_strength,
            "candidate_count": int(n),
            "legacy_candidate_count_avoided": int(
                n * sampling.informative_oversampling if informative else n
            ),
            "gaps_normalized": gaps,
            "split_counts": split_counts,
            "sampling_attempts": 1,
            "repaired_split_coverage": repaired,
        }
        if self.options.compute_metrics:
            metadata["metrics"] = irregularity_metrics(observed_times)
        return observed_times, metadata

    def _split_codes(self, times: np.ndarray) -> np.ndarray:
        s_cfg = self.config.sampling
        split_cfg = self.config.split
        horizon = s_cfg.time_end - s_cfg.time_start
        train_end = s_cfg.time_start + split_cfg.train_fraction * horizon
        validation_end = train_end + split_cfg.validation_fraction * horizon
        return np.where(times < train_end, 0, np.where(times < validation_end, 1, 2)).astype(
            np.int8, copy=False
        )

    def _generate_univariate_dataset(
        self,
        dataset_id: str,
        n_observations: int,
        rng: np.random.Generator,
    ) -> DatasetBundle:
        dense_time, latent, dynamics_meta = self._simulate_latent_process(rng)
        clean_dense, emission_meta = self._emit_univariate(latent, dense_time, rng)
        observed_time, sampling_meta = self._sample_times(
            n=n_observations,
            dense_time=dense_time,
            reference_signal=clean_dense,
            sampling=self.config.sampling,
            rng=rng,
        )
        clean_at_obs = np.interp(observed_time, dense_time, clean_dense)
        observed_value, noise_meta = self._add_measurement_noise(clean_at_obs, rng)

        value_dtype = np.float32 if self.options.compact_dtypes else np.float64
        delta_dtype = np.float32 if self.options.compact_dtypes else np.float64
        data: Dict[str, Any] = {
            "time": observed_time,
            "value": observed_value.astype(value_dtype, copy=False),
            "delta_t": _delta_t(observed_time).astype(delta_dtype, copy=False),
            "event_index": _event_index(n_observations),
        }
        if self.options.include_clean_value:
            data["clean_value"] = clean_at_obs.astype(value_dtype, copy=False)
        observations = pd.DataFrame(data, copy=False)
        _attach_labels(
            observations,
            dataset_id=dataset_id,
            channel_codes=None,
            split_codes=self._split_codes(observed_time),
            categorical=self.options.categorical_labels,
            univariate=True,
        )
        observations = observations[
            [
                "dataset_id",
                "series_id",
                "time",
                "value",
                *(["clean_value"] if self.options.include_clean_value else []),
                "delta_t",
                "split",
                "event_index",
            ]
        ]

        truth = None
        if self.config.store_dense_truth:
            truth = pd.DataFrame(
                {
                    "time": dense_time,
                    "clean_value": clean_dense.astype(value_dtype, copy=False),
                    "split": pd.Categorical.from_codes(
                        self._split_codes(dense_time),
                        categories=["train", "validation", "test"],
                        ordered=True,
                    ),
                },
                copy=False,
            )
            truth.insert(
                0,
                "series_id",
                pd.Categorical.from_codes(
                    np.zeros(len(truth), dtype=np.int8), categories=[dataset_id]
                ),
            )
            truth.insert(
                0,
                "dataset_id",
                pd.Categorical.from_codes(
                    np.zeros(len(truth), dtype=np.int8), categories=[dataset_id]
                ),
            )

        metadata = {
            "dataset_id": dataset_id,
            "kind": "univariate",
            "n_observations": int(n_observations),
            "time_start": float(self.config.sampling.time_start),
            "time_end": float(self.config.sampling.time_end),
            "sampling": sampling_meta,
            "dynamics": dynamics_meta,
            "emission": emission_meta,
            "noise": noise_meta,
            "fast_generator": True,
        }
        if self.options.compute_metrics:
            metadata["irregularity"] = irregularity_metrics(observed_time)
        return DatasetBundle(observations, truth, metadata)

    def _generate_multivariate_dataset(
        self,
        dataset_id: str,
        n_observations: int,
        n_channels: int,
        layout: MultivariateLayout,
        min_observations_per_channel: int,
        channel_rate_concentration: float,
        rng: np.random.Generator,
    ) -> DatasetBundle:
        dense_time, latent, dynamics_meta = self._simulate_latent_process(rng)
        clean_dense, emission_meta = self._emit_multivariate(
            latent, dense_time, n_channels, rng
        )

        value_dtype = np.float32 if self.options.compact_dtypes else np.float64
        delta_dtype = np.float32 if self.options.compact_dtypes else np.float64
        channel_dtype = np.min_scalar_type(max(n_channels - 1, 0))
        channel_names = [f"x{i:02d}" for i in range(n_channels)]
        sampling_meta: Dict[str, Any] = {"layout": layout, "channels": {}}
        noise_metadata: Dict[str, Any] = {}
        channel_metrics: Dict[str, Dict[str, float]] = {}

        if layout == "asynchronous":
            counts, rate_weights = _allocate_channel_counts(
                total=n_observations,
                n_channels=n_channels,
                minimum=min_observations_per_channel,
                concentration=channel_rate_concentration,
                rng=rng,
            )
            total_rows = int(n_observations)
            sampling_meta["channel_rate_weights"] = rate_weights.tolist()
            sampling_meta["channel_counts"] = counts.tolist()
        elif layout == "shared_time":
            counts = np.full(n_channels, n_observations, dtype=np.int64)
            total_rows = int(n_observations) * n_channels
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        all_time = np.empty(total_rows, dtype=np.float64)
        all_value = np.empty(total_rows, dtype=value_dtype)
        all_clean = (
            np.empty(total_rows, dtype=value_dtype)
            if self.options.include_clean_value
            else None
        )
        all_delta_channel = np.empty(total_rows, dtype=delta_dtype)
        all_channel = np.empty(total_rows, dtype=channel_dtype)

        shared_time: Optional[np.ndarray] = None
        if layout == "shared_time":
            aggregate_reference = np.mean(_robust_standardize(clean_dense, axis=0), axis=1)
            shared_time, shared_sampling_meta = self._sample_times(
                n=n_observations,
                dense_time=dense_time,
                reference_signal=aggregate_reference,
                sampling=self.config.sampling,
                rng=rng,
            )
            sampling_meta["shared"] = shared_sampling_meta

        position = 0
        for channel in range(n_channels):
            count = int(counts[channel])
            end = position + count
            channel_name = channel_names[channel]

            if layout == "asynchronous":
                channel_sampling = copy.deepcopy(self.config.sampling)
                channel_sampling.renewal_shape = max(
                    0.20,
                    channel_sampling.renewal_shape * float(rng.lognormal(0.0, 0.25)),
                )
                channel_sampling.cluster_width_fraction *= float(rng.lognormal(0.0, 0.20))
                channel_sampling.informative_value_strength *= float(rng.uniform(0.70, 1.30))
                channel_sampling.informative_derivative_strength *= float(rng.uniform(0.70, 1.30))
                observed_time, channel_sampling_meta = self._sample_times(
                    n=count,
                    dense_time=dense_time,
                    reference_signal=clean_dense[:, channel],
                    sampling=channel_sampling,
                    rng=rng,
                )
                sampling_meta["channels"][channel_name] = channel_sampling_meta
            else:
                assert shared_time is not None
                observed_time = shared_time

            clean_at_obs = np.interp(observed_time, dense_time, clean_dense[:, channel])
            observed_value, channel_noise_meta = self._add_measurement_noise(clean_at_obs, rng)

            all_time[position:end] = observed_time
            all_value[position:end] = observed_value.astype(value_dtype, copy=False)
            if all_clean is not None:
                all_clean[position:end] = clean_at_obs.astype(value_dtype, copy=False)
            all_delta_channel[position:end] = _delta_t(observed_time).astype(
                delta_dtype, copy=False
            )
            all_channel[position:end] = channel
            noise_metadata[channel_name] = channel_noise_meta
            if self.options.compute_metrics:
                channel_metrics[channel_name] = irregularity_metrics(observed_time)
            position = end

        if self.options.global_sort:
            order = np.lexsort((all_channel, all_time))
            all_time = all_time[order]
            all_value = all_value[order]
            if all_clean is not None:
                all_clean = all_clean[order]
            all_delta_channel = all_delta_channel[order]
            all_channel = all_channel[order]

        all_delta_global = _delta_t(all_time).astype(delta_dtype, copy=False)
        split_codes = self._split_codes(all_time)
        data: Dict[str, Any] = {
            "channel_index": all_channel,
            "time": all_time,
            "value": all_value,
            "delta_t_channel": all_delta_channel,
            "delta_t_global": all_delta_global,
            "event_index": _event_index(total_rows),
        }
        if all_clean is not None:
            data["clean_value"] = all_clean
        observations = pd.DataFrame(data, copy=False)
        _attach_labels(
            observations,
            dataset_id=dataset_id,
            channel_codes=all_channel,
            split_codes=split_codes,
            categorical=self.options.categorical_labels,
            channel_names=channel_names,
        )
        observations = observations[
            [
                "dataset_id",
                "channel",
                "channel_index",
                "time",
                "value",
                *(["clean_value"] if all_clean is not None else []),
                "delta_t_channel",
                "delta_t_global",
                "split",
                "event_index",
            ]
        ]

        truth = None
        if self.config.store_dense_truth:
            truth_rows = len(dense_time) * n_channels
            truth_channel = np.repeat(np.arange(n_channels, dtype=channel_dtype), len(dense_time))
            truth_time = np.tile(dense_time, n_channels)
            truth_clean = clean_dense.T.reshape(-1).astype(value_dtype, copy=False)
            truth = pd.DataFrame(
                {
                    "channel_index": truth_channel,
                    "time": truth_time,
                    "clean_value": truth_clean,
                    "split": pd.Categorical.from_codes(
                        self._split_codes(truth_time),
                        categories=["train", "validation", "test"],
                        ordered=True,
                    ),
                },
                copy=False,
            )
            truth.insert(
                0,
                "channel",
                pd.Categorical.from_codes(truth_channel, categories=channel_names),
            )
            truth.insert(
                0,
                "dataset_id",
                pd.Categorical.from_codes(
                    np.zeros(truth_rows, dtype=np.int8), categories=[dataset_id]
                ),
            )

        metadata = {
            "dataset_id": dataset_id,
            "kind": "multivariate",
            "layout": layout,
            "n_channels": int(n_channels),
            "requested_n_observations": int(n_observations),
            "returned_rows": int(total_rows),
            "time_start": float(self.config.sampling.time_start),
            "time_end": float(self.config.sampling.time_end),
            "sampling": sampling_meta,
            "dynamics": dynamics_meta,
            "emission": emission_meta,
            "noise": noise_metadata,
            "fast_generator": True,
            "global_sort": self.options.global_sort,
            "compact_dtypes": self.options.compact_dtypes,
        }
        if self.options.compute_metrics:
            metadata["irregularity_by_channel"] = channel_metrics
        return DatasetBundle(observations, truth, metadata)


def _event_index(n: int) -> np.ndarray:
    dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    return np.arange(n, dtype=dtype)


def _attach_labels(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    channel_codes: Optional[np.ndarray],
    split_codes: np.ndarray,
    categorical: bool,
    channel_names: Optional[List[str]] = None,
    univariate: bool = False,
) -> None:
    n = len(frame)
    if categorical:
        dataset_values: Any = pd.Categorical.from_codes(
            np.zeros(n, dtype=np.int8), categories=[dataset_id]
        )
        split_values: Any = pd.Categorical.from_codes(
            split_codes,
            categories=["train", "validation", "test"],
            ordered=True,
        )
    else:
        dataset_values = np.full(n, dataset_id, dtype=object)
        split_values = np.asarray(["train", "validation", "test"], dtype=object)[split_codes]

    frame.insert(0, "dataset_id", dataset_values)
    if univariate:
        if categorical:
            series_values: Any = pd.Categorical.from_codes(
                np.zeros(n, dtype=np.int8), categories=[dataset_id]
            )
        else:
            series_values = np.full(n, dataset_id, dtype=object)
        frame.insert(1, "series_id", series_values)
    else:
        assert channel_codes is not None and channel_names is not None
        if categorical:
            channel_values: Any = pd.Categorical.from_codes(
                channel_codes.astype(np.int64, copy=False), categories=channel_names
            )
        else:
            names = np.asarray(channel_names, dtype=object)
            channel_values = names[channel_codes.astype(np.int64, copy=False)]
        frame.insert(1, "channel", channel_values)
    frame["split"] = split_values


def _generate_normalized_times_fast(
    n: int,
    cfg: SamplingConfig,
    rng: np.random.Generator,
    *,
    use_numba: bool,
) -> Tuple[np.ndarray, str]:
    if n < 2:
        raise ValueError("n must be at least 2.")
    mode = cfg.mode
    if mode == "mixed":
        mode = str(
            rng.choice(
                ["jittered_grid", "renewal", "clustered", "bursty"],
                p=[0.15, 0.35, 0.25, 0.25],
            )
        )

    if mode == "jittered_grid":
        base = np.linspace(0.0, 1.0, n)
        nominal_dt = 1.0 / max(n - 1, 1)
        base += rng.uniform(-1.0, 1.0, size=n) * cfg.jitter_fraction * nominal_dt
        base[0] = 0.0
        base[-1] = 1.0
        times = np.clip(base, 0.0, 1.0)

    elif mode == "renewal":
        intervals = rng.gamma(shape=cfg.renewal_shape, scale=1.0, size=n - 1)
        times = np.empty(n, dtype=np.float64)
        times[0] = 0.0
        np.cumsum(intervals, out=times[1:])
        times /= max(times[-1], 1e-12)

    elif mode == "clustered":
        centers = np.sort(rng.uniform(0.03, 0.97, size=cfg.n_clusters))
        probabilities = rng.dirichlet(np.ones(cfg.n_clusters))
        assignments = rng.choice(cfg.n_clusters, size=n, replace=True, p=probabilities)
        times = centers[assignments]
        times += rng.normal(scale=cfg.cluster_width_fraction, size=n)
        background = rng.random(n) < 0.08
        if np.any(background):
            times[background] = rng.uniform(0.0, 1.0, size=int(background.sum()))
        np.clip(times, 0.0, 1.0, out=times)

    elif mode == "bursty":
        unit_exp = rng.exponential(scale=1.0, size=n - 1)
        if use_numba and _NUMBA_AVAILABLE:
            switch = rng.random(n - 1)
            intervals = _bursty_intervals_scan(
                switch,
                unit_exp,
                cfg.p_enter_burst,
                cfg.p_leave_burst,
                cfg.burst_rate_ratio,
            )
        else:
            # Fast vectorized approximation: burst starts and geometric durations.
            starts_mask = rng.random(n - 1) < cfg.p_enter_burst
            starts = np.flatnonzero(starts_mask)
            difference = np.zeros(n, dtype=np.int32)
            if starts.size:
                durations = rng.geometric(cfg.p_leave_burst, size=starts.size)
                ends = np.minimum(starts + durations, n - 1)
                np.add.at(difference, starts, 1)
                np.add.at(difference, ends, -1)
            burst = np.cumsum(difference[:-1]) > 0
            rates = np.where(burst, cfg.burst_rate_ratio, 1.0)
            intervals = unit_exp / rates
        times = np.empty(n, dtype=np.float64)
        times[0] = 0.0
        np.cumsum(intervals, out=times[1:])
        times /= max(times[-1], 1e-12)

    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")

    times.sort()
    return times, mode


def _map_around_gaps_fast(
    normalized_times: np.ndarray,
    gaps: List[Tuple[float, float]],
) -> np.ndarray:
    if not gaps:
        return normalized_times

    starts: List[float] = []
    ends: List[float] = []
    cursor = 0.0
    for gap_start, gap_end in gaps:
        if gap_start > cursor:
            starts.append(cursor)
            ends.append(gap_start)
        cursor = gap_end
    if cursor < 1.0:
        starts.append(cursor)
        ends.append(1.0)

    starts_arr = np.asarray(starts, dtype=np.float64)
    lengths = np.asarray(ends, dtype=np.float64) - starts_arr
    cumulative_length = np.cumsum(lengths)
    total = cumulative_length[-1]
    target = np.clip(normalized_times, 0.0, 1.0) * total
    segment = np.searchsorted(cumulative_length, target, side="right")
    segment = np.minimum(segment, len(lengths) - 1)
    previous = np.where(segment == 0, 0.0, cumulative_length[segment - 1])
    return starts_arr[segment] + (target - previous)


def _informative_cdf_warp(
    *,
    base_quantiles: np.ndarray,
    dense_time: np.ndarray,
    reference_signal: np.ndarray,
    gaps: List[Tuple[float, float]],
    value_strength: float,
    derivative_strength: float,
) -> np.ndarray:
    dense_norm = (dense_time - dense_time[0]) / max(dense_time[-1] - dense_time[0], 1e-12)
    derivative = np.gradient(reference_signal, dense_time)
    score = value_strength * np.abs(_robust_standardize(reference_signal))
    score += derivative_strength * np.abs(_robust_standardize(derivative))
    score -= np.max(score)
    weights = np.exp(np.clip(score, -30.0, 0.0))

    if gaps:
        allowed = np.ones_like(weights, dtype=bool)
        for start, end in gaps:
            allowed &= ~((dense_norm >= start) & (dense_norm <= end))
        weights = np.where(allowed, weights, 0.0)

    if not np.any(weights > 0):
        return _map_around_gaps_fast(base_quantiles, gaps)

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    # Flat CDF sections represent forbidden gaps. Remove duplicate x values
    # before inverse interpolation.
    unique = np.concatenate(([True], np.diff(cdf) > 0))
    xp = np.concatenate(([0.0], cdf[unique]))
    fp = np.concatenate(([dense_norm[0]], dense_norm[unique]))
    if xp[-1] < 1.0:
        xp = np.append(xp, 1.0)
        fp = np.append(fp, dense_norm[-1])
    result = np.interp(base_quantiles, xp, fp)
    return np.sort(result)


def _ensure_strictly_increasing_fast(
    times: np.ndarray,
    lower: float,
    upper: float,
    min_separation: float,
) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    values = np.clip(values, lower, upper)
    values.sort()
    n = values.size
    if n <= 1:
        return values

    max_possible = (upper - lower) / (n - 1)
    separation = min(max(min_separation, np.finfo(np.float64).eps), 0.25 * max_possible)
    free_span = (upper - lower) - separation * (n - 1)
    if free_span <= 0:
        return np.linspace(lower, upper, n)

    normalized = (values - lower) / max(upper - lower, np.finfo(np.float64).eps)
    result = lower + normalized * free_span
    result += separation * np.arange(n, dtype=np.float64)
    return result


__all__ = [
    "FastGenerationOptions",
    "FastIrregularTimeSeriesGenerator",
]
