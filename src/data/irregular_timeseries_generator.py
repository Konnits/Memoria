"""Synthetic benchmark generator for irregularly sampled time series.

The module separates:
1. A continuous-time latent data-generating process.
2. An observation process that decides when each variable is measured.
3. Measurement noise, outliers and chronological train/validation/test splits.

It generates:
- Univariate irregular datasets with an exact number of observations.
- Multivariate asynchronous datasets with an exact total number of events.
- Dense, noise-free ground truth for interpolation/forecasting evaluation.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SamplingMode = Literal[
    "jittered_grid",
    "renewal",
    "clustered",
    "bursty",
    "mixed",
]
NoiseDistribution = Literal["gaussian", "student_t", "mixture"]
MultivariateLayout = Literal["asynchronous", "shared_time"]
OutputFormat = Literal["csv", "parquet"]


@dataclass
class SamplingConfig:
    """Configuration of the observation-time process."""

    mode: SamplingMode = "mixed"
    time_start: float = 0.0
    time_end: float = 100.0

    # Jittered grid
    jitter_fraction: float = 0.40

    # Renewal process: inter-arrivals ~ Gamma(shape, scale)
    renewal_shape: float = 1.0

    # Clustered sampling
    n_clusters: int = 6
    cluster_width_fraction: float = 0.025

    # Bursty sampling: two-state rate process
    burst_rate_ratio: float = 10.0
    p_enter_burst: float = 0.08
    p_leave_burst: float = 0.25

    # Structural gaps, applied after generating the base time pattern
    n_gaps: int = 0
    total_gap_fraction: float = 0.0
    min_gap_fraction: float = 0.015
    stratify_gaps_by_split: bool = False

    # Informative observation process. A value of zero is non-informative.
    informative_value_strength: float = 0.0
    informative_derivative_strength: float = 0.0
    informative_oversampling: int = 8

    # Numerical protection and chronological benchmark coverage
    min_separation_fraction: float = 1e-7
    ensure_split_coverage: bool = True
    min_observations_per_split: int = 2
    max_resample_attempts: int = 40

    def validate(self) -> None:
        if self.time_end <= self.time_start:
            raise ValueError("time_end must be greater than time_start.")
        if self.jitter_fraction < 0:
            raise ValueError("jitter_fraction must be non-negative.")
        if self.renewal_shape <= 0:
            raise ValueError("renewal_shape must be positive.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.cluster_width_fraction <= 0:
            raise ValueError("cluster_width_fraction must be positive.")
        if self.burst_rate_ratio <= 1:
            raise ValueError("burst_rate_ratio must be greater than 1.")
        if not 0 <= self.p_enter_burst <= 1:
            raise ValueError("p_enter_burst must lie in [0, 1].")
        if not 0 <= self.p_leave_burst <= 1:
            raise ValueError("p_leave_burst must lie in [0, 1].")
        if self.n_gaps < 0:
            raise ValueError("n_gaps must be non-negative.")
        if not 0 <= self.total_gap_fraction < 0.80:
            raise ValueError("total_gap_fraction must lie in [0, 0.80).")
        if self.informative_oversampling < 2:
            raise ValueError("informative_oversampling must be at least 2.")
        if self.min_observations_per_split < 0:
            raise ValueError("min_observations_per_split must be non-negative.")
        if self.max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be at least 1.")


@dataclass
class DynamicsConfig:
    """Configuration of the continuous-time latent process."""

    latent_dim: int = 5
    dense_steps: int = 4096

    # Stable linear SDE dz = A z dt + f(t) dt + L dW.
    decay_range: Tuple[float, float] = (0.08, 0.35)
    coupling_strength: float = 0.30
    diffusion_range: Tuple[float, float] = (0.015, 0.09)

    # Smooth deterministic forcing
    n_seasonal_components: Tuple[int, int] = (1, 4)
    period_fraction_range: Tuple[float, float] = (0.04, 0.60)
    seasonal_strength: float = 0.55
    trend_strength: float = 0.35

    # Nonstationarity
    n_regime_changes: Tuple[int, int] = (0, 3)
    regime_shift_strength: float = 0.45
    n_event_pulses: Tuple[int, int] = (0, 4)
    event_strength: float = 0.70
    event_width_fraction: Tuple[float, float] = (0.006, 0.04)

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be at least 1.")
        if self.dense_steps < 256:
            raise ValueError("dense_steps should be at least 256.")
        if self.decay_range[0] <= 0 or self.decay_range[1] < self.decay_range[0]:
            raise ValueError("decay_range must be positive and ordered.")
        if self.diffusion_range[0] < 0 or self.diffusion_range[1] < self.diffusion_range[0]:
            raise ValueError("diffusion_range must be non-negative and ordered.")


@dataclass
class EmissionConfig:
    """Maps latent states to observed variables."""

    nonlinear_strength: float = 0.45
    interaction_strength: float = 0.20
    channel_scale_range: Tuple[float, float] = (0.6, 2.5)
    channel_offset_std: float = 0.7
    standardize_clean_signal: bool = True


@dataclass
class NoiseConfig:
    """Measurement-noise and outlier configuration."""

    distribution: NoiseDistribution = "mixture"
    noise_scale_range: Tuple[float, float] = (0.03, 0.16)
    student_df: float = 4.0
    heteroscedastic_strength: float = 0.35
    outlier_probability: float = 0.005
    outlier_scale: float = 4.0

    def validate(self) -> None:
        if self.noise_scale_range[0] < 0 or self.noise_scale_range[1] < self.noise_scale_range[0]:
            raise ValueError("noise_scale_range must be non-negative and ordered.")
        if self.student_df <= 2:
            raise ValueError("student_df must be greater than 2 for finite variance.")
        if not 0 <= self.outlier_probability < 1:
            raise ValueError("outlier_probability must lie in [0, 1).")


@dataclass
class SplitConfig:
    """Chronological split over the continuous time horizon."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def validate(self) -> None:
        values = np.array(
            [self.train_fraction, self.validation_fraction, self.test_fraction],
            dtype=float,
        )
        if np.any(values <= 0):
            raise ValueError("All split fractions must be positive.")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("Split fractions must sum to 1.")


@dataclass
class GeneratorConfig:
    """Top-level generator configuration."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    store_dense_truth: bool = True
    seed: int = 2026

    def validate(self) -> None:
        self.sampling.validate()
        self.dynamics.validate()
        self.noise.validate()
        self.split.validate()


@dataclass
class DatasetBundle:
    """One generated dataset and its known ground truth."""

    observations: pd.DataFrame
    truth: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


@dataclass
class SyntheticCollection:
    """A collection of independent synthetic datasets."""

    datasets: List[DatasetBundle]
    kind: Literal["univariate", "multivariate"]
    config: GeneratorConfig

    def observations(self) -> pd.DataFrame:
        if not self.datasets:
            return pd.DataFrame()
        return pd.concat([x.observations for x in self.datasets], ignore_index=True)

    def truth(self) -> pd.DataFrame:
        frames = [x.truth for x in self.datasets if x.truth is not None]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for bundle in self.datasets:
            obs = bundle.observations
            if self.kind == "univariate":
                metrics = irregularity_metrics(obs["time"].to_numpy())
                metrics.update(
                    {
                        "dataset_id": bundle.metadata["dataset_id"],
                        "kind": self.kind,
                        "n_rows": len(obs),
                        "n_channels": 1,
                    }
                )
                rows.append(metrics)
            else:
                channel_metrics = []
                for _, group in obs.groupby("channel", sort=True):
                    channel_metrics.append(irregularity_metrics(group["time"].to_numpy()))
                rows.append(
                    {
                        "dataset_id": bundle.metadata["dataset_id"],
                        "kind": self.kind,
                        "n_rows": len(obs),
                        "n_channels": int(obs["channel"].nunique()),
                        "mean_cv_dt": float(np.mean([m["cv_dt"] for m in channel_metrics])),
                        "max_gap_ratio": float(
                            np.max([m["max_gap_ratio"] for m in channel_metrics])
                        ),
                        "mean_burstiness": float(
                            np.mean([m["burstiness"] for m in channel_metrics])
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def save(self, output_dir: str | Path, file_format: OutputFormat = "csv") -> None:
        """Save each dataset in a separate folder."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        collection_metadata = {
            "kind": self.kind,
            "n_datasets": len(self.datasets),
            "generator_seed": int(self.config.seed),
            "config": asdict(self.config),
        }
        serialized_metadata = json.dumps(
            collection_metadata, indent=2, default=_json_default
        )
        for metadata_path in (
            output_path / "collection_metadata.json",
            output_path / f"collection_metadata_gseed{int(self.config.seed)}.json",
        ):
            metadata_path.write_text(serialized_metadata, encoding="utf-8")
        self.summary().to_csv(output_path / "collection_summary.csv", index=False)

        for bundle in self.datasets:
            dataset_id = str(bundle.metadata["dataset_id"])
            dataset_dir = output_path / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            if file_format == "csv":
                bundle.observations.to_csv(dataset_dir / "observations.csv", index=False)
                if bundle.truth is not None:
                    bundle.truth.to_csv(dataset_dir / "truth.csv", index=False)
            elif file_format == "parquet":
                try:
                    bundle.observations.to_parquet(
                        dataset_dir / "observations.parquet", index=False
                    )
                    if bundle.truth is not None:
                        bundle.truth.to_parquet(dataset_dir / "truth.parquet", index=False)
                except ImportError as exc:
                    raise ImportError(
                        "Parquet output requires pyarrow or fastparquet. "
                        "Use file_format='csv' or install pyarrow."
                    ) from exc
            else:
                raise ValueError(f"Unsupported file_format: {file_format}")

            dataset_metadata = dict(bundle.metadata)
            dataset_metadata.setdefault("generator_seed", int(self.config.seed))
            (dataset_dir / "metadata.json").write_text(
                json.dumps(dataset_metadata, indent=2, default=_json_default),
                encoding="utf-8",
            )


class IrregularTimeSeriesGenerator:
    """Generator for univariate and multivariate irregular time series."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = copy.deepcopy(config) if config is not None else GeneratorConfig()
        self.config.validate()

    def generate_univariate_collection(
        self,
        n_datasets: int,
        n_observations: int,
        *,
        dataset_prefix: str = "uni",
    ) -> SyntheticCollection:
        """Generate n_datasets, each with exactly n_observations rows."""

        if n_datasets < 1:
            raise ValueError("n_datasets must be at least 1.")
        if n_observations < 8:
            raise ValueError("n_observations must be at least 8.")

        seed_sequence = np.random.SeedSequence(self.config.seed)
        child_sequences = seed_sequence.spawn(n_datasets)
        datasets = []
        for i, child_seed in enumerate(child_sequences):
            rng = np.random.default_rng(child_seed)
            dataset_id = f"{dataset_prefix}_{i:04d}"
            datasets.append(
                self._generate_univariate_dataset(dataset_id, n_observations, rng)
            )
        return SyntheticCollection(datasets, "univariate", copy.deepcopy(self.config))

    def generate_multivariate_collection(
        self,
        n_datasets: int,
        n_observations: int,
        n_channels: int,
        *,
        layout: MultivariateLayout = "asynchronous",
        min_observations_per_channel: int = 8,
        channel_rate_concentration: float = 3.0,
        dataset_prefix: str = "multi",
    ) -> SyntheticCollection:
        """Generate multivariate datasets.

        asynchronous:
            n_observations is the exact TOTAL number of event rows across channels.
        shared_time:
            n_observations is the exact number of shared timestamps. The returned
            data is still in long format and therefore has n_observations*n_channels
            rows.
        """

        if n_datasets < 1:
            raise ValueError("n_datasets must be at least 1.")
        if n_channels < 2:
            raise ValueError("n_channels must be at least 2.")
        if n_observations < 8:
            raise ValueError("n_observations must be at least 8.")
        if layout == "asynchronous" and n_observations < n_channels * min_observations_per_channel:
            raise ValueError(
                "For asynchronous data, n_observations must be at least "
                "n_channels * min_observations_per_channel."
            )
        if channel_rate_concentration <= 0:
            raise ValueError("channel_rate_concentration must be positive.")

        seed_sequence = np.random.SeedSequence(self.config.seed)
        child_sequences = seed_sequence.spawn(n_datasets)
        datasets = []
        for i, child_seed in enumerate(child_sequences):
            rng = np.random.default_rng(child_seed)
            dataset_id = f"{dataset_prefix}_{i:04d}"
            datasets.append(
                self._generate_multivariate_dataset(
                    dataset_id=dataset_id,
                    n_observations=n_observations,
                    n_channels=n_channels,
                    layout=layout,
                    min_observations_per_channel=min_observations_per_channel,
                    channel_rate_concentration=channel_rate_concentration,
                    rng=rng,
                )
            )
        return SyntheticCollection(datasets, "multivariate", copy.deepcopy(self.config))

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

        observations = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "series_id": dataset_id,
                "time": observed_time,
                "value": observed_value,
                "clean_value": clean_at_obs,
            }
        )
        observations["delta_t"] = _delta_t(observations["time"].to_numpy())
        observations["split"] = self._split_labels(observations["time"].to_numpy())
        observations["event_index"] = np.arange(len(observations), dtype=int)

        truth = None
        if self.config.store_dense_truth:
            truth = pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "series_id": dataset_id,
                    "time": dense_time,
                    "clean_value": clean_dense,
                    "split": self._split_labels(dense_time),
                }
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
            "irregularity": irregularity_metrics(observed_time),
        }
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

        frames: List[pd.DataFrame] = []
        sampling_meta: Dict[str, Any] = {"layout": layout, "channels": {}}
        noise_metadata: Dict[str, Any] = {}

        if layout == "asynchronous":
            counts, rate_weights = _allocate_channel_counts(
                total=n_observations,
                n_channels=n_channels,
                minimum=min_observations_per_channel,
                concentration=channel_rate_concentration,
                rng=rng,
            )
            sampling_meta["channel_rate_weights"] = rate_weights.tolist()
            sampling_meta["channel_counts"] = counts.tolist()

            for channel in range(n_channels):
                # Small channel-specific changes make observation rates and patterns heterogeneous.
                channel_sampling = copy.deepcopy(self.config.sampling)
                channel_sampling.renewal_shape = max(
                    0.20,
                    channel_sampling.renewal_shape * float(rng.lognormal(0.0, 0.25)),
                )
                channel_sampling.cluster_width_fraction *= float(
                    rng.lognormal(0.0, 0.20)
                )
                channel_sampling.informative_value_strength *= float(
                    rng.uniform(0.70, 1.30)
                )
                channel_sampling.informative_derivative_strength *= float(
                    rng.uniform(0.70, 1.30)
                )

                observed_time, channel_sampling_meta = self._sample_times(
                    n=int(counts[channel]),
                    dense_time=dense_time,
                    reference_signal=clean_dense[:, channel],
                    sampling=channel_sampling,
                    rng=rng,
                )
                clean_at_obs = np.interp(
                    observed_time, dense_time, clean_dense[:, channel]
                )
                observed_value, channel_noise_meta = self._add_measurement_noise(
                    clean_at_obs, rng
                )

                channel_name = f"x{channel:02d}"
                frame = pd.DataFrame(
                    {
                        "dataset_id": dataset_id,
                        "channel": channel_name,
                        "channel_index": channel,
                        "time": observed_time,
                        "value": observed_value,
                        "clean_value": clean_at_obs,
                    }
                )
                frame["delta_t_channel"] = _delta_t(frame["time"].to_numpy())
                frames.append(frame)
                sampling_meta["channels"][channel_name] = channel_sampling_meta
                noise_metadata[channel_name] = channel_noise_meta

        elif layout == "shared_time":
            aggregate_reference = np.mean(_robust_standardize(clean_dense, axis=0), axis=1)
            shared_time, shared_sampling_meta = self._sample_times(
                n=n_observations,
                dense_time=dense_time,
                reference_signal=aggregate_reference,
                sampling=self.config.sampling,
                rng=rng,
            )
            sampling_meta["shared"] = shared_sampling_meta

            for channel in range(n_channels):
                clean_at_obs = np.interp(
                    shared_time, dense_time, clean_dense[:, channel]
                )
                observed_value, channel_noise_meta = self._add_measurement_noise(
                    clean_at_obs, rng
                )
                channel_name = f"x{channel:02d}"
                frame = pd.DataFrame(
                    {
                        "dataset_id": dataset_id,
                        "channel": channel_name,
                        "channel_index": channel,
                        "time": shared_time,
                        "value": observed_value,
                        "clean_value": clean_at_obs,
                    }
                )
                frame["delta_t_channel"] = _delta_t(frame["time"].to_numpy())
                frames.append(frame)
                noise_metadata[channel_name] = channel_noise_meta
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        observations = pd.concat(frames, ignore_index=True)
        observations = observations.sort_values(
            ["time", "channel_index"], kind="mergesort"
        ).reset_index(drop=True)
        observations["delta_t_global"] = _delta_t(observations["time"].to_numpy())
        observations["split"] = self._split_labels(observations["time"].to_numpy())
        observations["event_index"] = np.arange(len(observations), dtype=int)

        truth = None
        if self.config.store_dense_truth:
            truth_frames = []
            for channel in range(n_channels):
                truth_frames.append(
                    pd.DataFrame(
                        {
                            "dataset_id": dataset_id,
                            "channel": f"x{channel:02d}",
                            "channel_index": channel,
                            "time": dense_time,
                            "clean_value": clean_dense[:, channel],
                            "split": self._split_labels(dense_time),
                        }
                    )
                )
            truth = pd.concat(truth_frames, ignore_index=True)

        channel_irregularity = {
            channel: irregularity_metrics(group["time"].to_numpy())
            for channel, group in observations.groupby("channel", sort=True)
        }
        metadata = {
            "dataset_id": dataset_id,
            "kind": "multivariate",
            "layout": layout,
            "n_channels": int(n_channels),
            "requested_n_observations": int(n_observations),
            "returned_rows": int(len(observations)),
            "time_start": float(self.config.sampling.time_start),
            "time_end": float(self.config.sampling.time_end),
            "sampling": sampling_meta,
            "dynamics": dynamics_meta,
            "emission": emission_meta,
            "noise": noise_metadata,
            "irregularity_by_channel": channel_irregularity,
        }
        return DatasetBundle(observations, truth, metadata)

    def _simulate_latent_process(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        cfg = self.config.dynamics
        s_cfg = self.config.sampling
        dense_time = np.linspace(s_cfg.time_start, s_cfg.time_end, cfg.dense_steps)
        dt = float(dense_time[1] - dense_time[0])
        horizon = float(s_cfg.time_end - s_cfg.time_start)
        d = cfg.latent_dim

        raw_matrix = rng.normal(size=(d, d))
        spectral_norm = max(np.linalg.norm(raw_matrix, ord=2), 1e-12)
        raw_matrix = cfg.coupling_strength * raw_matrix / spectral_norm
        decay = float(rng.uniform(*cfg.decay_range))
        max_real_eigenvalue = float(np.max(np.real(np.linalg.eigvals(raw_matrix))))
        transition_matrix = raw_matrix - (max_real_eigenvalue + decay) * np.eye(d)

        diffusion_diag = rng.uniform(*cfg.diffusion_range, size=d)
        forcing = np.zeros((cfg.dense_steps, d), dtype=float)

        n_seasonal = int(
            rng.integers(
                cfg.n_seasonal_components[0],
                cfg.n_seasonal_components[1] + 1,
            )
        )
        seasonal_params = []
        for _ in range(n_seasonal):
            period = float(
                horizon * rng.uniform(*cfg.period_fraction_range)
            )
            amplitude = cfg.seasonal_strength * rng.normal(size=d) / math.sqrt(n_seasonal)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            forcing += np.sin(2.0 * np.pi * dense_time[:, None] / period + phase) * amplitude
            seasonal_params.append(
                {"period": period, "amplitude": amplitude.tolist(), "phase": phase}
            )

        centered_time = (dense_time - dense_time.mean()) / max(horizon, 1e-12)
        trend_vector = cfg.trend_strength * rng.normal(size=d)
        forcing += centered_time[:, None] * trend_vector

        n_regime_changes = int(
            rng.integers(cfg.n_regime_changes[0], cfg.n_regime_changes[1] + 1)
        )
        regime_times: List[float] = []
        regime_shifts: List[List[float]] = []
        if n_regime_changes > 0:
            candidates = rng.uniform(
                s_cfg.time_start + 0.12 * horizon,
                s_cfg.time_end - 0.12 * horizon,
                size=n_regime_changes,
            )
            for change_time in np.sort(candidates):
                shift = cfg.regime_shift_strength * rng.normal(size=d)
                forcing[dense_time >= change_time] += shift
                regime_times.append(float(change_time))
                regime_shifts.append(shift.tolist())

        n_events = int(rng.integers(cfg.n_event_pulses[0], cfg.n_event_pulses[1] + 1))
        event_params = []
        for _ in range(n_events):
            center = float(rng.uniform(s_cfg.time_start, s_cfg.time_end))
            width = float(horizon * rng.uniform(*cfg.event_width_fraction))
            amplitude = cfg.event_strength * rng.normal(size=d)
            pulse = np.exp(-0.5 * ((dense_time - center) / width) ** 2)
            forcing += pulse[:, None] * amplitude
            event_params.append(
                {"center": center, "width": width, "amplitude": amplitude.tolist()}
            )

        latent = np.zeros((cfg.dense_steps, d), dtype=float)
        latent[0] = rng.normal(scale=0.5, size=d)
        sqrt_dt = math.sqrt(dt)
        for i in range(1, cfg.dense_steps):
            drift = transition_matrix @ latent[i - 1] + forcing[i - 1]
            diffusion = diffusion_diag * sqrt_dt * rng.normal(size=d)
            latent[i] = latent[i - 1] + drift * dt + diffusion

        metadata = {
            "latent_dim": d,
            "dense_steps": cfg.dense_steps,
            "dt": dt,
            "transition_matrix": transition_matrix.tolist(),
            "diffusion_diag": diffusion_diag.tolist(),
            "decay": decay,
            "trend_vector": trend_vector.tolist(),
            "seasonal_components": seasonal_params,
            "regime_times": regime_times,
            "regime_shifts": regime_shifts,
            "event_pulses": event_params,
        }
        return dense_time, latent, metadata

    def _emit_univariate(
        self,
        latent: np.ndarray,
        dense_time: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.config.emission
        d = latent.shape[1]
        linear_weights = rng.normal(size=d)
        linear_weights /= max(np.linalg.norm(linear_weights), 1e-12)
        nonlinear_weights = rng.normal(size=d)
        nonlinear_weights /= max(np.linalg.norm(nonlinear_weights), 1e-12)

        clean = latent @ linear_weights
        clean += cfg.nonlinear_strength * np.tanh(latent @ nonlinear_weights)
        if d >= 2:
            clean += cfg.interaction_strength * np.tanh(latent[:, 0] * latent[:, 1])

        if cfg.standardize_clean_signal:
            clean = _robust_standardize(clean)
        scale = float(rng.uniform(*cfg.channel_scale_range))
        offset = float(rng.normal(scale=cfg.channel_offset_std))
        clean = offset + scale * clean

        metadata = {
            "linear_weights": linear_weights.tolist(),
            "nonlinear_weights": nonlinear_weights.tolist(),
            "scale": scale,
            "offset": offset,
        }
        return clean, metadata

    def _emit_multivariate(
        self,
        latent: np.ndarray,
        dense_time: np.ndarray,
        n_channels: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.config.emission
        d = latent.shape[1]

        linear_weights = rng.normal(size=(n_channels, d))
        linear_weights /= np.maximum(
            np.linalg.norm(linear_weights, axis=1, keepdims=True), 1e-12
        )
        nonlinear_weights = rng.normal(size=(n_channels, d))
        nonlinear_weights /= np.maximum(
            np.linalg.norm(nonlinear_weights, axis=1, keepdims=True), 1e-12
        )

        clean = latent @ linear_weights.T
        clean += cfg.nonlinear_strength * np.tanh(latent @ nonlinear_weights.T)

        if d >= 2:
            shared_interaction = np.tanh(latent[:, 0] * latent[:, 1])
            interaction_loadings = rng.normal(size=n_channels)
            clean += (
                cfg.interaction_strength
                * shared_interaction[:, None]
                * interaction_loadings[None, :]
            )
        else:
            interaction_loadings = np.zeros(n_channels)

        # Channel-specific periodic components prevent all variables from being
        # mere linear views of the same latent trajectory.
        horizon = float(dense_time[-1] - dense_time[0])
        channel_periods = horizon * rng.uniform(0.06, 0.75, size=n_channels)
        channel_phases = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
        channel_amplitudes = rng.normal(scale=0.20, size=n_channels)
        clean += channel_amplitudes[None, :] * np.sin(
            2.0 * np.pi * dense_time[:, None] / channel_periods[None, :]
            + channel_phases[None, :]
        )

        if cfg.standardize_clean_signal:
            clean = _robust_standardize(clean, axis=0)
        scales = rng.uniform(*cfg.channel_scale_range, size=n_channels)
        offsets = rng.normal(scale=cfg.channel_offset_std, size=n_channels)
        clean = offsets[None, :] + clean * scales[None, :]

        metadata = {
            "linear_weights": linear_weights.tolist(),
            "nonlinear_weights": nonlinear_weights.tolist(),
            "interaction_loadings": interaction_loadings.tolist(),
            "channel_periods": channel_periods.tolist(),
            "channel_phases": channel_phases.tolist(),
            "channel_amplitudes": channel_amplitudes.tolist(),
            "scales": scales.tolist(),
            "offsets": offsets.tolist(),
            "clean_correlation": np.corrcoef(clean, rowvar=False).tolist(),
        }
        return clean, metadata

    def _sample_times(
        self,
        n: int,
        dense_time: np.ndarray,
        reference_signal: np.ndarray,
        sampling: SamplingConfig,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        sampling.validate()
        if (
            sampling.ensure_split_coverage
            and n < 3 * sampling.min_observations_per_split
        ):
            raise ValueError(
                "n is too small for the requested min_observations_per_split."
            )

        informative = (
            sampling.informative_value_strength > 0
            or sampling.informative_derivative_strength > 0
        )
        candidate_n = n * sampling.informative_oversampling if informative else n

        observed_times: Optional[np.ndarray] = None
        selected_mode = sampling.mode
        gaps: List[Tuple[float, float]] = []
        split_counts: Dict[str, int] = {}
        attempts_used = 0

        coverage_satisfied = False
        for attempt in range(1, sampling.max_resample_attempts + 1):
            attempts_used = attempt
            normalized_times, selected_mode = _generate_normalized_times(
                candidate_n, sampling, rng
            )
            gaps = _generate_gaps(sampling, rng, split_config=self.config.split)
            normalized_times = _map_around_gaps(normalized_times, gaps)
            candidate_times = sampling.time_start + normalized_times * (
                sampling.time_end - sampling.time_start
            )
            candidate_times = _ensure_strictly_increasing(
                candidate_times,
                sampling.time_start,
                sampling.time_end,
                sampling.min_separation_fraction
                * (sampling.time_end - sampling.time_start),
            )

            if informative:
                signal = np.interp(candidate_times, dense_time, reference_signal)
                derivative_dense = np.gradient(reference_signal, dense_time)
                derivative = np.interp(candidate_times, dense_time, derivative_dense)
                signal_score = np.abs(_robust_standardize(signal))
                derivative_score = np.abs(_robust_standardize(derivative))
                logits = (
                    sampling.informative_value_strength * signal_score
                    + sampling.informative_derivative_strength * derivative_score
                )
                logits -= np.max(logits)
                weights = np.exp(np.clip(logits, -30.0, 0.0)) + 1e-12
                weights /= weights.sum()
                chosen = rng.choice(candidate_n, size=n, replace=False, p=weights)
                proposal = np.sort(candidate_times[chosen])
            else:
                proposal = candidate_times

            proposal = _ensure_strictly_increasing(
                proposal,
                sampling.time_start,
                sampling.time_end,
                sampling.min_separation_fraction
                * (sampling.time_end - sampling.time_start),
            )
            labels = self._split_labels(proposal)
            split_counts = {
                label: int(np.sum(labels == label))
                for label in ("train", "validation", "test")
            }
            observed_times = proposal

            if not sampling.ensure_split_coverage:
                coverage_satisfied = True
                break
            if min(split_counts.values()) >= sampling.min_observations_per_split:
                coverage_satisfied = True
                break

        assert observed_times is not None
        repaired_coverage = False
        if sampling.ensure_split_coverage and not coverage_satisfied:
            observed_times = self._repair_split_coverage(
                observed_times,
                gaps=gaps,
                minimum=sampling.min_observations_per_split,
                rng=rng,
            )
            repaired_coverage = True
            labels = self._split_labels(observed_times)
            split_counts = {
                label: int(np.sum(labels == label))
                for label in ("train", "validation", "test")
            }
        if len(observed_times) != n:
            raise RuntimeError("Internal error: sampler did not return the requested size.")

        metadata = {
            "requested_mode": sampling.mode,
            "selected_mode": selected_mode,
            "informative": informative,
            "informative_value_strength": sampling.informative_value_strength,
            "informative_derivative_strength": sampling.informative_derivative_strength,
            "candidate_count": int(candidate_n),
            "gaps_normalized": gaps,
            "split_counts": split_counts,
            "sampling_attempts": attempts_used,
            "repaired_split_coverage": repaired_coverage,
            "metrics": irregularity_metrics(observed_times),
        }
        return observed_times, metadata

    def _repair_split_coverage(
        self,
        times: np.ndarray,
        *,
        gaps: List[Tuple[float, float]],
        minimum: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Move a few donor events so every chronological split is represented."""

        if minimum <= 0:
            return times

        s_cfg = self.config.sampling
        split_cfg = self.config.split
        horizon = s_cfg.time_end - s_cfg.time_start
        train_end = s_cfg.time_start + split_cfg.train_fraction * horizon
        validation_end = train_end + split_cfg.validation_fraction * horizon
        split_bounds = {
            "train": (s_cfg.time_start, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, s_cfg.time_end),
        }
        absolute_gaps = [
            (s_cfg.time_start + a * horizon, s_cfg.time_start + b * horizon)
            for a, b in gaps
        ]

        repaired = np.asarray(times, dtype=float).copy()
        for _ in range(3 * minimum + 6):
            labels = self._split_labels(repaired)
            counts = {
                label: int(np.sum(labels == label))
                for label in ("train", "validation", "test")
            }
            deficits = [label for label, count in counts.items() if count < minimum]
            if not deficits:
                result = np.sort(repaired)
                return _ensure_strictly_increasing(
                    result,
                    s_cfg.time_start,
                    s_cfg.time_end,
                    s_cfg.min_separation_fraction * horizon,
                )

            target_label = deficits[0]
            donors = [label for label, count in counts.items() if count > minimum]
            if not donors:
                raise RuntimeError("No donor split available to repair split coverage.")
            donor_label = max(donors, key=lambda label: counts[label] - minimum)
            donor_indices = np.flatnonzero(labels == donor_label)
            donor_index = int(rng.choice(donor_indices))

            allowed = _subtract_intervals(split_bounds[target_label], absolute_gaps)
            if not allowed:
                raise RuntimeError(
                    f"Structural gaps fully cover the {target_label} split; "
                    "reduce total_gap_fraction or n_gaps."
                )
            repaired[donor_index] = _sample_from_intervals(allowed, rng)

        raise RuntimeError("Could not repair chronological split coverage.")

    def _add_measurement_noise(
        self,
        clean_values: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.config.noise
        n = len(clean_values)
        base_scale = float(rng.uniform(*cfg.noise_scale_range))
        z = np.abs(_robust_standardize(clean_values))
        local_scale = base_scale * (
            1.0 + cfg.heteroscedastic_strength * z / (1.0 + z)
        )

        if cfg.distribution == "gaussian":
            epsilon = rng.normal(size=n)
        elif cfg.distribution == "student_t":
            epsilon = rng.standard_t(cfg.student_df, size=n)
            epsilon /= math.sqrt(cfg.student_df / (cfg.student_df - 2.0))
        elif cfg.distribution == "mixture":
            heavy = rng.random(n) < 0.15
            epsilon = rng.normal(size=n)
            student = rng.standard_t(cfg.student_df, size=n)
            student /= math.sqrt(cfg.student_df / (cfg.student_df - 2.0))
            epsilon[heavy] = student[heavy]
        else:
            raise ValueError(f"Unsupported noise distribution: {cfg.distribution}")

        noise = local_scale * epsilon
        outlier_mask = rng.random(n) < cfg.outlier_probability
        if np.any(outlier_mask):
            noise[outlier_mask] += (
                cfg.outlier_scale
                * base_scale
                * rng.standard_t(cfg.student_df, size=int(outlier_mask.sum()))
            )

        metadata = {
            "distribution": cfg.distribution,
            "base_scale": base_scale,
            "heteroscedastic_strength": cfg.heteroscedastic_strength,
            "n_outliers": int(outlier_mask.sum()),
        }
        return clean_values + noise, metadata

    def _split_labels(self, times: np.ndarray) -> np.ndarray:
        s_cfg = self.config.sampling
        split_cfg = self.config.split
        horizon = s_cfg.time_end - s_cfg.time_start
        train_end = s_cfg.time_start + split_cfg.train_fraction * horizon
        validation_end = train_end + split_cfg.validation_fraction * horizon
        return np.where(
            times < train_end,
            "train",
            np.where(times < validation_end, "validation", "test"),
        )


# -----------------------------------------------------------------------------
# Presets and conversion utilities
# -----------------------------------------------------------------------------


def benchmark_preset(name: str, seed: int = 2026) -> GeneratorConfig:
    """Return a controlled benchmark scenario.

    Available presets:
    regular_control, renewal, bursty, long_gaps, informative,
    nonstationary, noisy, hard_mixed.
    """

    name = name.lower().strip()
    cfg = GeneratorConfig(seed=seed)

    if name == "regular_control":
        cfg.sampling.mode = "jittered_grid"
        cfg.sampling.jitter_fraction = 0.03
        cfg.dynamics.n_regime_changes = (0, 0)
        cfg.dynamics.n_event_pulses = (0, 1)
        cfg.noise.noise_scale_range = (0.01, 0.04)
        cfg.noise.heteroscedastic_strength = 0.0
        cfg.noise.outlier_probability = 0.0

    elif name == "renewal":
        cfg.sampling.mode = "renewal"
        cfg.sampling.renewal_shape = 0.75

    elif name == "bursty":
        cfg.sampling.mode = "bursty"
        cfg.sampling.burst_rate_ratio = 18.0
        cfg.sampling.p_enter_burst = 0.10
        cfg.sampling.p_leave_burst = 0.18

    elif name == "long_gaps":
        cfg.sampling.mode = "renewal"
        cfg.sampling.renewal_shape = 0.9
        cfg.sampling.n_gaps = 3
        cfg.sampling.total_gap_fraction = 0.28
        cfg.sampling.stratify_gaps_by_split = True

    elif name == "informative":
        cfg.sampling.mode = "renewal"
        cfg.sampling.renewal_shape = 1.0
        cfg.sampling.informative_value_strength = 1.2
        cfg.sampling.informative_derivative_strength = 1.5
        cfg.sampling.informative_oversampling = 12

    elif name == "nonstationary":
        cfg.sampling.mode = "renewal"
        cfg.dynamics.n_regime_changes = (2, 5)
        cfg.dynamics.regime_shift_strength = 0.85
        cfg.dynamics.n_event_pulses = (2, 6)
        cfg.dynamics.trend_strength = 0.65

    elif name == "noisy":
        cfg.sampling.mode = "mixed"
        cfg.noise.distribution = "mixture"
        cfg.noise.noise_scale_range = (0.10, 0.30)
        cfg.noise.heteroscedastic_strength = 0.90
        cfg.noise.outlier_probability = 0.025
        cfg.noise.outlier_scale = 7.0

    elif name == "hard_mixed":
        cfg.sampling.mode = "mixed"
        cfg.sampling.n_gaps = 3
        cfg.sampling.total_gap_fraction = 0.20
        cfg.sampling.informative_value_strength = 0.9
        cfg.sampling.informative_derivative_strength = 1.2
        cfg.sampling.informative_oversampling = 12
        cfg.dynamics.n_regime_changes = (1, 4)
        cfg.dynamics.regime_shift_strength = 0.75
        cfg.dynamics.n_event_pulses = (1, 6)
        cfg.noise.noise_scale_range = (0.07, 0.24)
        cfg.noise.heteroscedastic_strength = 0.75
        cfg.noise.outlier_probability = 0.015
        cfg.noise.outlier_scale = 6.0

    else:
        valid = [
            "regular_control",
            "renewal",
            "bursty",
            "long_gaps",
            "informative",
            "nonstationary",
            "noisy",
            "hard_mixed",
        ]
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")

    cfg.validate()
    return cfg


def events_to_grid(
    observations: pd.DataFrame,
    *,
    n_grid_points: int,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    method: Literal["linear", "previous", "nearest"] = "linear",
    include_mask: bool = True,
    include_time_since_observation: bool = True,
) -> pd.DataFrame:
    """Convert asynchronous long events to a common regular grid.

    This is useful for baselines that cannot ingest event streams directly.
    Interpolation is performed independently inside each dataset/channel.
    For honest forecasting evaluation, call this separately within each
    chronological split or use only past observations to construct a forecast
    input. Do not interpolate through the held-out future.
    """

    required = {"dataset_id", "channel", "time", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n_grid_points < 2:
        raise ValueError("n_grid_points must be at least 2.")

    result_frames = []
    for dataset_id, dataset in observations.groupby("dataset_id", sort=False):
        start = float(dataset["time"].min()) if time_start is None else float(time_start)
        end = float(dataset["time"].max()) if time_end is None else float(time_end)
        grid = np.linspace(start, end, n_grid_points)
        output = pd.DataFrame({"dataset_id": dataset_id, "time": grid})

        for channel, group in dataset.groupby("channel", sort=True):
            group = group.sort_values("time").drop_duplicates("time", keep="last")
            t = group["time"].to_numpy(dtype=float)
            y = group["value"].to_numpy(dtype=float)
            if len(t) == 0:
                values = np.full_like(grid, np.nan)
            elif method == "linear":
                values = np.interp(grid, t, y, left=np.nan, right=np.nan)
            elif method == "previous":
                idx = np.searchsorted(t, grid, side="right") - 1
                values = np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)
            elif method == "nearest":
                right = np.searchsorted(t, grid, side="left")
                left = np.clip(right - 1, 0, len(t) - 1)
                right = np.clip(right, 0, len(t) - 1)
                choose_right = np.abs(t[right] - grid) < np.abs(grid - t[left])
                idx = np.where(choose_right, right, left)
                values = y[idx]
            else:
                raise ValueError(f"Unsupported method: {method}")

            output[f"value_{channel}"] = values

            if include_mask:
                # Mask indicates exact or near-exact observations, not interpolated values.
                tolerance = (end - start) / max(n_grid_points - 1, 1) / 2.0
                nearest_distance = _nearest_distance(grid, t)
                output[f"mask_{channel}"] = (nearest_distance <= tolerance).astype(np.int8)

            if include_time_since_observation:
                previous_idx = np.searchsorted(t, grid, side="right") - 1
                delta = np.where(
                    previous_idx >= 0,
                    grid - t[np.clip(previous_idx, 0, len(t) - 1)],
                    np.nan,
                )
                output[f"delta_{channel}"] = delta

        result_frames.append(output)

    return pd.concat(result_frames, ignore_index=True)


def make_forecast_windows(
    observations: pd.DataFrame,
    *,
    context_duration: float,
    horizon_duration: float,
    stride: float,
    min_context_events: int = 16,
    min_target_events: int = 1,
) -> pd.DataFrame:
    """Create rolling context/target windows without future leakage.

    The output repeats event rows when windows overlap and adds:
    window_id, forecast_origin, role, and relative_time.
    It works for both univariate and multivariate long-format observations.
    """

    required = {"dataset_id", "time", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if context_duration <= 0 or horizon_duration <= 0 or stride <= 0:
        raise ValueError("context_duration, horizon_duration and stride must be positive.")
    if min_context_events < 1 or min_target_events < 1:
        raise ValueError("Minimum event counts must be at least 1.")

    windows: List[pd.DataFrame] = []
    window_counter = 0
    for dataset_id, dataset in observations.groupby("dataset_id", sort=False):
        dataset = dataset.sort_values("time")
        start = float(dataset["time"].min())
        end = float(dataset["time"].max())
        first_origin = start + context_duration
        last_origin = end - horizon_duration
        if first_origin > last_origin:
            continue

        origins = np.arange(first_origin, last_origin + 0.5 * stride, stride)
        for origin in origins:
            context_mask = (dataset["time"] >= origin - context_duration) & (
                dataset["time"] < origin
            )
            target_mask = (dataset["time"] >= origin) & (
                dataset["time"] <= origin + horizon_duration
            )
            if int(context_mask.sum()) < min_context_events:
                continue
            if int(target_mask.sum()) < min_target_events:
                continue

            context = dataset.loc[context_mask].copy()
            context["role"] = "context"
            target = dataset.loc[target_mask].copy()
            target["role"] = "target"
            window = pd.concat([context, target], ignore_index=True)
            window["window_id"] = f"{dataset_id}_w{window_counter:06d}"
            window["forecast_origin"] = float(origin)
            window["relative_time"] = window["time"] - float(origin)
            windows.append(window)
            window_counter += 1

    if not windows:
        return pd.DataFrame(
            columns=list(observations.columns)
            + ["role", "window_id", "forecast_origin", "relative_time"]
        )
    return pd.concat(windows, ignore_index=True)


def irregularity_metrics(times: Sequence[float]) -> Dict[str, float]:
    """Descriptive metrics of an irregular time grid."""

    t = np.sort(np.asarray(times, dtype=float))
    if len(t) < 2:
        return {
            "n": float(len(t)),
            "mean_dt": float("nan"),
            "std_dt": float("nan"),
            "cv_dt": float("nan"),
            "min_dt": float("nan"),
            "max_dt": float("nan"),
            "max_gap_ratio": float("nan"),
            "burstiness": float("nan"),
        }
    dt = np.diff(t)
    mean_dt = float(np.mean(dt))
    std_dt = float(np.std(dt))
    denominator = std_dt + mean_dt
    return {
        "n": float(len(t)),
        "mean_dt": mean_dt,
        "std_dt": std_dt,
        "cv_dt": float(std_dt / max(mean_dt, 1e-12)),
        "min_dt": float(np.min(dt)),
        "max_dt": float(np.max(dt)),
        "max_gap_ratio": float(np.max(dt) / max(mean_dt, 1e-12)),
        "burstiness": float((std_dt - mean_dt) / denominator)
        if denominator > 0
        else 0.0,
    }


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _generate_normalized_times(
    n: int, cfg: SamplingConfig, rng: np.random.Generator
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
        jitter = rng.uniform(-1.0, 1.0, size=n) * cfg.jitter_fraction * nominal_dt
        jitter[0] = 0.0
        jitter[-1] = 0.0
        times = np.clip(base + jitter, 0.0, 1.0)

    elif mode == "renewal":
        intervals = rng.gamma(shape=cfg.renewal_shape, scale=1.0, size=n - 1)
        times = np.concatenate([[0.0], np.cumsum(intervals)])
        times /= max(times[-1], 1e-12)

    elif mode == "clustered":
        centers = np.sort(rng.uniform(0.03, 0.97, size=cfg.n_clusters))
        cluster_probabilities = rng.dirichlet(np.ones(cfg.n_clusters))
        assignments = rng.choice(
            cfg.n_clusters, size=n, replace=True, p=cluster_probabilities
        )
        times = centers[assignments] + rng.normal(
            scale=cfg.cluster_width_fraction, size=n
        )
        # Add a small background component so clusters do not cover everything.
        background = rng.random(n) < 0.08
        times[background] = rng.uniform(0.0, 1.0, size=int(background.sum()))
        times = np.clip(times, 0.0, 1.0)

    elif mode == "bursty":
        state = 0
        intervals = np.empty(n - 1, dtype=float)
        for i in range(n - 1):
            if state == 0 and rng.random() < cfg.p_enter_burst:
                state = 1
            elif state == 1 and rng.random() < cfg.p_leave_burst:
                state = 0
            rate = cfg.burst_rate_ratio if state == 1 else 1.0
            intervals[i] = rng.exponential(scale=1.0 / rate)
        times = np.concatenate([[0.0], np.cumsum(intervals)])
        times /= max(times[-1], 1e-12)

    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")

    times = np.sort(times)
    return times, mode


def _generate_gaps(
    cfg: SamplingConfig,
    rng: np.random.Generator,
    split_config: Optional[SplitConfig] = None,
) -> List[Tuple[float, float]]:
    if cfg.n_gaps == 0 or cfg.total_gap_fraction <= 0:
        return []

    total_gap = cfg.total_gap_fraction
    if cfg.stratify_gaps_by_split:
        if split_config is None:
            raise ValueError("split_config es requerido para gaps estratificados.")
        if cfg.n_gaps < 3:
            raise ValueError("Los gaps estratificados requieren n_gaps >= 3.")
        boundaries = np.asarray(
            [
                0.0,
                split_config.train_fraction,
                split_config.train_fraction + split_config.validation_fraction,
                1.0,
            ],
            dtype=float,
        )
        assignments = np.arange(cfg.n_gaps) % 3
        gaps: List[Tuple[float, float]] = []
        for split_index in range(3):
            count = int(np.sum(assignments == split_index))
            if count == 0:
                continue
            region_start = float(boundaries[split_index])
            region_end = float(boundaries[split_index + 1])
            region_width = region_end - region_start
            # Cada split dedica la misma fracción de su propia duración a gaps.
            regional_gap = total_gap * region_width
            minimum_total = count * cfg.min_gap_fraction
            if minimum_total >= regional_gap:
                widths = np.full(count, regional_gap / count)
            else:
                remaining = regional_gap - minimum_total
                widths = cfg.min_gap_fraction + remaining * rng.dirichlet(np.ones(count))
            free_space = region_width - float(np.sum(widths))
            spacings = free_space * rng.dirichlet(np.ones(count + 1))
            cursor = region_start + float(spacings[0])
            for index, width in enumerate(widths):
                end = cursor + float(width)
                gaps.append((cursor, end))
                cursor = end + float(spacings[index + 1])
        return sorted(gaps)

    minimum_total = cfg.n_gaps * cfg.min_gap_fraction
    if minimum_total >= total_gap:
        widths = np.full(cfg.n_gaps, total_gap / cfg.n_gaps)
    else:
        remaining = total_gap - minimum_total
        widths = cfg.min_gap_fraction + remaining * rng.dirichlet(np.ones(cfg.n_gaps))

    # Place the gaps sequentially with random free-space allocation. This avoids overlap.
    free_space = 1.0 - float(np.sum(widths))
    spacings = free_space * rng.dirichlet(np.ones(cfg.n_gaps + 1))
    gaps: List[Tuple[float, float]] = []
    cursor = float(spacings[0])
    for i, width in enumerate(widths):
        start = cursor
        end = start + float(width)
        gaps.append((start, end))
        cursor = end + float(spacings[i + 1])
    return gaps


def _map_around_gaps(
    normalized_times: np.ndarray,
    gaps: List[Tuple[float, float]],
) -> np.ndarray:
    """Map quantiles in [0,1] to the complement of disjoint gaps."""

    if not gaps:
        return normalized_times

    allowed: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in gaps:
        if start > cursor:
            allowed.append((cursor, start))
        cursor = end
    if cursor < 1.0:
        allowed.append((cursor, 1.0))

    lengths = np.array([end - start for start, end in allowed], dtype=float)
    total_allowed = float(lengths.sum())
    cumulative = np.cumsum(lengths) / total_allowed
    previous = np.concatenate([[0.0], cumulative[:-1]])

    mapped = np.empty_like(normalized_times, dtype=float)
    for i, u in enumerate(normalized_times):
        segment = int(np.searchsorted(cumulative, u, side="right"))
        segment = min(segment, len(allowed) - 1)
        local_fraction = (u - previous[segment]) / max(
            cumulative[segment] - previous[segment], 1e-12
        )
        start, end = allowed[segment]
        mapped[i] = start + local_fraction * (end - start)
    return np.sort(mapped)


def _ensure_strictly_increasing(
    times: np.ndarray,
    lower: float,
    upper: float,
    min_separation: float,
) -> np.ndarray:
    times = np.sort(np.clip(np.asarray(times, dtype=float), lower, upper))
    n = len(times)
    if n == 0:
        return times

    # If requested separation is impossible, fall back to machine-level separation.
    max_possible = (upper - lower) / max(n - 1, 1)
    separation = min(max(min_separation, np.finfo(float).eps), 0.25 * max_possible)

    result = times.copy()
    result[0] = max(result[0], lower)
    for i in range(1, n):
        result[i] = max(result[i], result[i - 1] + separation)

    if result[-1] > upper:
        shift = result[-1] - upper
        result -= shift
        for i in range(n - 2, -1, -1):
            result[i] = min(result[i], result[i + 1] - separation)

    if result[0] < lower:
        # Extremely degenerate input; use a tiny regular grid as a safe fallback.
        result = np.linspace(lower, upper, n)
    return result


def _subtract_intervals(
    base: Tuple[float, float], gaps: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    intervals = [base]
    for gap_start, gap_end in gaps:
        updated: List[Tuple[float, float]] = []
        for start, end in intervals:
            if gap_end <= start or gap_start >= end:
                updated.append((start, end))
                continue
            if gap_start > start:
                updated.append((start, min(gap_start, end)))
            if gap_end < end:
                updated.append((max(gap_end, start), end))
        intervals = updated
    epsilon = np.finfo(float).eps
    return [(a, b) for a, b in intervals if b - a > epsilon]


def _sample_from_intervals(
    intervals: List[Tuple[float, float]], rng: np.random.Generator
) -> float:
    lengths = np.array([b - a for a, b in intervals], dtype=float)
    probabilities = lengths / lengths.sum()
    index = int(rng.choice(len(intervals), p=probabilities))
    start, end = intervals[index]
    # Avoid exact upper boundaries because chronological splits are half-open.
    return float(rng.uniform(start, np.nextafter(end, start)))


def _allocate_channel_counts(
    total: int,
    n_channels: int,
    minimum: int,
    concentration: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    remaining = total - n_channels * minimum
    weights = rng.dirichlet(np.full(n_channels, concentration))
    extra = rng.multinomial(remaining, weights) if remaining > 0 else np.zeros(n_channels, dtype=int)
    return extra + minimum, weights


def _delta_t(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.array([], dtype=float)
    return np.concatenate([[0.0], np.diff(times)])


def _robust_standardize(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=axis, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    standardized = (values - median) / scale
    if axis is None:
        return np.asarray(standardized).reshape(values.shape)
    return standardized


def _nearest_distance(grid: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if len(observed) == 0:
        return np.full_like(grid, np.inf, dtype=float)
    right = np.searchsorted(observed, grid, side="left")
    left = np.clip(right - 1, 0, len(observed) - 1)
    right = np.clip(right, 0, len(observed) - 1)
    return np.minimum(np.abs(grid - observed[left]), np.abs(observed[right] - grid))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    # Minimal reproducible example.
    config = benchmark_preset("hard_mixed", seed=1234)
    generator = IrregularTimeSeriesGenerator(config)

    univariate = generator.generate_univariate_collection(
        n_datasets=3,
        n_observations=250,
    )
    multivariate = generator.generate_multivariate_collection(
        n_datasets=2,
        n_observations=1200,
        n_channels=6,
        layout="asynchronous",
    )

    univariate.save("synthetic_output/univariate", file_format="csv")
    multivariate.save("synthetic_output/multivariate", file_format="csv")

    print(univariate.summary())
    print(multivariate.summary())
