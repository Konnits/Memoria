from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import yaml

from ts_transformer.data.scalers import StandardScaler
from ts_transformer.inference import ExperimentPredictor
from ts_transformer.models.time_series_transformer import TimeSeriesTransformer, TimeSeriesTransformerConfig


def _write_experiment(
    tmp_path,
    config: TimeSeriesTransformerConfig,
    *,
    drop_optional_head: bool = False,
) -> None:
    model = TimeSeriesTransformer(config)

    with open(tmp_path / "model_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f)

    state_dict = model.state_dict()
    if drop_optional_head:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("per_target_head.")
        }

    torch.save({"model_state_dict": state_dict}, tmp_path / "best_model.pt")

    scaler = StandardScaler()
    scaler.fit(np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))
    torch.save(
        {
            "value_scaler": scaler,
            "target_scaler": scaler,
            "feature_columns": ["valor"],
            "target_columns": ["valor"],
            "time_column": "timestamp",
        },
        tmp_path / "scalers.pt",
    )


def test_experiment_predictor_predicts_multiple_timestamps(tmp_path) -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=False,
        num_sensors=0,
    )
    _write_experiment(tmp_path, config)

    predictor = ExperimentPredictor.from_experiment_dir(tmp_path)
    frame = predictor.predict(
        past_values=[1.0, 2.0, 3.0, 4.0],
        past_timestamps=[100.0, 200.0, 300.0, 400.0],
        future_timestamps=[500.0, 650.0],
        return_dataframe=True,
    )

    assert list(frame.columns) == ["timestamp", "valor"]
    assert frame.shape == (2, 2)


def test_experiment_predictor_supports_single_sensor_event_models(tmp_path) -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=True,
        num_sensors=1,
    )
    _write_experiment(tmp_path, config, drop_optional_head=True)

    predictor = ExperimentPredictor.from_model_path(tmp_path / "best_model.pt")
    preds = predictor.predict(
        past_values=[1.0, 2.0, 3.0],
        past_timestamps=np.array([
            np.datetime64("2026-01-01T00:00:00"),
            np.datetime64("2026-01-01T00:15:00"),
            np.datetime64("2026-01-01T00:30:00"),
        ]),
        future_timestamps=np.array([np.datetime64("2026-01-01T00:45:00")]),
    )

    assert tuple(np.asarray(preds).shape) == (1,)


def test_experiment_predictor_loads_legacy_checkpoint_without_optional_head(tmp_path) -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=False,
        num_sensors=0,
    )
    _write_experiment(tmp_path, config, drop_optional_head=True)

    predictor = ExperimentPredictor.from_experiment_dir(tmp_path)
    preds = predictor.predict(
        past_values=[1.0, 2.0, 3.0],
        past_timestamps=[100.0, 200.0, 300.0],
        future_timestamps=[400.0],
    )

    assert tuple(np.asarray(preds).shape) == (1,)


def test_experiment_predictor_requires_sensor_ids_for_multi_sensor_event_models(tmp_path) -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=True,
        num_sensors=2,
    )
    _write_experiment(tmp_path, config)

    predictor = ExperimentPredictor.from_experiment_dir(tmp_path)

    try:
        predictor.predict(
            past_values=[1.0, 2.0, 3.0],
            past_timestamps=[100.0, 200.0, 300.0],
            future_timestamps=[400.0],
        )
    except ValueError as exc:
        assert "past_sensor_ids" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError cuando faltan past_sensor_ids.")