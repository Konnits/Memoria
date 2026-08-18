from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.data.sequence_builder import SequenceBuilder
from ts_transformer.inference.predictor import Predictor
from ts_transformer.models.query_cross_attention import TimeSeriesQueryCrossAttention
from ts_transformer.models.time_series_transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)


def _make_predictor(*, event_mode: bool) -> Predictor:
    output_dim = 2 if event_mode else 1
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=output_dim,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=event_mode,
        num_sensors=2 if event_mode else 0,
    )
    model = TimeSeriesTransformer(config)
    builder = SequenceBuilder(
        input_dim=1,
        target_token_value="zeros",
        use_sensor_ids=event_mode,
        num_sensors=2 if event_mode else 0,
        num_target_tokens=output_dim if event_mode else 1,
        target_sensor_ids=[0, 1] if event_mode else None,
    )
    return Predictor(model=model, sequence_builder=builder)


@pytest.mark.parametrize("event_mode", [False, True])
def test_predict_multi_targets_matches_independent_predictions(event_mode: bool) -> None:
    torch.manual_seed(7)
    predictor = _make_predictor(event_mode=event_mode)
    past_values = np.asarray([[0.2], [0.4], [0.1], [0.8]], dtype=np.float32)
    past_timestamps = np.asarray([1.0, 2.5, 6.0, 9.0], dtype=np.float32)
    future_timestamps = [10.0, 14.0, 25.0]
    sensor_ids = np.asarray([0, 1, 0, 1]) if event_mode else None

    expected = torch.stack(
        [
            predictor.predict_single(
                past_values,
                past_timestamps,
                timestamp,
                past_sensor_ids=sensor_ids,
                return_torch=True,
            )
            for timestamp in future_timestamps
        ]
    )
    actual = predictor.predict_multi_targets(
        past_values,
        past_timestamps,
        future_timestamps,
        past_sensor_ids=sensor_ids,
        return_torch=True,
    )

    assert actual.shape == (len(future_timestamps), predictor.model.output_dim)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_predict_multi_targets_invokes_model_once() -> None:
    predictor = _make_predictor(event_mode=False)
    calls = 0

    def count_forward_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = predictor.model.register_forward_hook(count_forward_call)
    try:
        predictor.predict_multi_targets(
            [[0.2], [0.4], [0.8]],
            [1.0, 3.0, 7.0],
            [8.0, 10.0, 15.0, 30.0],
        )
    finally:
        handle.remove()

    assert calls == 1


def test_predict_multi_targets_rejects_empty_targets() -> None:
    predictor = _make_predictor(event_mode=False)

    with pytest.raises(ValueError, match="al menos un timestamp"):
        predictor.predict_multi_targets(
            [[0.2], [0.4]],
            [1.0, 2.0],
            [],
        )


def test_default_event_builder_assigns_real_sensor_ids_for_query_cross() -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=2,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sensor_embedding=True,
        num_sensors=2,
    )
    predictor = Predictor(model=TimeSeriesQueryCrossAttention(config))

    assert predictor.sequence_builder.target_sensor_ids == [0, 1]
    prediction = predictor.predict_single(
        [[0.2], [0.4], [0.8]],
        [1.0, 3.0, 7.0],
        10.0,
        past_sensor_ids=[0, 1, 0],
        return_torch=True,
    )

    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
