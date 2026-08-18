from __future__ import annotations

import pytest
import torch

from ts_transformer.models.query_cross_attention import (
    QueryCrossAttentionConfig,
    RelativeLagBias,
    RelativeTimeCrossAttention,
    StableDiagonalContinuousState,
    TimeSeriesQueryCrossAttention,
)
from ts_transformer.models.time_series_transformer import TimeSeriesTransformerConfig


def _config(**overrides) -> TimeSeriesTransformerConfig:
    values = {
        "input_dim": 2,
        "output_dim": 2,
        "d_model": 12,
        "num_heads": 3,
        "num_layers": 1,
        "decoder_num_layers": 2,
        "dim_feedforward": 24,
        "dropout": 0.0,
        "time_scale": 2.0,
        "time_transform": "linear",
    }
    values.update(overrides)
    return TimeSeriesTransformerConfig(**values)


def _dense_batch() -> dict[str, torch.Tensor]:
    values = torch.randn(2, 8, 2)
    timestamps = torch.tensor(
        [
            [10.0, 10.2, 12.0, 17.0, 18.0, 20.0, 25.0, 40.0],
            [1.0, 2.5, 2.7, 8.0, 13.0, 18.0, 30.0, 31.0],
        ],
        dtype=torch.float64,
    )
    target_mask = torch.zeros(2, 8, dtype=torch.bool)
    target_mask[:, -3:] = True
    return {
        "input_values": values,
        "input_timestamps": timestamps,
        "is_target_mask": target_mask,
    }


def test_relative_lag_bias_starts_monotonic_and_is_origin_invariant() -> None:
    bias = RelativeLagBias(3, num_frequencies=4, min_scale=0.5, max_scale=8.0)
    query = torch.tensor([[11.0, 13.0]], dtype=torch.float64)
    history = torch.tensor([[1.0, 6.0, 10.0]], dtype=torch.float64)

    score_a, lag_a = bias(query, history, time_scale=2.0)
    score_b, lag_b = bias(query + 1e8, history + 1e8, time_scale=2.0)

    assert torch.allclose(lag_a, lag_b)
    assert torch.allclose(score_a, score_b)
    # Para una misma query, una observación más antigua recibe menor score.
    assert torch.all(score_a[..., 0] < score_a[..., 1])
    assert torch.all(score_a[..., 1] < score_a[..., 2])


def test_relative_lag_bias_is_invariant_to_consistent_time_unit_changes() -> None:
    bias = RelativeLagBias(3, num_frequencies=4, min_scale=0.5, max_scale=8.0)
    query = torch.tensor([[11.0, 13.0]], dtype=torch.float64)
    history = torch.tensor([[1.0, 6.0, 10.0]], dtype=torch.float64)

    seconds, lag_seconds = bias(query, history, time_scale=2.0)
    milliseconds, lag_milliseconds = bias(
        query * 1_000.0,
        history * 1_000.0,
        time_scale=2_000.0,
    )

    assert torch.allclose(lag_seconds, lag_milliseconds)
    assert torch.allclose(seconds, milliseconds)


def test_target_queries_are_independent_and_ignore_placeholder_values() -> None:
    torch.manual_seed(11)
    model = TimeSeriesQueryCrossAttention(_config()).eval()
    batch = _dense_batch()
    original = model(**batch)

    permutation = torch.tensor([2, 0, 1])
    permuted_batch = {key: value.clone() for key, value in batch.items()}
    permuted_batch["input_values"][:, -3:] = batch["input_values"][:, -3:][:, permutation]
    permuted_batch["input_timestamps"][:, -3:] = batch["input_timestamps"][:, -3:][:, permutation]
    permuted = model(**permuted_batch)

    assert torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=1e-5)

    changed_placeholders = {key: value.clone() for key, value in batch.items()}
    changed_placeholders["input_values"][:, -3:] = 100_000.0
    changed = model(**changed_placeholders)
    assert torch.allclose(changed, original, atol=1e-6, rtol=1e-5)


def test_model_preserves_small_float64_gaps_and_backpropagates_to_lag_kernel() -> None:
    torch.manual_seed(17)
    model = TimeSeriesQueryCrossAttention(
        _config(time_scale=1e-4),
        QueryCrossAttentionConfig(use_last_value_residual=False),
    ).train()
    batch = _dense_batch()
    base = 1e8
    batch["input_timestamps"] = torch.tensor(
        [
            [base + 0.0, base + 1e-5, base + 2e-5, base + 4e-5,
             base + 8e-5, base + 1e-4, base + 2e-4, base + 4e-4],
            [base + 0.0, base + 2e-5, base + 3e-5, base + 6e-5,
             base + 9e-5, base + 2e-4, base + 3e-4, base + 5e-4],
        ],
        dtype=torch.float64,
    )
    output = model(**batch, return_dict=True)
    loss = output["preds"].square().mean()
    loss.backward()

    assert torch.count_nonzero(output["relative_lags"]) > 0
    parameter = model.cross_layers[0].cross_attention.lag_bias.raw_decay_rate
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0


def test_gaussian_event_mode_uses_expected_contract() -> None:
    torch.manual_seed(23)
    config = _config(
        input_dim=1,
        output_dim=3,
        use_sensor_embedding=True,
        num_sensors=3,
        prediction_head="gaussian",
    )
    model = TimeSeriesQueryCrossAttention(
        config,
        QueryCrossAttentionConfig(
            temporal_feature_dim=2,
            use_ctssm=True,
        ),
    ).eval()
    values = torch.randn(2, 11, 1)
    timestamps = torch.tensor(
        [[0, 1, 2, 4, 7, 8, 10, 12, 14, 14, 14]] * 2,
        dtype=torch.float64,
    )
    target_mask = torch.zeros(2, 11, dtype=torch.bool)
    target_mask[:, -3:] = True
    sensor_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2]] * 2)
    external_features = torch.randn(2, 11, 2)

    output = model(
        values,
        timestamps,
        target_mask,
        input_sensor_ids=sensor_ids,
        temporal_features=external_features,
        return_dict=True,
        return_all_layers=True,
        return_attention_weights=True,
    )

    # Tres tokens (uno por sensor) forman un único horizonte multivariado.
    assert output["preds"].shape == (2, 3)
    assert output["log_scale"].shape == output["preds"].shape
    assert output["cross_attn_weights"].shape == (2, 3, 3, 8)
    assert len(output["all_layers"]["queries"]) == 2
    assert torch.isfinite(output["preds"]).all()
    assert torch.isfinite(output["log_scale"]).all()


def test_event_special_target_id_is_mapped_to_canonical_output_channels() -> None:
    config = _config(
        input_dim=1,
        output_dim=3,
        use_sensor_embedding=True,
        num_sensors=3,
    )
    model = TimeSeriesQueryCrossAttention(config).eval()
    values = torch.randn(1, 9, 1)
    timestamps = torch.tensor(
        [[0.0, 1.0, 2.0, 4.0, 7.0, 8.0, 12.0, 12.0, 12.0]],
        dtype=torch.float64,
    )
    target_mask = torch.zeros(1, 9, dtype=torch.bool)
    target_mask[:, -3:] = True
    # SensorEmbedding reserva num_sensors=3 como id target genérico.
    sensor_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 3, 3]])

    output = model(
        values,
        timestamps,
        target_mask,
        input_sensor_ids=sensor_ids,
    )

    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()

    invalid_timestamps = timestamps.clone()
    invalid_timestamps[0, -1] += 1.0
    with pytest.raises(ValueError, match="mismo timestamp"):
        model(
            values,
            invalid_timestamps,
            target_mask,
            input_sensor_ids=sensor_ids,
        )

    invalid_history_ids = sensor_ids.clone()
    invalid_history_ids[0, 0] = config.num_sensors
    with pytest.raises(ValueError, match="ids históricos"):
        model(
            values,
            timestamps,
            target_mask,
            input_sensor_ids=invalid_history_ids,
        )

    assert model.sensor_embedding is not None
    assert model.sensor_embedding.embedding.num_embeddings == config.num_sensors


def test_event_attention_has_learnable_same_vs_cross_sensor_prior() -> None:
    attention = RelativeTimeCrossAttention(
        d_model=4,
        num_heads=2,
        dropout=0.0,
        num_lag_frequencies=2,
        lag_min_scale=1.0,
        lag_max_scale=2.0,
        mask_history_after_query=True,
        use_relative_time_bias=False,
        use_sensor_relation_bias=True,
    ).eval()
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.q_proj.bias.zero_()
        attention.k_proj.weight.zero_()
        attention.k_proj.bias.zero_()

    _, weights, _ = attention(
        query=torch.zeros(1, 1, 4),
        memory=torch.zeros(1, 3, 4),
        query_timestamps=torch.tensor([[3.0]]),
        memory_timestamps=torch.tensor([[0.0, 1.0, 2.0]]),
        time_scale=1.0,
        query_sensor_ids=torch.tensor([[0]]),
        memory_sensor_ids=torch.tensor([[0, 1, 0]]),
        return_attention_weights=True,
    )

    assert weights is not None
    assert torch.all(weights[..., 0] > weights[..., 1])
    assert torch.all(weights[..., 2] > weights[..., 1])
    assert attention.sensor_relation_bias is not None
    assert "cross_sensor_bias" not in dict(attention.named_parameters())


def test_dense_model_does_not_allocate_unreachable_attention_parameters() -> None:
    model = TimeSeriesQueryCrossAttention(_config()).eval()
    for layer in model.cross_layers:
        attention = layer.cross_attention
        assert attention.sensor_relation_bias is None
        assert "head_offset" not in dict(attention.lag_bias.named_parameters())


def test_disabled_history_time2vec_freezes_only_its_exclusive_parameters() -> None:
    model = TimeSeriesQueryCrossAttention(
        _config(time_encoding_mode="time2vec", learnable_time_scale=True),
        QueryCrossAttentionConfig(
            use_history_time_encoding=False,
            use_query_horizon=True,
            use_relative_time_bias=False,
            use_temporal_film=False,
            use_ctssm=False,
        ),
    )

    assert model.time_encoding.log_time_scale is not None
    assert model.time_encoding.log_time_scale.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in model.time_encoding.named_parameters()
        if name != "log_time_scale"
    )
    assert any(
        parameter.requires_grad
        for name, parameter in model.query_time_encoding.named_parameters()
        if name != "log_time_scale"
    )


def test_continuous_state_decay_is_stable_and_gap_dependent() -> None:
    torch.manual_seed(29)
    module = StableDiagonalContinuousState(d_model=4, time_scale=1.0).eval()
    inputs = torch.randn(1, 3, 4)
    short_times = torch.tensor([[0.0, 0.1, 0.2]])
    long_times = torch.tensor([[0.0, 10.0, 20.0]])

    short_output = module(inputs, short_times)
    long_output = module(inputs, long_times)

    assert torch.isfinite(short_output).all()
    assert torch.isfinite(long_output).all()
    assert not torch.allclose(short_output[:, 1:], long_output[:, 1:])
    assert torch.all(torch.nn.functional.softplus(module.raw_decay_rate) > 0)


def test_parallel_affine_scan_matches_sequential_recurrence() -> None:
    torch.manual_seed(31)
    coefficients = torch.sigmoid(torch.randn(2, 9, 4))
    innovations = torch.randn(2, 9, 4)
    actual = StableDiagonalContinuousState._parallel_affine_scan(
        coefficients, innovations
    )

    state = torch.zeros(2, 4)
    expected = []
    for index in range(coefficients.shape[1]):
        state = coefficients[:, index] * state + innovations[:, index]
        expected.append(state)
    expected_tensor = torch.stack(expected, dim=1)

    assert torch.allclose(actual, expected_tensor, atol=1e-6, rtol=1e-5)


def test_left_padding_cannot_change_valid_predictions() -> None:
    torch.manual_seed(31)
    model = TimeSeriesQueryCrossAttention(_config()).eval()
    batch = _dense_batch()
    one = {key: value[:1] for key, value in batch.items()}
    expected = model(**one)

    padded = {
        "input_values": torch.cat((torch.full((1, 2, 2), 99.0), one["input_values"]), dim=1),
        "input_timestamps": torch.cat(
            (torch.tensor([[-1e9, 1e9]], dtype=torch.float64), one["input_timestamps"]),
            dim=1,
        ),
        "is_target_mask": torch.cat(
            (torch.zeros(1, 2, dtype=torch.bool), one["is_target_mask"]), dim=1
        ),
        "padding_mask": torch.tensor(
            [[True, True, False, False, False, False, False, False, False, False]]
        ),
    }
    actual = model(**padded)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    lengths_only = dict(padded)
    lengths_only.pop("padding_mask")
    lengths_only["lengths"] = torch.tensor([8])
    actual_from_lengths = model(**lengths_only)
    assert torch.allclose(actual_from_lengths, expected, atol=1e-6, rtol=1e-5)


def test_lengths_and_padding_must_describe_the_same_left_padding() -> None:
    model = TimeSeriesQueryCrossAttention(_config()).eval()
    batch = _dense_batch()
    with pytest.raises(ValueError, match="secuencias distintas"):
        model(
            **batch,
            padding_mask=torch.zeros(2, 8, dtype=torch.bool),
            lengths=torch.tensor([7, 8]),
        )


def test_temporal_ablation_flags_do_not_leak_target_slot_or_timestamp() -> None:
    torch.manual_seed(37)
    no_time = TimeSeriesQueryCrossAttention(
        _config(time_encoding_mode="ordinal"),
        QueryCrossAttentionConfig(
            use_relative_time_bias=False,
            use_temporal_film=False,
            use_query_horizon=False,
            use_history_time_encoding=False,
            use_ctssm=False,
            use_last_value_residual=False,
        ),
    ).eval()
    batch = _dense_batch()
    original = no_time(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["input_timestamps"][:, -3:] += torch.tensor([10.0, 100.0, 1000.0])
    changed = no_time(**changed_batch)

    assert torch.equal(changed, original)
    # Sin tiempo ni sensor, todas las queries son realmente indistinguibles;
    # el índice de slot no actúa como horizonte oculto.
    assert torch.allclose(original[:, 0], original[:, 1], atol=1e-7, rtol=1e-7)
    assert torch.allclose(original[:, 1], original[:, 2], atol=1e-7, rtol=1e-7)
    assert not no_time.time_emb_scale.requires_grad
    assert all(
        not parameter.requires_grad
        for layer in no_time.cross_layers
        for parameter in layer.cross_attention.lag_bias.parameters()
    )


def test_query_horizon_can_be_isolated_from_history_timing() -> None:
    torch.manual_seed(41)
    model = TimeSeriesQueryCrossAttention(
        _config(),
        QueryCrossAttentionConfig(
            use_relative_time_bias=False,
            use_temporal_film=False,
            use_query_horizon=True,
            use_history_time_encoding=False,
            use_ctssm=False,
            use_last_value_residual=False,
        ),
    ).eval()
    batch = _dense_batch()
    original = model(**batch)

    changed_history = {key: value.clone() for key, value in batch.items()}
    # Mantiene el último tiempo y los horizontes, pero altera los gaps históricos.
    changed_history["input_timestamps"][:, :4] = changed_history[
        "input_timestamps"
    ][:, :4].mean(dim=1, keepdim=True)
    history_output = model(**changed_history)
    assert torch.allclose(history_output, original, atol=1e-6, rtol=1e-5)

    changed_query = {key: value.clone() for key, value in batch.items()}
    changed_query["input_timestamps"][:, -1] += 17.0
    query_output = model(**changed_query)
    assert not torch.allclose(query_output[:, -1], original[:, -1])


@pytest.mark.parametrize("mode", ["sinusoidal", "mlp", "time2vec"])
def test_all_continuous_time_encodings_accept_float64_timestamps(mode: str) -> None:
    model = TimeSeriesQueryCrossAttention(_config(time_encoding_mode=mode)).eval()
    output = model(**_dense_batch())
    assert output.shape == (2, 3, 2)
    assert torch.isfinite(output).all()
