from __future__ import annotations

import pytest
import torch

from ts_transformer.models import (
    ContinuousBasisDecoderConfig,
    ContinuousHorizonBasis,
    TimeSeriesContinuousBasisDecoder,
)
from ts_transformer.models.time_series_transformer import (
    TimeSeriesTransformerConfig,
)


def _config(**overrides) -> TimeSeriesTransformerConfig:
    values = {
        "input_dim": 2,
        "output_dim": 2,
        "d_model": 12,
        "num_heads": 3,
        "num_layers": 1,
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


def test_horizon_basis_contains_continuous_trend_rbf_and_fourier_terms() -> None:
    config = ContinuousBasisDecoderConfig(
        trend_degree=2,
        num_rbf_bases=3,
        num_fourier_frequencies=2,
        min_basis_scale=0.5,
        max_basis_scale=8.0,
    )
    basis = ContinuousHorizonBasis(config)
    horizons = torch.tensor([[0.0, 1.0, 1.0 + 1e-6]])
    result = basis(horizons)

    assert result.shape == (1, 3, config.num_basis_functions)
    assert torch.equal(result[0, 0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.all(result[0, 0, 3:6] > 0.0)
    assert torch.equal(result[0, 0, 6:], torch.zeros(4))
    assert torch.allclose(result[:, 1], result[:, 2], atol=5e-5, rtol=5e-5)


def test_decoder_is_continuous_and_sensitive_to_physical_horizon() -> None:
    basis_config = ContinuousBasisDecoderConfig(
        trend_degree=1,
        num_rbf_bases=0,
        num_fourier_frequencies=0,
        derive_gap_features=False,
        use_history_time_encoding=False,
        use_last_value_residual=False,
    )
    model = TimeSeriesContinuousBasisDecoder(_config(), basis_config).eval()
    with torch.no_grad():
        model.coefficient_projection.weight.zero_()
        model.coefficient_projection.bias.zero_()
        # Canal 0, coeficiente de signed-log trend (base índice 1).
        model.coefficient_projection.bias[1] = 1.0

    values = torch.randn(1, 8, 2)
    timestamps = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.000002, 16.0]],
        dtype=torch.float64,
    )
    target_mask = torch.zeros(1, 8, dtype=torch.bool)
    target_mask[:, -3:] = True
    prediction = model(values, timestamps, target_mask)

    assert prediction.shape == (1, 3, 2)
    assert torch.allclose(
        prediction[:, 0, 0], prediction[:, 1, 0], atol=2e-6, rtol=2e-6
    )
    assert not torch.allclose(prediction[:, 0, 0], prediction[:, 2, 0])
    assert torch.equal(prediction[..., 1], torch.zeros_like(prediction[..., 1]))


def test_decoder_is_origin_invariant_and_has_no_target_slot_encoding() -> None:
    torch.manual_seed(113)
    model = TimeSeriesContinuousBasisDecoder(
        _config(time_encoding_mode="ordinal")
    ).eval()
    batch = _dense_batch()
    original = model(**batch)

    shifted = {key: value.clone() for key, value in batch.items()}
    shifted["input_timestamps"] += 1_000_000_000.0
    shifted_output = model(**shifted)
    assert torch.allclose(shifted_output, original, atol=2e-5, rtol=2e-5)

    permutation = torch.tensor([2, 0, 1])
    permuted = {key: value.clone() for key, value in batch.items()}
    permuted["input_timestamps"][:, -3:] = batch["input_timestamps"][:, -3:][
        :, permutation
    ]
    permuted["input_values"][:, -3:] = 100_000.0
    permuted_output = model(**permuted)
    assert torch.allclose(
        permuted_output, original[:, permutation], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("prediction_head", ["point", "gaussian"])
def test_event_contract_supports_multiple_horizons_and_external_features(
    prediction_head: str,
) -> None:
    torch.manual_seed(127)
    config = _config(
        input_dim=1,
        output_dim=3,
        use_sensor_embedding=True,
        num_sensors=3,
        prediction_head=prediction_head,
    )
    model = TimeSeriesContinuousBasisDecoder(
        config,
        ContinuousBasisDecoderConfig(
            temporal_feature_dim=2,
            use_ctssm=True,
        ),
    ).eval()
    values = torch.randn(2, 14, 1)
    timestamps = torch.tensor(
        [[0, 1, 2, 4, 5, 7, 8, 10, 12, 12, 12, 16, 16, 16]] * 2,
        dtype=torch.float64,
    )
    target_mask = torch.zeros(2, 14, dtype=torch.bool)
    target_mask[:, -6:] = True
    # SensorEmbedding reserva num_sensors como id target especial.
    sensor_ids = torch.tensor(
        [[0, 1, 2, 0, 1, 2, 0, 1, 3, 3, 3, 3, 3, 3]] * 2
    )
    temporal_features = torch.randn(2, 14, 2)

    output = model(
        values,
        timestamps,
        target_mask,
        input_sensor_ids=sensor_ids,
        temporal_features=temporal_features,
        return_dict=True,
        return_all_layers=True,
        return_attention_weights=True,
    )

    assert output["preds"].shape == (2, 2, 3)
    assert output["relative_horizons"].shape == (2, 6)
    assert output["horizon_basis"].shape[-1] == model.basis.output_dim
    assert len(output["all_layers"]["encoder"]) == config.num_layers
    assert torch.isfinite(output["preds"]).all()
    if prediction_head == "gaussian":
        assert output["log_scale"].shape == (2, 2, 3)
        assert torch.isfinite(output["log_scale"]).all()
    else:
        assert "log_scale" not in output

    assert model.sensor_embedding is not None
    assert model.sensor_embedding.embedding.num_embeddings == config.num_sensors


def test_disabled_history_time2vec_keeps_only_the_used_time_scale_trainable() -> None:
    model = TimeSeriesContinuousBasisDecoder(
        _config(time_encoding_mode="time2vec", learnable_time_scale=True),
        ContinuousBasisDecoderConfig(use_history_time_encoding=False),
    )

    assert model.time_encoding.log_time_scale is not None
    assert model.time_encoding.log_time_scale.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in model.time_encoding.named_parameters()
        if name != "log_time_scale"
    )


def test_left_padding_and_lengths_do_not_change_continuous_forecast() -> None:
    torch.manual_seed(131)
    model = TimeSeriesContinuousBasisDecoder(_config()).eval()
    batch = _dense_batch()
    one = {key: value[:1] for key, value in batch.items()}
    expected = model(**one)

    padded = {
        "input_values": torch.cat(
            (torch.full((1, 2, 2), 999.0), one["input_values"]), dim=1
        ),
        "input_timestamps": torch.cat(
            (
                torch.tensor([[-1e10, 1e10]], dtype=torch.float64),
                one["input_timestamps"],
            ),
            dim=1,
        ),
        "is_target_mask": torch.cat(
            (torch.zeros(1, 2, dtype=torch.bool), one["is_target_mask"]), dim=1
        ),
        "lengths": torch.tensor([8]),
    }
    actual = model(**padded)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_gaussian_decoder_starts_at_persistence_with_unit_scale_and_can_learn() -> None:
    torch.manual_seed(137)
    model = TimeSeriesContinuousBasisDecoder(
        _config(prediction_head="gaussian"),
        ContinuousBasisDecoderConfig(temporal_feature_dim=1),
    ).train()
    batch = _dense_batch()
    temporal_features = torch.randn(2, 8, 1)

    initial = model(
        **batch,
        temporal_features=temporal_features,
        return_dict=True,
    )
    # Los últimos tres tokens son queries: el índice 4 es la última
    # observación histórica densa y se repite para todos los horizontes.
    persistence = batch["input_values"][:, 4].unsqueeze(1).expand(-1, 3, -1)
    assert torch.equal(initial["preds"], persistence)
    assert torch.equal(initial["log_scale"], torch.zeros_like(initial["log_scale"]))

    objective = (
        (initial["preds"] - (persistence + 1.0)).square().mean()
        + (initial["log_scale"] - 0.25).square().mean()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    objective.backward()

    coefficient_grad = model.coefficient_projection.weight.grad
    target_coefficient_grad = model.target_coefficient_projection.weight.grad
    assert coefficient_grad is not None and torch.count_nonzero(coefficient_grad) > 0
    assert (
        target_coefficient_grad is not None
        and torch.count_nonzero(target_coefficient_grad) > 0
    )
    optimizer.step()

    updated = model(
        **batch,
        temporal_features=temporal_features,
        return_dict=True,
    )
    assert not torch.equal(updated["preds"], persistence)
    assert not torch.equal(updated["log_scale"], torch.zeros_like(updated["log_scale"]))


def test_event_decoder_initial_mean_is_per_sensor_persistence() -> None:
    config = _config(
        input_dim=1,
        output_dim=3,
        use_sensor_embedding=True,
        num_sensors=3,
        prediction_head="gaussian",
    )
    model = TimeSeriesContinuousBasisDecoder(config).eval()
    values = torch.arange(11, dtype=torch.float32).view(1, 11, 1)
    timestamps = torch.tensor(
        [[0, 1, 2, 4, 5, 7, 8, 10, 12, 12, 12]], dtype=torch.float64
    )
    target_mask = torch.zeros(1, 11, dtype=torch.bool)
    target_mask[:, -3:] = True
    sensor_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2]])

    output = model(
        values,
        timestamps,
        target_mask,
        input_sensor_ids=sensor_ids,
        return_dict=True,
    )

    assert torch.equal(output["preds"], torch.tensor([[6.0, 7.0, 5.0]]))
    assert torch.equal(output["log_scale"], torch.zeros_like(output["log_scale"]))
