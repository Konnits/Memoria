from __future__ import annotations

import torch

from ts_transformer.features.time_encoding import TimePositionalEncoding
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.models.time_series_transformer import TimeSeriesTransformerConfig


def test_ordinal_encoding_ignores_real_time_gaps() -> None:
    encoding = TimePositionalEncoding(d_model=8, mode="ordinal", time_transform="linear")
    regular = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    irregular = torch.tensor([[10.0, 10.1, 50.0, 900.0]])

    assert torch.allclose(encoding(regular), encoding(irregular))


def test_ordinal_encoding_starts_at_first_valid_token_with_left_padding() -> None:
    encoding = TimePositionalEncoding(d_model=8, mode="ordinal", time_transform="linear")
    padded = torch.tensor([[0.0, 0.0, 10.0, 12.0, 30.0]])
    padding_mask = torch.tensor([[True, True, False, False, False]])
    unpadded = torch.tensor([[10.0, 12.0, 30.0]])

    padded_result = encoding(padded, padding_mask=padding_mask)
    unpadded_result = encoding(unpadded)

    assert torch.allclose(padded_result[:, 2:], unpadded_result)


def test_transformer_can_disable_target_role_embedding() -> None:
    config = TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        use_target_flag_embedding=False,
    )
    model = TimeSeriesTransformer(config)

    assert model.flag_embedding is None
    assert model.use_target_flag_embedding is False
