from __future__ import annotations

import torch

from ts_transformer.features.value_embedding import FeatureEmbedding
from ts_transformer.models.attention import MultiHeadSelfAttention
from ts_transformer.models.time_series_transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)


def _config(**overrides) -> TimeSeriesTransformerConfig:
    values = {
        "input_dim": 1,
        "output_dim": 1,
        "d_model": 8,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "time_scale": 2.0,
        "time_transform": "log1p",
    }
    values.update(overrides)
    return TimeSeriesTransformerConfig(**values)


def test_padding_content_cannot_change_valid_token_predictions() -> None:
    """El key padding mask basta: no es necesario alterar el encoding temporal."""
    torch.manual_seed(19)
    model = TimeSeriesTransformer(_config()).eval()

    unpadded_values = torch.tensor([[[0.2], [0.4], [0.9], [0.0]]])
    unpadded_times = torch.tensor([[10.0, 11.0, 15.0, 21.0]])
    unpadded_target = torch.tensor([[False, False, False, True]])
    unpadded_output = model(
        input_values=unpadded_values,
        input_timestamps=unpadded_times,
        is_target_mask=unpadded_target,
        return_dict=True,
    )

    padded_values = torch.cat(
        [torch.tensor([[[999.0], [-321.0]]]), unpadded_values], dim=1
    )
    padded_times = torch.cat(
        [torch.tensor([[50_000.0, -50_000.0]]), unpadded_times], dim=1
    )
    padded_target = torch.tensor([[False, False, False, False, False, True]])
    padding_mask = torch.tensor([[True, True, False, False, False, False]])
    padded_output = model(
        input_values=padded_values,
        input_timestamps=padded_times,
        is_target_mask=padded_target,
        padding_mask=padding_mask,
        return_dict=True,
    )

    assert torch.allclose(
        unpadded_output["encoder_output"],
        padded_output["encoder_output"][:, 2:],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        unpadded_output["preds"], padded_output["preds"], atol=1e-6, rtol=1e-5
    )


def test_temporal_bias_receives_linear_time_even_with_log_encoding() -> None:
    """El bias modela intervalos; no debe heredar time_transform='log1p'."""
    model = TimeSeriesTransformer(_config(use_temporal_attn_bias=True)).eval()
    captured_tau: list[torch.Tensor] = []

    def capture_tau(_module, args) -> None:
        captured_tau.append(args[0].detach().clone())

    handle = model.temporal_attn_bias.register_forward_pre_hook(capture_tau)
    try:
        model(
            input_values=torch.tensor([[[0.1], [0.2], [0.3], [0.0]]]),
            input_timestamps=torch.tensor([[10.0, 12.0, 18.0, 22.0]]),
            is_target_mask=torch.tensor([[False, False, False, True]]),
        )
    finally:
        handle.remove()

    assert len(captured_tau) == 1
    assert torch.equal(captured_tau[0], torch.tensor([[0.0, 1.0, 4.0, 6.0]]))


def test_causal_sdpa_preserves_mask_when_temporal_bias_is_present() -> None:
    """Agregar bias temporal nunca debe abrir atención hacia el futuro."""
    torch.manual_seed(23)
    attention = MultiHeadSelfAttention(8, 2, dropout=0.0).eval()
    inputs = torch.randn(1, 4, 8)
    changed_future = inputs.clone()
    changed_future[:, -1] = 1_000.0

    bias = torch.zeros(1, 2, 4, 4)
    bias[:, :, -1, 1:] = -20.0
    original, _ = attention(inputs, temporal_bias=bias, is_causal=True)
    perturbed, _ = attention(changed_future, temporal_bias=bias, is_causal=True)
    without_bias, _ = attention(inputs, is_causal=True)

    assert torch.allclose(original[:, :-1], perturbed[:, :-1], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original[:, -1], without_bias[:, -1])


def test_scalar_features_are_normalized_only_after_projection() -> None:
    """LayerNorm sobre d_in=1 borraría toda la señal en modo evento."""
    embedding = FeatureEmbedding(d_in=1, d_model=4, use_layernorm=True)
    with torch.no_grad():
        embedding.proj.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        embedding.proj.bias.zero_()

    output = embedding(torch.tensor([[[0.0], [2.0]]]))

    assert embedding.ln is not None
    assert embedding.ln.normalized_shape == (4,)
    assert not torch.allclose(output[:, 0], output[:, 1])
