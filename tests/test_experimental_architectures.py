from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts.benchmark_synthetic import (
    EXPERIMENTAL_MODEL_NAMES,
    OPTIMIZED_SIZE,
    build_models,
    load_benchmark_recipes,
)
from ts_transformer.models.attention import MultiHeadSelfAttention
from ts_transformer.models.time_series_transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)
from ts_transformer.training.optimizers import OptimizerConfig, build_optimizer
from ts_transformer.training.train_loop import Trainer
from ts_transformer.utils import load_model_config


ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides) -> TimeSeriesTransformerConfig:
    values = {
        "input_dim": 2,
        "output_dim": 2,
        "d_model": 8,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "time_scale": 2.0,
        "time_transform": "linear",
    }
    values.update(overrides)
    return TimeSeriesTransformerConfig(**values)


def _batch(num_targets: int = 2) -> dict[str, torch.Tensor]:
    values = torch.randn(2, 6, 2)
    timestamps = torch.tensor(
        [[0.0, 0.5, 2.0, 3.0, 7.0, 9.0], [1.0, 1.2, 4.0, 8.0, 9.0, 15.0]]
    )
    is_target = torch.zeros(2, 6, dtype=torch.bool)
    is_target[:, -num_targets:] = True
    return {
        "input_values": values,
        "input_timestamps": timestamps,
        "is_target_mask": is_target,
    }


def test_gaussian_head_returns_mean_by_default_and_scale_in_dict() -> None:
    model = TimeSeriesTransformer(_config(prediction_head="gaussian")).eval()
    batch = _batch()

    mean = model(**batch)
    output = model(**batch, return_dict=True)

    assert mean.shape == (2, 2, 2)
    assert output["log_scale"].shape == mean.shape
    assert torch.equal(mean, output["preds"])
    assert torch.isfinite(output["log_scale"]).all()


def test_gaussian_nll_respects_target_mask_and_backpropagates() -> None:
    trainer = object.__new__(Trainer)
    mean = torch.tensor([[[0.0, 50.0], [2.0, -50.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [4.0, 0.0]]])
    log_scale = torch.tensor([[[0.0, 4.0], [math.log(2.0), -4.0]]], requires_grad=True)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss = trainer._compute_loss(mean, target, mask, log_scale=log_scale)
    expected = (
        0.5 + 0.5 * math.log(2.0 * math.pi)
        + math.log(2.0) + 0.5 + 0.5 * math.log(2.0 * math.pi)
    ) / 2.0
    loss.backward()

    assert loss.item() == pytest.approx(expected)
    assert torch.equal(mean.grad[..., 1], torch.zeros_like(mean.grad[..., 1]))
    assert torch.equal(log_scale.grad[..., 1], torch.zeros_like(log_scale.grad[..., 1]))


def test_learnable_time_scale_has_gradient_and_no_weight_decay() -> None:
    model = TimeSeriesTransformer(_config(learnable_time_scale=True)).train()
    output = model(**_batch()).square().mean()
    output.backward()

    parameter = model.time_encoding.log_time_scale
    assert parameter is not None
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert torch.count_nonzero(parameter.grad) > 0

    optimizer = build_optimizer(
        model,
        OptimizerConfig(optimizer_name="adamw", weight_decay=0.1),
    )
    matching_groups = [
        group
        for group in optimizer.param_groups
        if any(item is parameter for item in group["params"])
    ]
    assert len(matching_groups) == 1
    assert matching_groups[0]["weight_decay"] == 0.0


def test_learnable_time_scale_rejects_ordinal_encoding() -> None:
    with pytest.raises(ValueError, match="ignora las distancias temporales"):
        TimeSeriesTransformer(
            _config(learnable_time_scale=True, time_encoding_mode="ordinal")
        )


def test_continuous_rope_attention_is_invariant_to_time_origin() -> None:
    torch.manual_seed(7)
    q = torch.randn(2, 2, 5, 4)
    k = torch.randn(2, 2, 5, 4)
    positions = torch.tensor(
        [[0.0, 0.2, 1.0, 3.5, 8.0], [0.0, 1.0, 1.4, 5.0, 6.0]]
    )

    q_a, k_a = MultiHeadSelfAttention._apply_continuous_rope(
        q, k, positions, rope_base=10_000.0
    )
    q_b, k_b = MultiHeadSelfAttention._apply_continuous_rope(
        q, k, positions + 123.0, rope_base=10_000.0
    )

    scores_a = q_a @ k_a.transpose(-2, -1)
    scores_b = q_b @ k_b.transpose(-2, -1)
    assert torch.allclose(scores_a, scores_b, atol=2e-5, rtol=2e-5)


def test_time_window_matches_dense_attention_when_window_covers_history() -> None:
    torch.manual_seed(11)
    attention = MultiHeadSelfAttention(8, 2, dropout=0.0).eval()
    x = torch.randn(2, 6, 8)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5, 2.0, 4.0], [0.0, 0.3, 1.0, 2.0, 4.0, 8.0]]
    )
    padding = torch.tensor(
        [[True, True, False, False, False, False], [False] * 6]
    )

    dense, _ = attention(x, key_padding_mask=padding, is_causal=True)
    sparse, weights = attention(
        x,
        key_padding_mask=padding,
        is_causal=True,
        temporal_positions=positions,
        temporal_attention_window=100.0,
    )

    assert weights is None
    assert torch.allclose(dense[~padding], sparse[~padding], atol=1e-6, rtol=1e-5)


def test_time_window_rejects_non_left_padding() -> None:
    attention = MultiHeadSelfAttention(8, 2, dropout=0.0).eval()
    x = torch.randn(1, 4, 8)
    positions = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="left-padding"):
        attention(
            x,
            key_padding_mask=torch.tensor([[False, False, True, False]]),
            temporal_positions=positions,
            temporal_attention_window=2.0,
        )


def test_experimental_benchmark_variants_are_opt_in() -> None:
    recipes = load_benchmark_recipes(
        ROOT / "configs" / "benchmark" / "synthetic_optuna_best.yaml"
    )
    model_cfg = load_model_config(
        str(ROOT / "configs" / "model" / "synthetic_transformer.yaml")
    )
    data = {
        "model_input_dim": 1,
        "output_dim": 1,
        "input_dim": 1,
        "use_events": False,
        "time_scale": 1.0,
    }

    default_models = build_models(model_cfg, data, (OPTIMIZED_SIZE,), recipes)
    experimental_models = build_models(
        model_cfg,
        data,
        (OPTIMIZED_SIZE,),
        recipes,
        include_experimental=True,
        experimental_temporal_window=3.5,
    )

    assert set(EXPERIMENTAL_MODEL_NAMES).isdisjoint(default_models)
    assert set(EXPERIMENTAL_MODEL_NAMES).issubset(experimental_models)
    assert experimental_models["Custom-Gaussian"].model.config.prediction_head == "gaussian"
    assert experimental_models["Custom-LearnableScale"].model.config.learnable_time_scale
    assert experimental_models["Custom-RoPE"].model.config.use_continuous_rope
    assert (
        experimental_models["Custom-TimeWindow"].model.config.temporal_attention_window
        == 3.5
    )
