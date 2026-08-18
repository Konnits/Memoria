from __future__ import annotations

import pytest
import torch

from ts_transformer.training.metrics import (
    compute_gaussian_metrics,
    compute_structured_regression_metrics,
)


def test_structured_metrics_report_targets_and_channels_with_mask() -> None:
    targets = torch.zeros(2, 2, 2)
    preds = torch.tensor(
        [
            [[1.0, 100.0], [2.0, 4.0]],
            [[3.0, 100.0], [4.0, 8.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    metrics = compute_structured_regression_metrics(preds, targets, mask, prefix="test_")

    assert metrics["test_mse_target_0"] == pytest.approx(5.0)
    assert metrics["test_mse_target_1"] == pytest.approx(10.0)
    assert metrics["test_mse_channel_0"] == pytest.approx(7.5)
    assert "test_mse_channel_1" not in metrics


def test_gaussian_metrics_are_calibrated_for_known_quantiles_and_respect_mask() -> None:
    mean = torch.zeros(1, 2, 3)
    targets = torch.tensor([[[-1.0, 0.0, 1.0], [-3.0, 3.0, 100.0]]])
    log_scale = torch.zeros_like(targets)
    mask = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])

    metrics = compute_gaussian_metrics(mean, targets, log_scale, mask, prefix="test_")

    assert metrics["test_mean_sigma"] == pytest.approx(1.0)
    assert metrics["test_coverage_90"] == pytest.approx(3.0 / 5.0)
    assert metrics["test_nll"] > 0.0
    assert metrics["test_crps"] > 0.0
    assert "test_coverage_error_95" in metrics


def test_gaussian_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="misma shape"):
        compute_gaussian_metrics(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, 1),
        )
