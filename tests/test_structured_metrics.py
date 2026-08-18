from __future__ import annotations

import pytest
import torch

from ts_transformer.training.metrics import compute_structured_regression_metrics


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
