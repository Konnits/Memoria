from __future__ import annotations

import torch

from ts_transformer.training.dilate_loss import DILATELoss
from ts_transformer.training.train_loop import Trainer


def test_all_observed_mask_preserves_unmasked_dilate() -> None:
    loss_fn = DILATELoss(alpha=0.5, gamma=0.1)
    preds = torch.tensor([[[0.0], [1.0], [2.0]]], requires_grad=True)
    targets = torch.tensor([[[0.2], [0.8], [2.2]]])
    mask = torch.ones_like(targets)

    unmasked = loss_fn.forward_parts(preds, targets)
    masked = loss_fn.forward_parts(preds, targets, target_mask=mask)

    assert torch.equal(masked.total, unmasked.total)
    assert torch.equal(masked.shape, unmasked.shape)
    assert torch.equal(masked.temporal, unmasked.temporal)


def test_fully_missing_horizon_does_not_affect_dilate_or_gradients() -> None:
    loss_fn = DILATELoss(alpha=0.5, gamma=0.1)
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    targets_a = torch.tensor([[[0.0], [10.0], [2.0]]])
    targets_b = torch.tensor([[[0.0], [-1_000_000.0], [2.0]]])

    preds_a = torch.tensor([[[0.1], [50.0], [1.8]]], requires_grad=True)
    loss_a = loss_fn(preds_a, targets_a, target_mask=mask)
    loss_a.backward()

    preds_b = preds_a.detach().clone().requires_grad_(True)
    loss_b = loss_fn(preds_b, targets_b, target_mask=mask)
    loss_b.backward()

    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(preds_a.grad, preds_b.grad)
    assert preds_a.grad[0, 1, 0] == 0.0


def test_partially_missing_channel_is_excluded_from_pairwise_distance() -> None:
    loss_fn = DILATELoss(alpha=1.0, gamma=0.1)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    targets_a = torch.tensor([[[0.0, 5.0], [1.0, 6.0]]])
    targets_b = torch.tensor([[[0.0, -1_000.0], [1.0, 2_000.0]]])
    preds = torch.tensor([[[0.2, 100.0], [0.8, -100.0]]], requires_grad=True)

    loss_a = loss_fn(preds, targets_a, target_mask=mask)
    loss_b = loss_fn(preds, targets_b, target_mask=mask)
    loss_a.backward()

    assert torch.allclose(loss_a, loss_b)
    assert torch.count_nonzero(preds.grad[..., 1]) == 0


def test_fully_unobserved_batch_returns_differentiable_zero() -> None:
    loss_fn = DILATELoss(alpha=0.5, gamma=0.1)
    preds = torch.randn(2, 3, 1, requires_grad=True)
    targets = torch.randn(2, 3, 1)
    mask = torch.zeros_like(targets)

    loss = loss_fn(preds, targets, target_mask=mask)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(preds.grad) == 0


def test_trainer_routes_target_mask_to_dilate() -> None:
    trainer = object.__new__(Trainer)
    trainer.loss_fn = DILATELoss(alpha=1.0, gamma=0.1)
    preds = torch.tensor([[[0.0], [100.0], [2.0]]], requires_grad=True)
    targets = torch.tensor([[[0.0], [-100.0], [2.0]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])

    actual = trainer._compute_loss(preds, targets, mask)
    expected = trainer.loss_fn(preds, targets, target_mask=mask)

    assert torch.allclose(actual, expected)
