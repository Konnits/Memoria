from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _soft_dtw(distance: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Batched differentiable DTW using the soft-min recursion.

    Parameters
    ----------
    distance:
        Pairwise distance matrix with shape [B, N, M].
    gamma:
        Smoothing parameter. Lower values approach classic DTW.
    """
    if distance.ndim != 3:
        raise ValueError(f"distance must have shape [B, N, M], got {tuple(distance.shape)}")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0.")

    batch_size, n_steps, m_steps = distance.shape
    inf = torch.tensor(float("inf"), device=distance.device, dtype=distance.dtype)
    r = torch.full(
        (batch_size, n_steps + 2, m_steps + 2),
        inf,
        device=distance.device,
        dtype=distance.dtype,
    )
    r[:, 0, 0] = 0.0

    for i in range(1, n_steps + 1):
        for j in range(1, m_steps + 1):
            prev = torch.stack(
                (
                    r[:, i - 1, j - 1],
                    r[:, i - 1, j],
                    r[:, i, j - 1],
                ),
                dim=-1,
            )
            soft_min = -gamma * torch.logsumexp(-prev / gamma, dim=-1)
            r[:, i, j] = distance[:, i - 1, j - 1] + soft_min

    return r[:, n_steps, m_steps]


@dataclass
class DILATEParts:
    total: torch.Tensor
    shape: torch.Tensor
    temporal: torch.Tensor


class DILATELoss(nn.Module):
    """
    Shape and Time Loss for time-series forecasting.

    The loss combines a soft-DTW shape term with a temporal distortion term
    derived from the soft alignment path:

        loss = alpha * shape_loss + (1 - alpha) * temporal_loss

    Inputs are expected as [B, T, D]. If [B, D] is provided, it is treated as a
    single-step sequence. ``target_mask`` is optional and uses the same shape:
    missing dimensions are excluded from pairwise distances, while time steps
    with no observed target dimensions are removed from the alignment.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.01,
        normalize_shape: bool = True,
        normalize_temporal: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalize_shape = bool(normalize_shape)
        self.normalize_temporal = bool(normalize_temporal)
        self.eps = float(eps)
        self._omega_cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_parts(preds, targets, target_mask=target_mask).total

    def forward_parts(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> DILATEParts:
        if preds.ndim == 2:
            preds = preds.unsqueeze(1)
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)
        if preds.shape != targets.shape:
            raise ValueError(
                "preds and targets must have the same shape. "
                f"preds={tuple(preds.shape)}, targets={tuple(targets.shape)}"
            )

        if target_mask is None:
            return self._forward_complete_sequence(preds, targets)

        if target_mask.ndim == 2:
            target_mask = target_mask.unsqueeze(1)
        if target_mask.shape != targets.shape:
            raise ValueError(
                "target_mask must have the same shape as targets. "
                f"target_mask={tuple(target_mask.shape)}, targets={tuple(targets.shape)}"
            )

        observed_mask = target_mask.to(device=targets.device) > 0
        if torch.all(observed_mask):
            return self._forward_complete_sequence(preds, targets)

        # Un horizonte sin ninguna variable observada no debe participar ni
        # como predicción ni como target en el alineamiento. Como cada muestra
        # puede conservar una longitud distinta, procesamos esas subsecuencias
        # por separado y promediamos sólo las que contienen supervisión.
        sample_parts: list[DILATEParts] = []
        for batch_idx in range(preds.shape[0]):
            valid_steps = observed_mask[batch_idx].any(dim=-1)
            if not torch.any(valid_steps):
                continue

            sample_parts.append(
                self._forward_complete_sequence(
                    preds[batch_idx : batch_idx + 1, valid_steps, :],
                    targets[batch_idx : batch_idx + 1, valid_steps, :],
                    observed_mask=observed_mask[
                        batch_idx : batch_idx + 1, valid_steps, :
                    ],
                )
            )

        if not sample_parts:
            # Cero diferenciable: un batch sin supervisión no debe actualizar
            # el modelo ni fallar durante backward().
            zero = preds.sum() * 0.0
            return DILATEParts(total=zero, shape=zero, temporal=zero)

        return DILATEParts(
            total=torch.stack([parts.total for parts in sample_parts]).mean(),
            shape=torch.stack([parts.shape for parts in sample_parts]).mean(),
            temporal=torch.stack([parts.temporal for parts in sample_parts]).mean(),
        )

    def _forward_complete_sequence(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> DILATEParts:
        # The temporal term is a gradient of soft-DTW wrt the distance matrix.
        # It needs autograd even during validation, but only needs higher-order
        # gradients when the model prediction itself requires gradients.
        needs_model_grad = bool(preds.requires_grad)
        with torch.enable_grad():
            preds_for_distance = preds if needs_model_grad else preds.detach()
            targets_for_distance = targets if needs_model_grad else targets.detach()

            if observed_mask is None:
                distance = torch.cdist(
                    preds_for_distance,
                    targets_for_distance,
                    p=2,
                ).pow(2)
            else:
                mask = observed_mask.to(
                    device=targets.device,
                    dtype=preds_for_distance.dtype,
                )
                safe_targets = torch.where(
                    observed_mask,
                    targets_for_distance,
                    torch.zeros_like(targets_for_distance),
                )
                pairwise_error = (
                    preds_for_distance.unsqueeze(2)
                    - safe_targets.unsqueeze(1)
                ).pow(2)
                # La máscara pertenece a cada target j y se comparte sobre
                # todos los posibles pasos de predicción i del alineamiento.
                distance = (pairwise_error * mask.unsqueeze(1)).sum(dim=-1)

            if not needs_model_grad:
                distance.requires_grad_(True)

            shape_values = _soft_dtw(distance, gamma=self.gamma)
            if self.normalize_shape:
                shape_values = shape_values / float(max(preds.shape[1], targets.shape[1]))
            shape_loss = shape_values.mean()

            if preds.shape[1] <= 1 or targets.shape[1] <= 1 or self.alpha >= 1.0:
                temporal_loss = shape_loss.new_tensor(0.0)
            else:
                alignment = torch.autograd.grad(
                    shape_values.sum(),
                    distance,
                    create_graph=needs_model_grad,
                    retain_graph=needs_model_grad,
                )[0]
                omega = self._temporal_distortion_matrix(
                    preds.shape[1],
                    targets.shape[1],
                    device=preds.device,
                    dtype=preds.dtype,
                )
                temporal_values = (alignment * omega.unsqueeze(0)).sum(dim=(1, 2))
                if self.normalize_temporal:
                    temporal_values = temporal_values / alignment.sum(dim=(1, 2)).clamp_min(self.eps)
                temporal_loss = temporal_values.mean()

            total = self.alpha * shape_loss + (1.0 - self.alpha) * temporal_loss
            return DILATEParts(total=total, shape=shape_loss, temporal=temporal_loss)

    def _temporal_distortion_matrix(
        self,
        n_steps: int,
        m_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (int(n_steps), int(m_steps), str(device), dtype)
        cached = self._omega_cache.get(cache_key)
        if cached is not None:
            return cached

        i = torch.arange(n_steps, device=device, dtype=dtype).unsqueeze(1)
        j = torch.arange(m_steps, device=device, dtype=dtype).unsqueeze(0)
        omega = (i - j).pow(2)
        if self.normalize_temporal and max(n_steps, m_steps) > 1:
            omega = omega / float(max(n_steps, m_steps) - 1) ** 2
        self._omega_cache[cache_key] = omega
        return omega
