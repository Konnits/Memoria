from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = "",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Calcula métricas básicas de regresión (MSE, RMSE, MAE, MAPE).

    Parameters
    ----------
    preds:
        Tensor [N, D] con predicciones.
    targets:
        Tensor [N, D] con valores verdaderos.
    prefix:
        Prefijo opcional para los nombres de las métricas (ej. "val_").
    eps:
        Pequeño valor para evitar divisiones por cero en MAPE.

    Returns
    -------
    metrics:
        Diccionario con métricas agregadas sobre todas las dimensiones.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds y targets deben tener la misma shape. "
            f"preds: {tuple(preds.shape)}, targets: {tuple(targets.shape)}"
        )

    # Convertimos a float32 por si vienen en otro dtype
    preds = preds.float()
    targets = targets.float()

    diff = preds - targets
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5
    mae = diff.abs().mean().item()

    # MAPE: error porcentual absoluto medio
    denom = targets.abs().clamp(min=eps)
    mape = (diff.abs() / denom).mean().item() * 100.0

    p = prefix or ""
    return {
        f"{p}mse": mse,
        f"{p}rmse": rmse,
        f"{p}mae": mae,
        f"{p}mape": mape,
    }


def compute_structured_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Calcula MSE, RMSE y MAE por horizonte y por canal.

    Las dimensiones esperadas son ``[N, M, D]``: ejemplos, objetivos
    temporales y canales de salida. La función omite celdas no observadas
    mediante ``mask`` y no calcula MAPE porque el benchmark trabaja en una
    escala estandarizada cuyo cero no tiene interpretación porcentual.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            "preds y targets deben tener la misma shape para métricas estructuradas. "
            f"preds={tuple(preds.shape)}, targets={tuple(targets.shape)}"
        )
    if preds.ndim != 3:
        return {}
    if mask is not None and mask.shape != targets.shape:
        raise ValueError(
            "mask debe tener la misma shape que targets. "
            f"mask={tuple(mask.shape)}, targets={tuple(targets.shape)}"
        )

    preds = preds.float()
    targets = targets.float()
    valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0
    metrics: Dict[str, float] = {}

    def add_metrics(name: str, selected: torch.Tensor) -> None:
        if not torch.any(selected):
            return
        diff = preds[selected] - targets[selected]
        mse = (diff ** 2).mean().item()
        metrics[f"{prefix}mse_{name}"] = mse
        metrics[f"{prefix}rmse_{name}"] = mse ** 0.5
        metrics[f"{prefix}mae_{name}"] = diff.abs().mean().item()

    for target_index in range(targets.shape[1]):
        selection = torch.zeros_like(valid)
        selection[:, target_index, :] = valid[:, target_index, :]
        add_metrics(f"target_{target_index}", selection)

    for channel_index in range(targets.shape[2]):
        selection = torch.zeros_like(valid)
        selection[:, :, channel_index] = valid[:, :, channel_index]
        add_metrics(f"channel_{channel_index}", selection)

    return metrics


def compute_gaussian_metrics(
    mean: torch.Tensor,
    targets: torch.Tensor,
    log_scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Métricas propias y de calibración para una Normal heteroscedástica.

    ``log_scale`` representa ``log(sigma)``. La CRPS se calcula en forma
    cerrada y las coberturas usan intervalos centrales nominales. Todas las
    métricas respetan la misma máscara de observación que la pérdida.
    """
    if mean.shape != targets.shape or log_scale.shape != targets.shape:
        raise ValueError(
            "mean, targets y log_scale deben tener la misma shape. "
            f"mean={tuple(mean.shape)}, targets={tuple(targets.shape)}, "
            f"log_scale={tuple(log_scale.shape)}"
        )
    if mask is not None and mask.shape != targets.shape:
        raise ValueError(
            "mask debe tener la misma shape que targets. "
            f"mask={tuple(mask.shape)}, targets={tuple(targets.shape)}"
        )

    mean = mean.float()
    targets = targets.float()
    log_scale = log_scale.float()
    valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0
    if not torch.any(valid):
        return {}

    mu = mean[valid]
    y = targets[valid]
    ls = log_scale[valid].clamp(-12.0, 8.0)
    sigma = ls.exp().clamp_min(1e-8)
    z = (y - mu) / sigma

    nll = ls + 0.5 * z.square() + 0.5 * math.log(2.0 * math.pi)
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))

    metrics: Dict[str, float] = {
        f"{prefix}nll": nll.mean().item(),
        f"{prefix}crps": crps.mean().item(),
        f"{prefix}mean_sigma": sigma.mean().item(),
    }
    # z críticos de intervalos centrales 50%, 80%, 90% y 95%.
    for level, critical in ((50, 0.67448975), (80, 1.28155157), (90, 1.64485363), (95, 1.95996398)):
        covered = (y - mu).abs() <= critical * sigma
        metrics[f"{prefix}coverage_{level}"] = covered.float().mean().item()
        metrics[f"{prefix}coverage_error_{level}"] = abs(
            metrics[f"{prefix}coverage_{level}"] - level / 100.0
        )
    return metrics
