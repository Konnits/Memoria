from __future__ import annotations

from typing import Dict

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