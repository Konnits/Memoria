from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .dilate_loss import DILATELoss


def mse_loss() -> nn.Module:
    """L2 / Mean Squared Error por defecto para regresión."""
    return nn.MSELoss()


def mae_loss() -> nn.Module:
    """L1 / Mean Absolute Error."""
    return nn.L1Loss()


def huber_loss(delta: float = 1.0) -> nn.Module:
    """Pérdida de Huber (robusta a outliers)."""

    return nn.SmoothL1Loss(beta=delta)  # beta es el parámetro de transición


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    """
    Retorna una función de pérdida según su nombre.

    Parameters
    ----------
    name:
        Nombre de la pérdida. Opciones:
        - "mse"
        - "mae"
        - "huber"
        - "dilate"
    kwargs:
        Parámetros adicionales (por ejemplo delta para huber).

    Returns
    -------
    loss_fn:
        Instancia de nn.Module con la pérdida seleccionada.
    """
    name = name.lower()
    if name == "mse":
        return mse_loss()
    elif name == "mae":
        return mae_loss()
    elif name == "huber":
        delta = float(kwargs.get("delta", 1.0))
        return huber_loss(delta=delta)
    elif name == "dilate":
        return DILATELoss(
            alpha=float(kwargs.get("alpha", 0.5)),
            gamma=float(kwargs.get("gamma", 0.01)),
            normalize_shape=bool(kwargs.get("normalize_shape", True)),
            normalize_temporal=bool(kwargs.get("normalize_temporal", True)),
        )
    else:
        raise ValueError(f"Función de pérdida no soportada: {name}")
