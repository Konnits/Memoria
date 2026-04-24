from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torch import nn, optim


@dataclass
class OptimizerConfig:
    """
    Configuración para el optimizador y el scheduler.

    Atributos principales:
    - optimizer_name:
        * "adam"
        * "adamw"
        * "sgd"
    - lr:
        Learning rate.
    - weight_decay:
        Regularización L2.
    - momentum:
        Sólo relevante para SGD.
    - betas:
        Sólo relevante para Adam/AdamW.
    - scheduler_name:
        * None: sin scheduler.
        * "step": StepLR.
        * "cosine": CosineAnnealingLR.
    - scheduler_step_size:
        Step size para StepLR.
    - scheduler_gamma:
        Factor de multiplicación del LR en StepLR.
    - scheduler_T_max:
        T_max para CosineAnnealingLR.
    """

    optimizer_name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)

    scheduler_name: Optional[Literal["step", "cosine", "cosine_warmup"]] = None
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_T_max: int = 50
    warmup_epochs: int = 0


def _build_adam_like_optimizer(
    optimizer_cls,
    params,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
):
    """Selecciona fused/foreach automáticamente cuando los parámetros están en CUDA."""
    common_kwargs = {
        "lr": lr,
        "betas": betas,
        "weight_decay": weight_decay,
    }

    has_cuda_params = any(p.is_cuda for p in _iter_optimizer_params(params))
    if not has_cuda_params:
        return optimizer_cls(params, **common_kwargs)

    # fused suele ser más rápido en GPUs modernas; fallback seguro a foreach.
    try:
        return optimizer_cls(params, **common_kwargs, fused=True)
    except (TypeError, RuntimeError):
        return optimizer_cls(params, **common_kwargs, foreach=True)


def _iter_optimizer_params(params):
    for item in params:
        if isinstance(item, dict):
            yield from item.get("params", [])
        else:
            yield item


def _build_parameter_groups(
    model: nn.Module,
    weight_decay: float,
):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "time_encoding.time2vec" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not no_decay_params:
        return decay_params

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> optim.Optimizer:
    """
    Construye un optimizador para los parámetros de `model`
    según la configuración proporcionada.
    """
    params = _build_parameter_groups(model, config.weight_decay)

    name = config.optimizer_name.lower()
    if name == "adam":
        return _build_adam_like_optimizer(
            optim.Adam,
            params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif name == "adamw":
        return _build_adam_like_optimizer(
            optim.AdamW,
            params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif name == "sgd":
        return optim.SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizador no soportado: {config.optimizer_name}")


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Scheduler con warmup lineal seguido de cosine annealing.

    Durante las primeras `warmup_epochs` épocas, el LR sube linealmente
    de 0 a base_lr. Luego decae con cosine annealing hasta 0 en T_max épocas.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        T_max: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(warmup_epochs, 0)
        self.T_max = T_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Warmup lineal: escalar de 0 a base_lr
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay después del warmup
            progress = (epoch - self.warmup_epochs) / max(
                1, self.T_max - self.warmup_epochs
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: optim.Optimizer,
    config: OptimizerConfig,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Construye un scheduler de LR opcional según la configuración.

    Retorna:
    - scheduler: instancia de _LRScheduler, o None si no se usa scheduler.
    """
    if config.scheduler_name is None:
        return None

    name = config.scheduler_name.lower()
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_T_max,
        )
    elif name == "cosine_warmup":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            T_max=config.scheduler_T_max,
        )
    else:
        raise ValueError(f"Scheduler no soportado: {config.scheduler_name}")
