"""
Módulos de entrenamiento para el TimeSeriesTransformer.

Incluye:
- Definición de funciones de pérdida.
- Métricas de evaluación.
- Construcción de optimizadores y schedulers.
- Bucle de entrenamiento de alto nivel (Trainer).
"""

from .dilate_loss import DILATELoss
from .losses import get_loss_fn
from .metrics import compute_regression_metrics
from .optimizers import OptimizerConfig, build_optimizer, build_scheduler
from .train_loop import TrainingConfig, Trainer, train_model

__all__ = [
    "get_loss_fn",
    "DILATELoss",
    "compute_regression_metrics",
    "OptimizerConfig",
    "build_optimizer",
    "build_scheduler",
    "TrainingConfig",
    "Trainer",
    "train_model",
]
