"""
Módulos de manejo de datos para el TimeSeriesTransformer.

Incluye:
- Datasets para construir ventanas de historia + target.
- Construcción de secuencias con token target.
- Collate functions para DataLoader.
- Scalers (normalización) para features continuas.
- Funciones de split temporal train/val/test.
"""

from .timeseries_dataset import TimeSeriesDataset, EventTimeSeriesDataset
from .sequence_builder import SequenceBuilder
from .collate import build_collate_fn
from .samplers import BucketBatchSampler
from .scalers import StandardScaler, MinMaxScaler
from .splits import (
    time_series_train_val_test_split,
    split_dataframe_by_time,
)

__all__ = [
    "TimeSeriesDataset",
    "EventTimeSeriesDataset",
    "SequenceBuilder",
    "build_collate_fn",
    "BucketBatchSampler",
    "StandardScaler",
    "MinMaxScaler",
    "time_series_train_val_test_split",
    "split_dataframe_by_time",
]