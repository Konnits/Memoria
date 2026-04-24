from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def time_series_train_val_test_split(
    timestamps: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera índices (enteros) para un split temporal train/val/test.

    Supone que `timestamps` está ya ordenado por tiempo.

    Parameters
    ----------
    timestamps:
        Array 1D con los timestamps (se usa sólo la longitud).
    train_ratio:
        Proporción de datos para entrenamiento.
    val_ratio:
        Proporción de datos para validación.
        El resto se asigna a test.

    Returns
    -------
    train_idx, val_idx, test_idx:
        Arrays de índices para cada split.
    """
    ts = np.asarray(timestamps)
    n = ts.shape[0]

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio debe estar en (0,1).")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio debe estar en [0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio debe ser < 1.0.")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Splits demasiado pequeños: n_train={n_train}, n_val={n_val}, n_test={n_test} para n={n}."
        )

    indices = np.arange(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def split_dataframe_by_time(
    df: pd.DataFrame,
    time_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en train/val/test respetando el orden temporal.

    Parameters
    ----------
    df:
        DataFrame con los datos.
    time_column:
        Nombre de la columna que contiene el timestamp (datetime o numérico).
    train_ratio:
        Proporción de datos para entrenamiento.
    val_ratio:
        Proporción de datos para validación.

    Returns
    -------
    df_train, df_val, df_test:
        DataFrames particionados.
    """
    if time_column not in df.columns:
        raise ValueError(f"La columna de tiempo '{time_column}' no existe en el DataFrame.")

    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    timestamps = df_sorted[time_column].to_numpy()

    train_idx, val_idx, test_idx = time_series_train_val_test_split(
        timestamps, train_ratio=train_ratio, val_ratio=val_ratio
    )

    df_train = df_sorted.iloc[train_idx].reset_index(drop=True)
    df_val = df_sorted.iloc[val_idx].reset_index(drop=True)
    df_test = df_sorted.iloc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test