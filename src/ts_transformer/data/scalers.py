from __future__ import annotations

from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class StandardScaler:
    """
    Escalador tipo "standard" (resta media y divide por desviación estándar).

    Se puede usar con np.ndarray o torch.Tensor.
    Guarda la media y desviación estándar para poder hacer inverse_transform.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.eps = eps

    def fit(self, data: ArrayLike, axis: int = 0) -> "StandardScaler":
        """
        Calcula media y desviación estándar a lo largo de `axis`.

        Parameters
        ----------
        data:
            np.ndarray o torch.Tensor.
        axis:
            Eje a lo largo del cual se calcula la estadística.
        """
        x = _to_numpy(data)
        self.mean_ = np.nanmean(x, axis=axis)
        self.std_ = np.nanstd(x, axis=axis)
        # Evitar división por cero
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, data: ArrayLike) -> ArrayLike:
        """
        Aplica (x - mean) / std.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler debe ser fit() antes de transform().")

        if isinstance(data, torch.Tensor):
            device = data.device
            dtype = data.dtype
            x = _to_numpy(data)
            x_scaled = (x - self.mean_) / self.std_
            return torch.as_tensor(x_scaled, device=device, dtype=dtype)
        else:
            x = _to_numpy(data)
            return (x - self.mean_) / self.std_

    def fit_transform(self, data: ArrayLike, axis: int = 0) -> ArrayLike:
        self.fit(data, axis=axis)
        return self.transform(data)

    def inverse_transform(self, data: ArrayLike) -> ArrayLike:
        """
        Revierte la transformación: x = data * std + mean.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler debe ser fit() antes de inverse_transform().")

        if isinstance(data, torch.Tensor):
            device = data.device
            dtype = data.dtype
            x = _to_numpy(data)
            x_inv = x * self.std_ + self.mean_
            return torch.as_tensor(x_inv, device=device, dtype=dtype)
        else:
            x = _to_numpy(data)
            return x * self.std_ + self.mean_


class MinMaxScaler:
    """
    Escalador tipo "min-max": (x - min) / (max - min).

    Se puede usar con np.ndarray o torch.Tensor.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.range_: np.ndarray | None = None
        self.eps = eps

    def fit(self, data: ArrayLike, axis: int = 0) -> "MinMaxScaler":
        x = _to_numpy(data)
        self.min_ = np.nanmin(x, axis=axis)
        self.max_ = np.nanmax(x, axis=axis)
        self.range_ = self.max_ - self.min_
        self.range_ = np.where(self.range_ < self.eps, 1.0, self.range_)
        return self

    def transform(self, data: ArrayLike) -> ArrayLike:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler debe ser fit() antes de transform().")

        if isinstance(data, torch.Tensor):
            device = data.device
            dtype = data.dtype
            x = _to_numpy(data)
            x_scaled = (x - self.min_) / self.range_
            return torch.as_tensor(x_scaled, device=device, dtype=dtype)
        else:
            x = _to_numpy(data)
            return (x - self.min_) / self.range_

    def fit_transform(self, data: ArrayLike, axis: int = 0) -> ArrayLike:
        self.fit(data, axis=axis)
        return self.transform(data)

    def inverse_transform(self, data: ArrayLike) -> ArrayLike:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler debe ser fit() antes de inverse_transform().")

        if isinstance(data, torch.Tensor):
            device = data.device
            dtype = data.dtype
            x = _to_numpy(data)
            x_inv = x * self.range_ + self.min_
            return torch.as_tensor(x_inv, device=device, dtype=dtype)
        else:
            x = _to_numpy(data)
            return x * self.range_ + self.min_