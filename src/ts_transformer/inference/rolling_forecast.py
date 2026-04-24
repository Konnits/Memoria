from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Optional, Dict, Any

import numpy as np
import torch

from .predictor import Predictor, ArrayLike


@dataclass
class RollingForecastConfig:
    """
    Configuración para pronósticos rolling (multi-step).

    - mode:
        * "fixed_history": todas las predicciones usan la misma historia
          original (no se alimentan las predicciones como nuevas entradas).
        * En el futuro se puede extender con un modo "autoregressive" que
          vaya incorporando las predicciones al historial.
    """

    mode: str = "fixed_history"


class RollingForecaster:
    """
    Utilidad para realizar pronósticos multi-step usando un Predictor.

    Por ahora implementa un modo "fixed_history", donde:
    - Tienes un historial (past_values, past_timestamps).
    - Tienes una lista de timestamps objetivo futuros.
    - Para cada timestamp, se llama a Predictor.predict_single con la misma
      historia, y se devuelven todas las predicciones en un array.

    Es una envoltura conveniente para no repetir el bucle.
    """

    def __init__(
        self,
        predictor: Predictor,
        config: Optional[RollingForecastConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        predictor:
            Instancia de Predictor ya configurada.
        config:
            Configuración del RollingForecaster.
        """
        if config is None:
            config = RollingForecastConfig()

        self.predictor = predictor
        self.config = config

        if self.config.mode not in ("fixed_history",):
            raise ValueError(
                f"Modo de RollingForecaster no soportado: {self.config.mode}. "
                "Por ahora sólo se admite 'fixed_history'."
            )

    def forecast(
        self,
        past_values: ArrayLike,
        past_timestamps: ArrayLike,
        future_timestamps: Sequence[Union[float, int]],
        past_sensor_ids: Optional[ArrayLike] = None,
        return_torch: bool = False,
    ) -> ArrayLike:
        """
        Realiza un pronóstico multi-step.

        Parameters
        ----------
        past_values:
            Array 2D [L, input_dim] con la historia.
        past_timestamps:
            Array 1D [L] con timestamps de la historia.
        future_timestamps:
            Secuencia de timestamps objetivo para los que se desea predecir.
        past_sensor_ids:
            Array 1D [L] con ids de sensor cuando el modelo usa tokens de evento.
        return_torch:
            Si True, devuelve un torch.Tensor [N_future, output_dim].
            Si False, devuelve np.ndarray.

        Returns
        -------
        preds:
            Array [N_future, output_dim] con las predicciones para cada timestamp.
        """
        if self.config.mode == "fixed_history":
            return self.predictor.predict_multi_targets(
                past_values=past_values,
                past_timestamps=past_timestamps,
                target_timestamps=future_timestamps,
                past_sensor_ids=past_sensor_ids,
                return_torch=return_torch,
            )

        # Bloque para futuras extensiones:
        raise RuntimeError("Modo de RollingForecaster inválido o no implementado.")

    # ------------------------------------------------------------------
    # Helper para construir horizon relativo (si usas offsets en lugar
    # de timestamps absolutos).
    # ------------------------------------------------------------------
    @staticmethod
    def build_future_timestamps_from_offsets(
        last_timestamp: Union[float, int],
        offsets: Sequence[Union[float, int]],
    ) -> np.ndarray:
        """
        Construye una lista de timestamps futuros a partir de un timestamp
        final de la historia y una serie de offsets.

        Ejemplo:
            last_timestamp = 1000.0
            offsets = [900, 1800]   # segundos
            -> future_timestamps = [1900.0, 2800.0]

        Parameters
        ----------
        last_timestamp:
            Último timestamp de la historia.
        offsets:
            Secuencia de offsets (en las mismas unidades de tiempo).

        Returns
        -------
        future_timestamps:
            np.ndarray 1D con los timestamps futuros.
        """
        last = float(last_timestamp)
        offs = np.asarray(offsets, dtype=float)
        return last + offs