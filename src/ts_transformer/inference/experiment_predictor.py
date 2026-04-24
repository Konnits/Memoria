from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from ..models.time_series_transformer import TimeSeriesTransformer
from ..utils import load_model_config
from .predictor import Predictor, PredictorConfig, _build_sequence_builder_for_model, _load_checkpoint_state_dict, _load_model_state_dict
from .rolling_forecast import RollingForecaster

TimestampScalar = Union[int, float, np.datetime64, pd.Timestamp, datetime]
TimestampLike = Union[TimestampScalar, Sequence[TimestampScalar], np.ndarray, pd.Series, pd.Index]


@dataclass(frozen=True)
class ExperimentArtifacts:
    experiment_dir: Path
    checkpoint_path: Path
    model_config_path: Path
    scalers_path: Path


class ExperimentPredictor:
    """
    Contenedor de alto nivel para cargar un experimento entrenado y predecir.

    Permite:
    - Cargar desde carpeta de experimento o desde una ruta directa al checkpoint.
    - Reconstruir modelo, scalers y metadatos de columnas.
    - Pedir predicciones para uno o varios timestamps futuros.
    - Aceptar timestamps numéricos o datetimes, y devolver opcionalmente un DataFrame.
    """

    def __init__(
        self,
        predictor: Predictor,
        artifacts: ExperimentArtifacts,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        time_column: str,
        use_event_tokens: bool,
    ) -> None:
        self.predictor = predictor
        self.forecaster = RollingForecaster(predictor)
        self.artifacts = artifacts
        self.feature_columns = tuple(feature_columns)
        self.target_columns = tuple(target_columns)
        self.time_column = time_column
        self.use_event_tokens = bool(use_event_tokens)

    @property
    def device(self) -> str:
        return str(self.predictor.device)

    @property
    def model(self) -> TimeSeriesTransformer:
        return self.predictor.model

    @classmethod
    def from_experiment_dir(
        cls,
        experiment_dir: Union[str, Path],
        device: str = "cpu",
        target_token_value: str = "zeros",
    ) -> "ExperimentPredictor":
        artifacts = cls._resolve_artifacts(experiment_dir)
        return cls._from_artifacts(
            artifacts=artifacts,
            device=device,
            target_token_value=target_token_value,
        )

    @classmethod
    def from_model_path(
        cls,
        model_path: Union[str, Path],
        device: str = "cpu",
        target_token_value: str = "zeros",
    ) -> "ExperimentPredictor":
        model_path_obj = Path(model_path)
        if not model_path_obj.exists() or not model_path_obj.is_file():
            raise FileNotFoundError(f"No se encontró el checkpoint: {model_path_obj}")

        artifacts = cls._resolve_artifacts(model_path_obj.parent, checkpoint_path=model_path_obj)
        return cls._from_artifacts(
            artifacts=artifacts,
            device=device,
            target_token_value=target_token_value,
        )

    @classmethod
    def _from_artifacts(
        cls,
        artifacts: ExperimentArtifacts,
        device: str,
        target_token_value: str,
    ) -> "ExperimentPredictor":
        model_cfg = load_model_config(str(artifacts.model_config_path))
        model = TimeSeriesTransformer(model_cfg)

        state_dict = _load_checkpoint_state_dict(str(artifacts.checkpoint_path), device=device)
        _load_model_state_dict(model, state_dict, str(artifacts.checkpoint_path))

        scalers_obj = torch.load(str(artifacts.scalers_path), map_location="cpu", weights_only=False)
        feature_columns = tuple(scalers_obj.get("feature_columns") or [])
        target_columns = tuple(scalers_obj.get("target_columns") or [])
        time_column = str(scalers_obj.get("time_column") or "timestamp")

        sequence_builder = _build_sequence_builder_for_model(
            model,
            target_token_value=target_token_value,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )

        predictor = Predictor(
            model=model,
            config=PredictorConfig(device=device, target_token_value=target_token_value),
            value_scaler=scalers_obj.get("value_scaler"),
            target_scaler=scalers_obj.get("target_scaler"),
            sequence_builder=sequence_builder,
        )

        return cls(
            predictor=predictor,
            artifacts=artifacts,
            feature_columns=feature_columns,
            target_columns=target_columns,
            time_column=time_column,
            use_event_tokens=bool(model.config.use_sensor_embedding),
        )

    @staticmethod
    def _resolve_artifacts(
        experiment_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> ExperimentArtifacts:
        exp_dir = Path(experiment_dir)
        if not exp_dir.exists() or not exp_dir.is_dir():
            raise FileNotFoundError(f"No se encontró la carpeta del experimento: {exp_dir}")

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else exp_dir / "best_model.pt"
        model_config = exp_dir / "model_config.yaml"
        scalers = exp_dir / "scalers.pt"

        for path in (checkpoint, model_config, scalers):
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el artefacto requerido: {path}")

        return ExperimentArtifacts(
            experiment_dir=exp_dir,
            checkpoint_path=checkpoint,
            model_config_path=model_config,
            scalers_path=scalers,
        )

    def predict(
        self,
        past_values: Any,
        past_timestamps: TimestampLike,
        future_timestamps: TimestampLike,
        past_sensor_ids: Optional[Any] = None,
        return_dataframe: bool = False,
        return_torch: bool = False,
    ) -> Any:
        values = self._coerce_past_values(past_values)
        history_timestamps = self._coerce_timestamp_array(past_timestamps, name="past_timestamps")
        sensor_ids = self._coerce_sensor_ids(past_sensor_ids, seq_len=values.shape[0])

        if values.shape[0] != history_timestamps.shape[0]:
            raise ValueError(
                "past_values y past_timestamps deben tener la misma longitud."
            )

        future_numeric, future_original = self._coerce_future_timestamps(future_timestamps)

        if future_numeric.shape[0] == 1:
            preds = self.predictor.predict_single(
                past_values=values,
                past_timestamps=history_timestamps,
                past_sensor_ids=sensor_ids,
                target_timestamp=float(future_numeric[0]),
                return_torch=return_torch,
            )
            if return_dataframe:
                preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
                return self._build_prediction_frame(future_original, preds_np.reshape(1, -1))
            return preds

        preds = self.forecaster.forecast(
            past_values=values,
            past_timestamps=history_timestamps,
            future_timestamps=future_numeric.tolist(),
            return_torch=return_torch,
            past_sensor_ids=sensor_ids,
        )

        if return_dataframe:
            preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
            return self._build_prediction_frame(future_original, preds_np)
        return preds

    def predict_from_offsets(
        self,
        past_values: Any,
        past_timestamps: TimestampLike,
        future_offsets: Sequence[Union[int, float]],
        past_sensor_ids: Optional[Any] = None,
        return_dataframe: bool = False,
        return_torch: bool = False,
    ) -> Any:
        history_timestamps = self._coerce_timestamp_array(past_timestamps, name="past_timestamps")
        if history_timestamps.size == 0:
            raise ValueError("past_timestamps no puede estar vacío.")

        future_numeric = RollingForecaster.build_future_timestamps_from_offsets(
            last_timestamp=float(history_timestamps[-1]),
            offsets=future_offsets,
        )
        return self.predict(
            past_values=past_values,
            past_timestamps=history_timestamps,
            future_timestamps=future_numeric,
            past_sensor_ids=past_sensor_ids,
            return_dataframe=return_dataframe,
            return_torch=return_torch,
        )

    def _coerce_past_values(self, past_values: Any) -> np.ndarray:
        if isinstance(past_values, pd.DataFrame):
            if self.use_event_tokens:
                if len(self.feature_columns) == 1 and self.feature_columns[0] in past_values.columns:
                    arr = past_values[[self.feature_columns[0]]].to_numpy(dtype=np.float32)
                else:
                    raise ValueError(
                        "Para modelos con event tokens, pasa una serie/array [L] o [L, 1], "
                        "o un DataFrame con la columna de feature correspondiente."
                    )
            else:
                missing = [col for col in self.feature_columns if col not in past_values.columns]
                if missing:
                    raise ValueError(
                        f"Faltan columnas de entrada en past_values: {missing}."
                    )
                arr = past_values[list(self.feature_columns)].to_numpy(dtype=np.float32)
        elif isinstance(past_values, pd.Series):
            arr = past_values.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(past_values, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("past_values debe ser 1D o 2D.")

        expected_dim = 1 if self.use_event_tokens else self.model.input_dim
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"past_values debe tener shape [L, {expected_dim}], pero recibió {arr.shape}."
            )
        return arr

    def _coerce_sensor_ids(self, past_sensor_ids: Optional[Any], seq_len: int) -> Optional[np.ndarray]:
        if not self.use_event_tokens:
            if past_sensor_ids is not None:
                raise ValueError("Este modelo no usa sensor_ids.")
            return None

        if past_sensor_ids is None:
            if int(self.model.config.num_sensors) == 1:
                return np.zeros(seq_len, dtype=np.int64)
            return None

        arr = np.asarray(past_sensor_ids, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("past_sensor_ids debe ser 1D [L].")
        if arr.shape[0] != seq_len:
            raise ValueError("past_sensor_ids debe tener la misma longitud que past_values.")
        return arr

    def _coerce_future_timestamps(self, future_timestamps: TimestampLike) -> tuple[np.ndarray, np.ndarray]:
        arr = self._coerce_timestamp_array(future_timestamps, name="future_timestamps")
        original = self._coerce_original_timestamp_array(future_timestamps)
        if arr.shape[0] != original.shape[0]:
            raise ValueError("No se pudieron alinear los future_timestamps originales y numéricos.")
        return arr, original

    @staticmethod
    def _coerce_original_timestamp_array(values: TimestampLike) -> np.ndarray:
        if isinstance(values, (pd.Series, pd.Index)):
            return values.to_numpy()
        if np.isscalar(values):
            return np.asarray([values])
        return np.asarray(values)

    @staticmethod
    def _coerce_timestamp_array(values: TimestampLike, name: str) -> np.ndarray:
        if isinstance(values, (pd.Series, pd.Index)):
            arr = values.to_numpy()
        elif np.isscalar(values):
            arr = np.asarray([values])
        else:
            arr = np.asarray(values)

        if arr.ndim != 1:
            raise ValueError(f"{name} debe ser un array 1D o un escalar.")

        if np.issubdtype(arr.dtype, np.datetime64):
            return (arr.astype("datetime64[ns]").astype(np.int64) / 1e9).astype(np.float32)

        if arr.dtype == object:
            try:
                dt = pd.to_datetime(arr, errors="raise")
            except Exception:
                return arr.astype(np.float32)
            return (dt.astype("int64") / 1e9).to_numpy(dtype=np.float32)

        return arr.astype(np.float32)

    def _build_prediction_frame(
        self,
        future_timestamps: np.ndarray,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        if predictions.ndim != 2:
            raise ValueError("predictions debe ser 2D [N, output_dim].")

        data = {self.time_column: future_timestamps}
        for idx, col in enumerate(self.target_columns or [f"target_{i}" for i in range(predictions.shape[1])]):
            data[str(col)] = predictions[:, idx]
        return pd.DataFrame(data)
