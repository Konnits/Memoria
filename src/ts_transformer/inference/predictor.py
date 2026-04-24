from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Any

import numpy as np
import torch
from torch import nn

from ..models.time_series_transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)
from ..data.sequence_builder import SequenceBuilder
from ..data.scalers import StandardScaler, MinMaxScaler

ArrayLike = Union[np.ndarray, torch.Tensor]
_OPTIONAL_HEAD_KEYS = {
    "per_target_head.weight",
    "per_target_head.bias",
}


@dataclass
class PredictorConfig:
    """
    Configuración de alto nivel para el Predictor.

    Atributos principales:
    - device: dispositivo sobre el que correr el modelo ("cpu", "cuda", etc.).
    - target_token_value: cómo inicializar los valores del token target
      en la secuencia (debe coincidir con SequenceBuilder).
        * "zeros": vector de ceros.
        * "last": copia del último valor de la historia.
    """

    device: str = "cpu"
    target_token_value: str = "zeros"


class Predictor:
    """
    Envoltorio de alto nivel para usar un TimeSeriesTransformer en inferencia.

    Maneja:
    - Envío del modelo al dispositivo correcto y modo eval().
    - Construcción de la secuencia con token target (via SequenceBuilder).
    - Aplicación opcional de scalers (valor/target) para normalizar y
      des-normalizar predicciones.
    """

    def __init__(
        self,
        model: TimeSeriesTransformer,
        config: Optional[PredictorConfig] = None,
        value_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
        target_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
        sequence_builder: Optional[SequenceBuilder] = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Instancia ya construida de TimeSeriesTransformer.
        config:
            Configuración del Predictor (device, etc.).
        value_scaler:
            Escalador usado sobre las variables de entrada durante el entrenamiento.
            Si se proporciona, se aplicará transform() a los valores de entrada
            y inverse_transform() a las predicciones si target_scaler no se usa.
        target_scaler:
            Escalador usado sobre los targets durante el entrenamiento.
            Si se proporciona, se aplicará inverse_transform() sobre las
            predicciones del modelo.
        sequence_builder:
            Instancia de SequenceBuilder. Si es None, se crea una por defecto
            usando model.input_dim y config.target_token_value.
        """
        if config is None:
            config = PredictorConfig()

        self.config = config
        self.model = model
        self.device = torch.device(config.device)

        self.model.to(self.device)
        self.model.eval()

        self.value_scaler = value_scaler
        self.target_scaler = target_scaler

        if sequence_builder is None:
            use_sensor_ids = bool(model.config.use_sensor_embedding)
            num_target_tokens = model.output_dim if use_sensor_ids else 1
            sequence_builder = SequenceBuilder(
                input_dim=model.input_dim,
                target_token_value=config.target_token_value,  # type: ignore[arg-type]
                use_sensor_ids=use_sensor_ids,
                num_sensors=int(model.config.num_sensors),
                num_target_tokens=num_target_tokens,
                target_sensor_ids=None,
            )
        self.sequence_builder = sequence_builder

    @torch.no_grad()
    def predict_single(
        self,
        past_values: ArrayLike,
        past_timestamps: ArrayLike,
        target_timestamp: Union[float, int],
        past_sensor_ids: Optional[ArrayLike] = None,
        return_torch: bool = False,
    ) -> ArrayLike:
        """
        Predice el valor de salida en un único timestamp objetivo, dada
        una historia de valores y timestamps.

        Parameters
        ----------
        past_values:
            Array 2D [L, input_dim] con la historia de valores.
        past_timestamps:
            Array 1D [L] con los timestamps correspondientes.
        past_sensor_ids:
            Array 1D [L] con ids de sensor para modelos entrenados con
            use_sensor_embedding=True. En modelos densos debe ser None.
        target_timestamp:
            Timestamp objetivo (float o int).
        return_torch:
            Si True, devuelve un torch.Tensor; si False, devuelve np.ndarray.

        Returns
        -------
        pred:
            Vector [output_dim] con la predicción en target_timestamp,
            ya des-normalizada si se configuró target_scaler/value_scaler.
        """
        # Convertir a tensores en CPU para compatibilidad con scalers
        past_values_t = self._to_tensor_2d(past_values, name="past_values")
        past_timestamps_t = self._to_tensor_1d(past_timestamps, name="past_timestamps")
        target_timestamp_t = torch.as_tensor([float(target_timestamp)], dtype=torch.float32)
        past_sensor_ids_t = self._normalize_sensor_ids(
            past_sensor_ids,
            seq_len=past_values_t.shape[0],
        )

        # Aplicar escalado de valores (si corresponde)
        if self.value_scaler is not None:
            # Nuestro scaler acepta np o torch; usamos torch directamente.
            past_values_t = self.value_scaler.transform(past_values_t)  # type: ignore[arg-type]

        # Construir sample al estilo TimeSeriesDataset
        # target_values es un placeholder (no se usa en inferencia)
        dummy_target_values = torch.zeros((1, self.model.output_dim), dtype=torch.float32)
        sample = {
            "past_values": past_values_t,
            "past_timestamps": past_timestamps_t,
            "target_timestamp": target_timestamp_t,
            "target_values": dummy_target_values,
        }
        if past_sensor_ids_t is not None:
            sample["past_sensor_ids"] = past_sensor_ids_t

        # Construir secuencia con token target (sin batch)
        seq = self.sequence_builder(sample)
        # Añadir dimensión de batch
        input_values = seq["input_values"].unsqueeze(0).to(self.device)       # [1, L+1, input_dim]
        input_timestamps = seq["input_timestamps"].unsqueeze(0).to(self.device)  # [1, L+1]
        is_target_mask = seq["is_target_mask"].unsqueeze(0).to(self.device)   # [1, L+1]
        input_sensor_ids = seq.get("input_sensor_ids", None)
        if input_sensor_ids is not None:
            input_sensor_ids = input_sensor_ids.unsqueeze(0).to(self.device)

        # padding_mask = None (asumimos que no hay padding en inferencia single)
        preds = self.model(
            input_values=input_values,
            input_timestamps=input_timestamps,
            is_target_mask=is_target_mask,
            input_sensor_ids=input_sensor_ids,
            padding_mask=None,
            attn_mask=None,
            return_dict=False,
        )  # [1, output_dim]

        # Pasar a CPU para des-escalar
        preds_cpu = preds.detach().cpu()  # [1, output_dim]
        pred_1d = preds_cpu[0]  # [output_dim]

        # Des-escalar
        if self.target_scaler is not None:
            pred_1d = self.target_scaler.inverse_transform(pred_1d)  # type: ignore[arg-type]
        elif self.value_scaler is not None:
            # Caso: el mismo scaler se aplicó a inputs y outputs
            pred_1d = self.value_scaler.inverse_transform(pred_1d)  # type: ignore[arg-type]

        if return_torch:
            return pred_1d
        else:
            return pred_1d.numpy()

    @torch.no_grad()
    def predict_multi_targets(
        self,
        past_values: ArrayLike,
        past_timestamps: ArrayLike,
        target_timestamps: Sequence[Union[float, int]],
        past_sensor_ids: Optional[ArrayLike] = None,
        return_torch: bool = False,
    ) -> ArrayLike:
        """
        Predice sobre múltiples timestamps objetivo, reutilizando el mismo
        historial. Realiza una llamada al modelo por cada timestamp objetivo.

        Parameters
        ----------
        past_values:
            Array 2D [L, input_dim].
        past_timestamps:
            Array 1D [L].
        target_timestamps:
            Secuencia de timestamps objetivo.
        past_sensor_ids:
            Array 1D [L] con ids de sensor cuando el modelo usa tokens de evento.
        return_torch:
            Si True, devuelve un torch.Tensor [N_targets, output_dim].
            Si False, devuelve np.ndarray del mismo shape.

        Returns
        -------
        preds:
            Array [N_targets, output_dim] con las predicciones para cada timestamp.
        """
        preds_list = []
        for t_star in target_timestamps:
            pred = self.predict_single(
                past_values=past_values,
                past_timestamps=past_timestamps,
                past_sensor_ids=past_sensor_ids,
                target_timestamp=t_star,
                return_torch=True,
            )  # [output_dim] tensor
            preds_list.append(pred.unsqueeze(0))

        preds_all = torch.cat(preds_list, dim=0)  # [N_targets, output_dim]

        if return_torch:
            return preds_all
        else:
            return preds_all.numpy()

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor_2d(x: ArrayLike, name: str) -> torch.Tensor:
        x_arr = torch.as_tensor(x, dtype=torch.float32)
        if x_arr.ndim != 2:
            raise ValueError(
                f"{name} debe ser 2D [L, D], pero recibió shape {tuple(x_arr.shape)}."
            )
        return x_arr

    @staticmethod
    def _to_tensor_1d(x: ArrayLike, name: str) -> torch.Tensor:
        x_arr = torch.as_tensor(x, dtype=torch.float32)
        if x_arr.ndim != 1:
            raise ValueError(
                f"{name} debe ser 1D [L], pero recibió shape {tuple(x_arr.shape)}."
            )
        return x_arr

    def _normalize_sensor_ids(
        self,
        sensor_ids: Optional[ArrayLike],
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        use_sensor_ids = bool(getattr(self.sequence_builder, "use_sensor_ids", False))

        if not use_sensor_ids:
            if sensor_ids is not None:
                raise ValueError(
                    "Este Predictor no usa sensor_ids; elimina past_sensor_ids de la llamada."
                )
            return None

        if sensor_ids is None:
            num_sensors = int(getattr(self.sequence_builder, "num_sensors", 0))
            if num_sensors == 1:
                return torch.zeros(seq_len, dtype=torch.long)

            raise ValueError(
                "El modelo fue entrenado con use_sensor_embedding=True; debes pasar "
                "past_sensor_ids en inferencia cuando hay más de un sensor."
            )

        sensor_ids_t = torch.as_tensor(sensor_ids, dtype=torch.long)
        if sensor_ids_t.ndim != 1:
            raise ValueError(
                "past_sensor_ids debe ser 1D [L]."
            )
        if sensor_ids_t.shape[0] != seq_len:
            raise ValueError(
                "past_sensor_ids debe tener la misma longitud que past_values/past_timestamps."
            )
        return sensor_ids_t

    # ------------------------------------------------------------------
    # Helpers opcionales para cargar desde checkpoints
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        value_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
        target_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
        target_token_value: str = "zeros",
        feature_columns: Optional[Sequence[str]] = None,
        target_columns: Optional[Sequence[str]] = None,
    ) -> "Predictor":
        """
        Crea un Predictor a partir de un checkpoint guardado con torch.save.

        Se asume que el checkpoint es un dict con al menos:
            {
                "config": dict compatible con TimeSeriesTransformerConfig,
                "model_state_dict": state_dict del modelo
            }

        Los scalers se pasan explícitamente (no se cargan del checkpoint
        por defecto, aunque podrías extender este método para hacerlo).
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "config" not in ckpt or "model_state_dict" not in ckpt:
            raise ValueError(
                "El checkpoint debe contener al menos 'config' y 'model_state_dict'."
            )

        config_dict = ckpt["config"]
        config_model = TimeSeriesTransformerConfig(**config_dict)

        model = TimeSeriesTransformer(config_model)
        _load_model_state_dict(model, ckpt["model_state_dict"], checkpoint_path)

        pred_config = PredictorConfig(
            device=device,
            target_token_value=target_token_value,
        )
        sequence_builder = _build_sequence_builder_for_model(
            model,
            target_token_value=target_token_value,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )

        return cls(
            model=model,
            config=pred_config,
            value_scaler=value_scaler,
            target_scaler=target_scaler,
            sequence_builder=sequence_builder,
        )


def _load_checkpoint_state_dict(checkpoint_path: str, device: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(
            f"Formato de checkpoint no soportado en {checkpoint_path}."
        )

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"El checkpoint {checkpoint_path} no contiene un state_dict válido."
        )

    return state_dict


def _load_model_state_dict(
    model: TimeSeriesTransformer,
    state_dict: Dict[str, torch.Tensor],
    checkpoint_path: str,
) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing_required = [key for key in missing if key not in _OPTIONAL_HEAD_KEYS]
    unexpected_required = [key for key in unexpected if key not in _OPTIONAL_HEAD_KEYS]

    if missing_required or unexpected_required:
        raise RuntimeError(
            "Error al cargar el checkpoint "
            f"{checkpoint_path}. Missing={missing_required}, unexpected={unexpected_required}"
        )


def _build_sequence_builder_for_model(
    model: TimeSeriesTransformer,
    target_token_value: str,
    feature_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
) -> SequenceBuilder:
    target_sensor_ids = None
    use_sensor_ids = bool(model.config.use_sensor_embedding)
    num_target_tokens = model.output_dim if use_sensor_ids else 1

    if use_sensor_ids and feature_columns is not None and target_columns is not None:
        target_sensor_ids = [
            feature_columns.index(col) if col in feature_columns else int(model.config.num_sensors)
            for col in target_columns
        ]

    return SequenceBuilder(
        input_dim=model.input_dim,
        target_token_value=target_token_value,  # type: ignore[arg-type]
        use_sensor_ids=use_sensor_ids,
        num_sensors=int(model.config.num_sensors),
        num_target_tokens=num_target_tokens,
        target_sensor_ids=target_sensor_ids,
    )

def build_predictor_from_experiment(
    exp_dir: str,
    device: str = "cpu",
    target_token_value: str = "zeros",
) -> "Predictor":
    """
    Construye un Predictor a partir de una carpeta de experimento
    generada por train.py (best_model.pt, model_config.yaml, scalers.pt).

    Parameters
    ----------
    exp_dir:
        Carpeta del experimento.
    device:
        Dispositivo ("cpu", "cuda", "cuda:0", ...).
    target_token_value:
        Debe coincidir con el usado en entrenamiento
        (normalmente "zeros" como en train.py).

    Returns
    -------
    predictor:
        Instancia de Predictor lista para usar.
    """
    import os
    from ..utils import load_model_config

    model_cfg_path = os.path.join(exp_dir, "model_config.yaml")
    ckpt_path = os.path.join(exp_dir, "best_model.pt")
    scalers_path = os.path.join(exp_dir, "scalers.pt")

    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"No se encontró {model_cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No se encontró {ckpt_path}")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"No se encontró {scalers_path}")

    # 1) Config del modelo
    model_cfg = load_model_config(model_cfg_path)
    model = TimeSeriesTransformer(model_cfg)

    # 2) Pesos
    state_dict = _load_checkpoint_state_dict(ckpt_path, device=device)
    _load_model_state_dict(model, state_dict, ckpt_path)

    # 3) Scalers
    scalers_obj = torch.load(scalers_path, map_location="cpu", weights_only=False)
    value_scaler = scalers_obj["value_scaler"]
    target_scaler = scalers_obj["target_scaler"]
    feature_columns = scalers_obj.get("feature_columns", None)
    target_columns = scalers_obj.get("target_columns", None)

    # 4) Predictor
    pred_cfg = PredictorConfig(
        device=device,
        target_token_value=target_token_value,
    )
    sequence_builder = _build_sequence_builder_for_model(
        model,
        target_token_value=target_token_value,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    return Predictor(
        model=model,
        config=pred_cfg,
        value_scaler=value_scaler,
        target_scaler=target_scaler,
        sequence_builder=sequence_builder,
    )