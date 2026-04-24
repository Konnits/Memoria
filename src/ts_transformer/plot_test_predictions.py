from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ts_transformer.data import (
    TimeSeriesDataset,
    EventTimeSeriesDataset,
    split_dataframe_by_time,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.data.sequence_builder import SequenceBuilder
from ts_transformer.data.collate import build_collate_fn
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.inference.predictor import _load_model_state_dict
from ts_transformer.utils import (
    load_data_config,
)
from ts_transformer.utils.config import load_model_config
from ts_transformer.utils.logging import setup_logging, get_logger


def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    """
    Misma lógica que en train.py:
    - datetime64 -> segundos desde epoch
    - numérico -> float directamente
    """
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    else:
        return col.astype("float32").to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graficar predicción vs target real en el set de test."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/toy_example.yaml",
        help="Ruta al YAML de configuración de datos.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/exp_default",
        help="Carpeta del experimento (donde están best_model.pt, scalers.pt, model_config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Dispositivo ("cuda", "cpu"). Si es None, usa cuda si está disponible.',
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Máximo número de puntos a mostrar en la gráfica (para no explotar la vista).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp_dir = args.experiment_dir
    os.makedirs(exp_dir, exist_ok=True)

    # Logging sencillo (a consola)
    log_file = os.path.join(exp_dir, "plot_test.log")
    setup_logging(log_file=log_file)
    logger = get_logger("plot_test")

    # ------------------------------------------------------------------
    # Cargar configs y datos
    # ------------------------------------------------------------------
    logger.info(f"Cargando DataConfig desde {args.data_config}...")
    data_cfg = load_data_config(args.data_config)

    logger.info(f"Leyendo CSV de datos desde {data_cfg.csv_path}...")
    df = pd.read_csv(data_cfg.csv_path)

    time_col = data_cfg.time_column
    feature_cols = data_cfg.feature_columns
    target_cols = data_cfg.target_columns

    if time_col not in df.columns:
        raise ValueError(f"La columna de tiempo '{time_col}' no existe en el CSV.")
    for col in feature_cols + target_cols:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el CSV.")

    df = df.sort_values(time_col).reset_index(drop=True)

    # Split train/val/test igual que en train.py
    from ts_transformer.data.splits import split_dataframe_by_time as split_df_time

    logger.info("Realizando split temporal train/val/test...")
    df_train, df_val, df_test = split_df_time(
        df,
        time_column=time_col,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
    )

    logger.info(
        f"  train: {len(df_train)} filas, "
        f"val: {len(df_val)} filas, "
        f"test: {len(df_test)} filas"
    )

    # Pasar df_test a arrays
    def df_to_arrays(df_part: pd.DataFrame):
        ts = _timestamps_to_float(df_part[time_col])
        X = df_part[feature_cols].to_numpy(dtype="float32")
        y = df_part[target_cols].to_numpy(dtype="float32")
        return ts, X, y

    ts_test, X_test, y_test = df_to_arrays(df_test)

    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]
    logger.info(f"Dimensiones (test): input_dim={input_dim}, output_dim={output_dim}")

    # ------------------------------------------------------------------
    # Cargar scalers y model_config del experimento
    # ------------------------------------------------------------------
    scalers_path = os.path.join(exp_dir, "scalers.pt")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(
            f"No se encontró {scalers_path}. ¿Ejecutaste primero el entrenamiento?"
        )

    logger.info(f"Cargando scalers desde {scalers_path}...")
    scalers_obj = torch.load(scalers_path, map_location="cpu", weights_only=False)
    value_scaler = scalers_obj["value_scaler"]
    target_scaler = scalers_obj["target_scaler"]

    # Escalar test igual que train/val
    X_test_scaled = value_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    values_test = np.concatenate([X_test_scaled, y_test_scaled], axis=1)

    # ------------------------------------------------------------------
    # Construir dataset y dataloader de test
    # ------------------------------------------------------------------
    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length,
        target_offset=data_cfg.target_offset,
        stride=data_cfg.stride,
        min_history_length=data_cfg.min_history_length,
        target_offset_choices=data_cfg.target_offset_choices,
        target_offset_min=data_cfg.target_offset_min,
        target_offset_max=data_cfg.target_offset_max,
        num_targets=data_cfg.num_targets,
    )

    if data_cfg.use_event_tokens:
        target_sensor_ids = [
            feature_cols.index(c) if c in feature_cols else input_dim
            for c in target_cols
        ]
        seq_builder = SequenceBuilder(
            input_dim=1,
            target_token_value="zeros",
            use_sensor_ids=True,
            num_sensors=input_dim,
            num_target_tokens=output_dim,
            target_sensor_ids=target_sensor_ids,
        )
        ds_test = EventTimeSeriesDataset(
            values=X_test_scaled,
            timestamps=ts_test,
            targets=y_test_scaled,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_builder=seq_builder,
        )
    else:
        seq_builder = SequenceBuilder(
            input_dim=input_dim,
            target_token_value="zeros",  # debe coincidir con train.py
            use_sensor_ids=False,
            num_sensors=0,
            num_target_tokens=1,
            target_sensor_ids=None,
        )
        ds_test = TimeSeriesDataset(
            values=values_test,
            timestamps=ts_test,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            targets=None,  # se toman de values a partir de input_dim
            sequence_builder=seq_builder,
        )

    collate_fn = build_collate_fn(
        pad_to_max_length=True,
    )

    from torch.utils.data import DataLoader

    _nw = data_cfg.num_workers
    test_loader = DataLoader(
        ds_test,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=_nw,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_nw > 0,
    )

    logger.info(f"DataLoader de test listo: test_batches={len(test_loader)}")

    # ------------------------------------------------------------------
    # Cargar modelo entrenado (mejor checkpoint)
    # ------------------------------------------------------------------
    model_cfg_path = os.path.join(exp_dir, "model_config.yaml")
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(
            f"No se encontró {model_cfg_path}. ¿Ejecutaste primero el entrenamiento?"
        )

    logger.info(f"Cargando TimeSeriesTransformerConfig desde {model_cfg_path}...")
    model_cfg = load_model_config(model_cfg_path)

    # Por si acaso, asegurar input_dim/output_dim correctos
    model_cfg.input_dim = 1 if data_cfg.use_event_tokens else input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(data_cfg.use_event_tokens)
    model_cfg.num_sensors = input_dim if data_cfg.use_event_tokens else 0

    model = TimeSeriesTransformer(model_cfg)

    ckpt_path = os.path.join(exp_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No se encontró {ckpt_path}. Asegúrate de que el entrenamiento guardó el mejor modelo."
        )

    logger.info(f"Cargando pesos del modelo desde {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_model_state_dict(model, ckpt["model_state_dict"], ckpt_path)

    # Dispositivo
    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info(f"Usando device = {device_str}")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Inferencia sobre test
    # ------------------------------------------------------------------
    all_preds_scaled = []
    all_targets_scaled = []
    all_target_times = []

    logger.info("Realizando inferencia sobre el set de test...")

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            input_timestamps = batch["input_timestamps"].to(device)
            is_target_mask = batch["is_target_mask"].to(device)
            input_sensor_ids = batch.get("input_sensor_ids", None)
            if input_sensor_ids is not None:
                input_sensor_ids = input_sensor_ids.to(device)
            target_values = batch["target_values"]  # en CPU (scaled)
            padding_mask = batch.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            preds = model(
                input_values=input_values,
                input_timestamps=input_timestamps,
                is_target_mask=is_target_mask,
                input_sensor_ids=input_sensor_ids,
                padding_mask=padding_mask,
                attn_mask=None,
                return_dict=False,
            )  # [B, D_out]

            all_preds_scaled.append(preds.detach().cpu())
            all_targets_scaled.append(target_values.detach().cpu())
            all_target_times.append(batch["target_timestamps"].detach().cpu())

    if not all_preds_scaled:
        logger.warning("No se obtuvieron muestras en test_loader.")
        return

    preds_scaled_t = torch.cat(all_preds_scaled, dim=0)
    targets_scaled_t = torch.cat(all_targets_scaled, dim=0)
    target_times_t = torch.cat(all_target_times, dim=0)

    if preds_scaled_t.ndim == 3:
        preds_scaled = preds_scaled_t.reshape(-1, preds_scaled_t.shape[-1]).numpy()
    elif preds_scaled_t.ndim == 2:
        preds_scaled = preds_scaled_t.numpy()
    else:
        raise ValueError(f"Shape de predicciones no soportada: {tuple(preds_scaled_t.shape)}")

    if targets_scaled_t.ndim == 3:
        targets_scaled = targets_scaled_t.reshape(-1, targets_scaled_t.shape[-1]).numpy()
    elif targets_scaled_t.ndim == 2:
        targets_scaled = targets_scaled_t.numpy()
    else:
        raise ValueError(f"Shape de targets no soportada: {tuple(targets_scaled_t.shape)}")

    if target_times_t.ndim == 2:
        target_times_float = target_times_t.reshape(-1).numpy()
    elif target_times_t.ndim == 1:
        target_times_float = target_times_t.numpy()
    else:
        raise ValueError(f"Shape de timestamps target no soportada: {tuple(target_times_t.shape)}")

    # Des-escalar a espacio original
    preds = target_scaler.inverse_transform(preds_scaled)      # [N, D_out]
    targets = target_scaler.inverse_transform(targets_scaled)  # [N, D_out]

    # Convertir timestamps a datetime
    time_index = pd.to_datetime(target_times_float.astype("float64"), unit="s")

    # Si hay muchos puntos, recortamos para la gráfica
    N = preds.shape[0]
    if args.max_points is not None and N > args.max_points:
        logger.info(
            f"N={N} puntos en test; se mostrarán sólo los últimos {args.max_points} en la gráfica."
        )
        start = N - args.max_points
        time_index = time_index[start:]
        preds = preds[start:]
        targets = targets[start:]

    # ------------------------------------------------------------------
    # Graficar (para 1 dimensión de salida; si tienes más, puedes extender esto)
    # ------------------------------------------------------------------
    logger.info("Generando gráfica predicción vs target real (test)...")

    plt.figure(figsize=(12, 5))
    plt.plot(time_index, targets[:, 0], label="Valor real (test)")
    plt.plot(time_index, preds[:, 0], label="Predicción modelo (test)", alpha=0.8)
    plt.xlabel("Tiempo")
    plt.ylabel(target_cols[0] if len(target_cols) == 1 else "Valor")
    plt.title("Predicción vs valor real en set de test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logger.info("Gráfica generada correctamente.")


if __name__ == "__main__":
    main()