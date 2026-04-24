from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml

from ts_transformer.data import (
    TimeSeriesDataset,
    EventTimeSeriesDataset,
    SequenceBuilder,
    build_collate_fn,
    BucketBatchSampler,
    StandardScaler,
    split_dataframe_by_time,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.training import TrainingConfig, Trainer
from ts_transformer.utils import (
    DataConfig,
    load_data_config,
    load_model_config,
    load_training_config,
    setup_logging,
    get_logger,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento del TimeSeriesTransformer con token target."
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/transformer_base.yaml",
        help="Ruta al YAML de configuración del modelo.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/toy_example.yaml",
        help="Ruta al YAML de configuración de datos.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training/default.yaml",
        help="Ruta al YAML de configuración de entrenamiento.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=("pretrain", "finetune"),
        default="finetune",
        help="Etapa de entrenamiento: pretrain o finetune.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint opcional para inicializar pesos antes de entrenar. "
            "Acepta formato {'model_state_dict': ...} o state_dict directo."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Nombre del experimento para guardar artefactos en experiments/<nombre>. "
            "Si no se define, se usa checkpoint_dir del training-config."
        ),
    )
    return parser.parse_args()


def _resolve_experiment_dir(
    checkpoint_dir_from_cfg: str | None,
    experiment_name: str | None,
) -> str:
    """
    Resuelve la carpeta final del experimento.

    Prioridad:
    1) --experiment-name  -> experiments/<name>
    2) checkpoint_dir del YAML
    3) experiments/exp_default
    """
    if experiment_name is not None and experiment_name.strip() != "":
        safe_name = experiment_name.strip().replace(" ", "_")
        return os.path.join("experiments", safe_name)

    return checkpoint_dir_from_cfg or "experiments/exp_default"


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Carga un checkpoint y extrae su state_dict.

    Soporta:
    - dict con clave "model_state_dict"
    - state_dict directo (dict de tensores)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

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


def _load_compatible_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    logger,
) -> None:
    """
    Inicializa el modelo con pesos compatibles por nombre y shape.

    Esto evita fallar cuando cambia la cabeza de salida entre pretrain/finetune.
    """
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    model_state = model.state_dict()

    compatible = {}
    skipped = []

    for key, tensor in state_dict.items():
        if key in model_state and model_state[key].shape == tensor.shape:
            compatible[key] = tensor
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(
            f"No se encontraron pesos compatibles para cargar desde {checkpoint_path}."
        )

    missing, unexpected = model.load_state_dict(compatible, strict=False)

    logger.info(
        "Inicialización desde checkpoint completada: "
        f"cargadas={len(compatible)}, omitidas_por_shape={len(skipped)}, "
        f"missing_post_load={len(missing)}, unexpected_post_load={len(unexpected)}"
    )

    if skipped:
        logger.info(
            "Se omitieron algunas claves por incompatibilidad (esperado si cambia head)."
        )


def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    """
    Convierte una columna de tiempo a float.

    - Si es datetime64, la convierte a segundos desde epoch.
    - Si no, la castea a float directamente.
    """
    if np.issubdtype(col.dtype, np.datetime64):
        # ns -> s
        return (col.view("int64") / 1e9).astype("float32")
    else:
        return col.astype("float32").to_numpy()


def _worker_init_fn(worker_id: int):
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Cargar configs
    # ------------------------------------------------------------------
    data_cfg: DataConfig = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)
    training_cfg, seed = load_training_config(args.training_config)

    # Preparar carpeta de experimento / checkpoints
    exp_dir = _resolve_experiment_dir(training_cfg.checkpoint_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(exp_dir, "train.log")
    setup_logging(log_file=log_file)
    logger = get_logger("train")

    logger.info("Cargando configuraciones...")
    logger.info(f"  data_config = {args.data_config}")
    logger.info(f"  model_config = {args.model_config}")
    logger.info(f"  training_config = {args.training_config}")
    logger.info(f"  stage = {args.stage}")
    logger.info(f"  init_checkpoint = {args.init_checkpoint}")
    logger.info(f"  experiment_name = {args.experiment_name}")
    logger.info(f"  experiment_dir = {exp_dir}")

    # Semilla global
    set_global_seed(seed, deterministic=False)
    logger.info(f"Semilla global fijada en {seed}")

    # ------------------------------------------------------------------
    # Cargar datos
    # ------------------------------------------------------------------
    logger.info(f"Leyendo datos desde {data_cfg.csv_path}...")
    df = pd.read_csv(data_cfg.csv_path)

    time_col = data_cfg.time_column
    feature_cols = data_cfg.feature_columns
    target_cols = data_cfg.target_columns

    if time_col not in df.columns:
        raise ValueError(f"La columna de tiempo '{time_col}' no existe en el CSV.")
    for col in feature_cols + target_cols:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el CSV.")

    # Ordenar por tiempo
    df = df.sort_values(time_col).reset_index(drop=True)

    # Split temporal train/val/test
    logger.info("Realizando split temporal train/val/test...")
    df_train, df_val, df_test = split_dataframe_by_time(
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

    # ------------------------------------------------------------------
    # Pasar DataFrames a arrays (timestamps, features, targets)
    # ------------------------------------------------------------------
    def df_to_arrays(df_part: pd.DataFrame):
        ts = _timestamps_to_float(df_part[time_col])
        X = df_part[feature_cols].to_numpy(dtype="float32")
        y = df_part[target_cols].to_numpy(dtype="float32")
        return ts, X, y

    ts_train, X_train, y_train = df_to_arrays(df_train)
    ts_val, X_val, y_val = df_to_arrays(df_val)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    logger.info(f"Dimensiones: input_dim={input_dim}, output_dim={output_dim}")

    # ------------------------------------------------------------------
    # Scalers (sólo en entrenamiento; se pueden guardar para inferencia)
    # ------------------------------------------------------------------
    value_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = value_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)

    X_val_scaled = value_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)

    # Concatenar features y targets en un solo array values
    values_train = np.concatenate([X_train_scaled, y_train_scaled], axis=1)  # [T_train, D_total]
    values_val = np.concatenate([X_val_scaled, y_val_scaled], axis=1)

    # ------------------------------------------------------------------
    # Construir datasets y dataloaders
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
        logger.info("Modo event-tokens activado: se usarán tokens (sensor_id, t, value).")
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

        ds_train = EventTimeSeriesDataset(
            values=X_train_scaled,
            timestamps=ts_train,
            targets=y_train_scaled,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_builder=seq_builder,
        )
        ds_val = EventTimeSeriesDataset(
            values=X_val_scaled,
            timestamps=ts_val,
            targets=y_val_scaled,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_builder=seq_builder,
        )
    else:
        seq_builder = SequenceBuilder(
            input_dim=input_dim,
            target_token_value="zeros",  # puedes cambiar a "last" si quieres
            use_sensor_ids=False,
            num_sensors=0,
            num_target_tokens=1,
            target_sensor_ids=None,
        )

        ds_train = TimeSeriesDataset(
            values=values_train,
            timestamps=ts_train,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            targets=None,  # Se toman de values a partir de input_dim
            sequence_builder=seq_builder,
        )

        ds_val = TimeSeriesDataset(
            values=values_val,
            timestamps=ts_val,
            config=ds_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            targets=None,
            sequence_builder=seq_builder,
        )

    collate_fn = build_collate_fn(
        pad_to_max_length=True,
    )

    from torch.utils.data import DataLoader

    _use_cuda = torch.cuda.is_available() and "cuda" in training_cfg.device
    _nw = data_cfg.num_workers

    g_train = torch.Generator().manual_seed(seed)
    g_val = torch.Generator().manual_seed(seed + 1)

    loader_kwargs = {
        "num_workers": _nw,
        "collate_fn": collate_fn,
        "pin_memory": _use_cuda,
        "persistent_workers": _nw > 0,
    }
    if _nw > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    if data_cfg.use_event_tokens:
        train_sampler = BucketBatchSampler(
            lengths=ds_train.get_approx_lengths(),
            batch_size=data_cfg.batch_size,
            shuffle=data_cfg.shuffle_train,
            drop_last=True,
            generator=g_train,
        )
        val_sampler = BucketBatchSampler(
            lengths=ds_val.get_approx_lengths(),
            batch_size=data_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            generator=g_val,
        )
        
        train_loader = DataLoader(
            ds_train,
            batch_sampler=train_sampler,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            ds_val,
            batch_sampler=val_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=data_cfg.batch_size,
            shuffle=data_cfg.shuffle_train,
            drop_last=True,
            generator=g_train if data_cfg.shuffle_train else None,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            ds_val,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    logger.info(
        f"DataLoaders listos: "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    # ------------------------------------------------------------------
    # Construir modelo
    # ------------------------------------------------------------------
    # Ajustar input_dim y output_dim según los datos
    model_cfg.input_dim = 1 if data_cfg.use_event_tokens else input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(data_cfg.use_event_tokens)
    model_cfg.num_sensors = input_dim if data_cfg.use_event_tokens else 0

    model = TimeSeriesTransformer(model_cfg)

    # Inicialización opcional desde checkpoint (útil para finetune)
    if args.init_checkpoint is not None:
        if not os.path.exists(args.init_checkpoint):
            raise FileNotFoundError(
                f"No se encontró init-checkpoint: {args.init_checkpoint}"
            )
        _load_compatible_weights(model, args.init_checkpoint, logger)

    logger.info(f"Modelo TimeSeriesTransformer:\n{model}")

    # Asegurar que el checkpoint_dir de TrainingConfig apunte al experimento
    training_cfg.checkpoint_dir = exp_dir

    # ------------------------------------------------------------------
    # Entrenar
    # ------------------------------------------------------------------
    logger.info("Comenzando entrenamiento...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_cfg,
    )

    history = trainer.fit()

    # En pretraining guardamos un alias explícito para facilitar etapa finetune
    if args.stage == "pretrain":
        best_ckpt_path = os.path.join(exp_dir, "best_model.pt")
        pretrain_ckpt_path = os.path.join(exp_dir, "pretrained_model.pt")
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
            torch.save(best_ckpt, pretrain_ckpt_path)
            logger.info(
                f"Checkpoint de pretraining guardado en {pretrain_ckpt_path}"
            )
        else:
            logger.warning(
                "No se encontró best_model.pt para crear pretrained_model.pt"
            )

    # Guardar historia de entrenamiento
    history_path = os.path.join(exp_dir, "history.yaml")
    with open(history_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f)
    logger.info(f"Historia de entrenamiento guardada en {history_path}")

    # Guardar config del modelo
    model_cfg_path = os.path.join(exp_dir, "model_config.yaml")
    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(model_cfg), f)
    logger.info(f"Configuración del modelo guardada en {model_cfg_path}")

    # Guardar scalers (para usar luego en inferencia)
    scalers_path = os.path.join(exp_dir, "scalers.pt")
    torch.save(
        {
            "value_scaler": value_scaler,
            "target_scaler": target_scaler,
            "feature_columns": feature_cols,
            "target_columns": target_cols,
            "time_column": time_col,
        },
        scalers_path,
    )
    logger.info(f"Scalers guardados en {scalers_path}")

    logger.info("Entrenamiento finalizado.")


if __name__ == "__main__":
    main()