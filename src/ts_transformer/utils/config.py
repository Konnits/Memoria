from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import yaml

from ..models.time_series_transformer import TimeSeriesTransformerConfig
from ..training.train_loop import TrainingConfig
from ..training.optimizers import OptimizerConfig


@dataclass
class DataConfig:
    """
    Configuración para la carga de datos y DataLoader.

    csv_path:
        Ruta al archivo CSV (o similar) con los datos.
    time_column:
        Nombre de la columna que contiene el timestamp.
    feature_columns:
        Lista de columnas usadas como features de entrada.
    target_columns:
        Lista de columnas usadas como targets (salida).
    train_ratio:
        Proporción de datos para entrenamiento.
    val_ratio:
        Proporción de datos para validación (el resto va a test).
    history_length:
        Longitud de la historia (número de puntos pasados por ejemplo).
    target_offset:
        Cuántos pasos después del "anchor" se predice.
    stride:
        Saltos entre ejemplos consecutivos en el dataset.
    batch_size:
        Tamaño de batch.
    num_workers:
        Número de workers para el DataLoader.
    shuffle_train:
        Si True, mezcla el dataset de entrenamiento en cada época.
    min_history_length:
        Longitud MÍNIMA de la historia (si es None, se usa historia fija = history_length).
    target_offset_choices:
        Lista de offsets posibles (en pasos) para samplear un horizonte variable.
    target_offset_min / target_offset_max:
        Alternativa compacta para samplear offsets enteros en [min, max].
        Si target_offset_choices está definido, tiene prioridad.
    num_targets:
        Cantidad de timestamps futuros a predecir por muestra.
    use_event_tokens:
        Si True, construye tokens por observación (sensor_id, t, value)
        en lugar de secuencias densas por paso temporal.
    """

    csv_path: str
    time_column: str
    feature_columns: List[str]
    target_columns: List[str]
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    history_length: int = 24
    target_offset: int = 1
    stride: int = 1
    batch_size: int = 32
    num_workers: int = 0
    shuffle_train: bool = True
    min_history_length: int | None = None
    target_offset_choices: List[int] | None = None
    target_offset_min: int | None = None
    target_offset_max: int | None = None
    num_targets: int = 1
    use_event_tokens: bool = False


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data_config(path: str) -> DataConfig:
    """
    Carga un DataConfig desde un archivo YAML.
    """
    cfg = _load_yaml(path)
    return DataConfig(
        csv_path=cfg["csv_path"],
        time_column=cfg["time_column"],
        feature_columns=list(cfg["feature_columns"]),
        target_columns=list(cfg["target_columns"]),
        train_ratio=float(cfg.get("train_ratio", 0.7)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        history_length=int(cfg.get("history_length", 24)),
        target_offset=int(cfg.get("target_offset", 1)),
        stride=int(cfg.get("stride", 1)),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 0)),
        shuffle_train=bool(cfg.get("shuffle_train", True)),
        min_history_length=(
            int(cfg["min_history_length"])
            if "min_history_length" in cfg
            else None
        ),
        target_offset_choices=(
            list(cfg["target_offset_choices"])
            if "target_offset_choices" in cfg
            else None
        ),
        target_offset_min=(
            int(cfg["target_offset_min"])
            if "target_offset_min" in cfg
            else None
        ),
        target_offset_max=(
            int(cfg["target_offset_max"])
            if "target_offset_max" in cfg
            else None
        ),
        num_targets=int(cfg.get("num_targets", 1)),
        use_event_tokens=bool(cfg.get("use_event_tokens", False)),
    )


def load_model_config(path: str) -> TimeSeriesTransformerConfig:
    """
    Carga un TimeSeriesTransformerConfig desde YAML.

    Nota: input_dim y output_dim se ajustarán luego según el DataConfig.
    """
    cfg = _load_yaml(path)

    return TimeSeriesTransformerConfig(
        input_dim=int(cfg.get("input_dim", 1)),
        output_dim=int(cfg.get("output_dim", 1)),
        d_model=int(cfg.get("d_model", 128)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        activation=str(cfg.get("activation", "relu")),
        time_scale=float(cfg.get("time_scale", 900.0)),
        time_transform=str(cfg.get("time_transform", "log1p")),
        use_causal_mask=bool(cfg.get("use_causal_mask", False)),
        use_sensor_embedding=bool(cfg.get("use_sensor_embedding", False)),
        num_sensors=int(cfg.get("num_sensors", 0)),
        time_encoding_mode=str(cfg.get("time_encoding_mode", "sinusoidal")),
        readout_mode=str(cfg.get("readout_mode", "target_token")),
        use_temporal_attn_bias=bool(cfg.get("use_temporal_attn_bias", False)),
        use_target_flag_embedding=bool(cfg.get("use_target_flag_embedding", True)),
        validate_inputs=bool(cfg.get("validate_inputs", True)),
    )


def load_training_config(path: str) -> Tuple[TrainingConfig, int]:
    """
    Carga un TrainingConfig y la semilla global desde YAML.

    El YAML de training se asume con forma aproximada:

    num_epochs: 20
    device: "cuda"
    loss_name: "mse"
    grad_clip_norm: 1.0
    log_every_n_steps: 50
    checkpoint_dir: "experiments/exp_default"
    save_best_on: "val_rmse"
    seed: 42
    optimizer:
      optimizer_name: "adam"
      lr: 0.001
      weight_decay: 0.0
      momentum: 0.9
      betas: [0.9, 0.999]
      scheduler_name: "cosine"
      scheduler_step_size: 10
      scheduler_gamma: 0.1
      scheduler_T_max: 20
    """
    cfg = _load_yaml(path)

    opt_cfg_raw = cfg.get("optimizer", {})
    opt_cfg = OptimizerConfig(
        optimizer_name=opt_cfg_raw.get("optimizer_name", "adam"),
        lr=float(opt_cfg_raw.get("lr", 1e-3)),
        weight_decay=float(opt_cfg_raw.get("weight_decay", 0.0)),
        momentum=float(opt_cfg_raw.get("momentum", 0.9)),
        betas=tuple(opt_cfg_raw.get("betas", [0.9, 0.999])),  # type: ignore[arg-type]
        scheduler_name=opt_cfg_raw.get("scheduler_name", None),
        scheduler_step_size=int(opt_cfg_raw.get("scheduler_step_size", 10)),
        scheduler_gamma=float(opt_cfg_raw.get("scheduler_gamma", 0.1)),
        scheduler_T_max=int(opt_cfg_raw.get("scheduler_T_max", 50)),
        warmup_epochs=int(opt_cfg_raw.get("warmup_epochs", 0)),
    )

    train_cfg = TrainingConfig(
        num_epochs=int(cfg.get("num_epochs", 20)),
        device=str(cfg.get("device", "cpu")),
        loss_name=str(cfg.get("loss_name", "mse")),
        optimizer_config=opt_cfg,
        grad_clip_norm=float(cfg.get("grad_clip_norm", 0.0)),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        checkpoint_dir=cfg.get("checkpoint_dir", None),
        save_best_on=str(cfg.get("save_best_on", "val_loss")),
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)),
        unfreeze_lr=(
            float(cfg["unfreeze_lr"]) if "unfreeze_lr" in cfg else None
        ),
        input_noise_std=float(cfg.get("input_noise_std", 0.0)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0)),
        restore_best_weights=bool(cfg.get("restore_best_weights", True)),
        use_amp=bool(cfg.get("use_amp", False)),
        enable_cuda_runtime_optimizations=bool(cfg.get("enable_cuda_runtime_optimizations", True)),
        use_torch_compile=bool(cfg.get("use_torch_compile", False)),
        torch_compile_mode=str(cfg.get("torch_compile_mode", "reduce-overhead")),
        torch_compile_fullgraph=bool(cfg.get("torch_compile_fullgraph", False)),
    )

    seed = int(cfg.get("seed", 42))
    return train_cfg, seed