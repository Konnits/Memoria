import os
import sys
import argparse
import copy
import yaml
from colorama import init, Fore, Style

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ts_transformer.data import (
    TimeSeriesDataset,
    EventTimeSeriesDataset,
    SequenceBuilder,
    build_collate_fn,
    StandardScaler,
    split_dataframe_by_time,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.models.time_series_transformer import TimeSeriesTransformerConfig
from ts_transformer.training import TrainingConfig, Trainer
from ts_transformer.utils import (
    load_data_config,
    load_training_config,
    load_model_config,
    set_global_seed,
    setup_logging,
    get_logger,
)

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Comparativa de Modelos Multi-Objetivo")
    parser.add_argument("--data-config", type=str, default="configs/data/toy_example.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default.yaml")
    # Forzamos num_targets > 1 si no viene indicado en el config para probar la capacidad de fluctuación
    parser.add_argument("--num-targets", type=int, default=3, help="Tiempos futuros a predecir por ventana")
    parser.add_argument("--epochs", type=int, default=0, help="Forzar número de epocas (0=usa yaml)")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def main():
    args = parse_args()
    init(autoreset=True)
    
    # 1. Cargar Configuración
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)
    training_cfg, seed = load_training_config(args.training_config)
    
    if args.epochs > 0:
        training_cfg.num_epochs = args.epochs

    set_global_seed(seed, deterministic=False)
    setup_logging()
    logger = get_logger("comparativa")

    logger.info(Fore.GREEN + "=== Iniciando Pipeline Comparativo Multi-Objetivo ===" + Style.RESET_ALL)
    logger.info(f"Dispositivo detectado para entrenamiento: {training_cfg.device}")
    if training_cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning(Fore.RED + "¡ADVERTENCIA! Se configuró 'cuda' pero no está disponible. PyTorch usará CPU o fallará si el código no está adaptado." + Style.RESET_ALL)
    df = pd.read_csv(data_cfg.csv_path)
    time_col = data_cfg.time_column
    df = df.sort_values(time_col).reset_index(drop=True)
    
    df_train, df_val, df_test = split_dataframe_by_time(
        df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio
    )
    
    def df_to_arrays(df_part):
        ts = _timestamps_to_float(df_part[time_col])
        X = df_part[data_cfg.feature_columns].to_numpy(dtype="float32")
        y = df_part[data_cfg.target_columns].to_numpy(dtype="float32")
        return ts, X, y

    ts_train, X_train, y_train = df_to_arrays(df_train)
    ts_val, X_val, y_val = df_to_arrays(df_val)
    
    value_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_train_scaled = value_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_val_scaled = value_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)
    
    values_train = np.concatenate([X_train_scaled, y_train_scaled], axis=1)
    values_val = np.concatenate([X_val_scaled, y_val_scaled], axis=1)
    output_dim = y_train.shape[1]
    input_dim = X_train.shape[1]
    
    # 3. Preparar Datasets con Múltiples Targets
    # Forzamos num_targets
    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length,
        target_offset_choices=data_cfg.target_offset_choices,
        target_offset_min=data_cfg.target_offset_min,
        target_offset_max=data_cfg.target_offset_max,
        stride=data_cfg.stride,
        min_history_length=data_cfg.min_history_length,
        num_targets=args.num_targets
    )

    use_events = data_cfg.use_event_tokens
    
    if use_events:
        target_sensor_ids = [data_cfg.feature_columns.index(c) if c in data_cfg.feature_columns else input_dim for c in data_cfg.target_columns]
        seq_builder = SequenceBuilder(input_dim=1, target_token_value="zeros", use_sensor_ids=True, num_sensors=input_dim, num_target_tokens=output_dim, target_sensor_ids=target_sensor_ids)
        ds_train = EventTimeSeriesDataset(X_train_scaled, ts_train, y_train_scaled, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        ds_val = EventTimeSeriesDataset(X_val_scaled, ts_val, y_val_scaled, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        model_input_dim = 1
    else:
        seq_builder = SequenceBuilder(input_dim=input_dim, target_token_value="zeros", use_sensor_ids=False, num_sensors=0, num_target_tokens=1)
        ds_train = TimeSeriesDataset(values_train, ts_train, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        ds_val = TimeSeriesDataset(values_val, ts_val, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        model_input_dim = input_dim

    collate_fn = build_collate_fn(pad_to_max_length=True)
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(ds_train, batch_size=data_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=data_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Construir Modelos
    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0

    custom_model = TimeSeriesTransformer(model_cfg)
    
    # STraTS
    d_model_sota = model_cfg.d_model
    strats_base = STraTSNetwork(num_features=input_dim + 1, d_model=d_model_sota, num_classes=output_dim)
    strats_wrapped = MultiHorizonBaselineWrapper(strats_base, "strats", d_model_sota, output_dim, use_sensor_embedding=use_events)
    
    # CoFormer
    coformer_base = CompatibleTransformer(num_variates=input_dim if use_events else 1, d_model=d_model_sota, num_classes=output_dim)
    coformer_wrapped = MultiHorizonBaselineWrapper(coformer_base, "coformer", d_model_sota, output_dim, use_sensor_embedding=use_events)
    
    models = {
        "Custom": custom_model,
        "STraTS_Adapter": strats_wrapped,
        "CoFormer_Adapter": coformer_wrapped
    }
    
    histories = {}
    
    # 5. Loop Comparativo
    exp_dir = "experiments/comparative_run"
    os.makedirs(exp_dir, exist_ok=True)
    
    for name, model in models.items():
        logger.info(Fore.CYAN + f"\nEntrenando arquitectura: {name}..." + Style.RESET_ALL)
        
        cfg_model = copy.deepcopy(training_cfg)
        cfg_model.checkpoint_dir = os.path.join(exp_dir, name)
        os.makedirs(cfg_model.checkpoint_dir, exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg_model,
        )
        
        history = trainer.fit()
        histories[name] = history
        logger.info(Fore.YELLOW + f"Finalizado {name}. Mejor Val Loss: {min(history['val_loss']):.4f}" + Style.RESET_ALL)
        
    logger.info(Fore.GREEN + "\nComparativa Finalizada con Éxito." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
