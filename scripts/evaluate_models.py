import os
import sys
import argparse
import copy
from colorama import init, Fore, Style

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import numpy as np
import pandas as pd

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
from ts_transformer.training import TrainingConfig, Trainer
from ts_transformer.utils import load_data_config, load_model_config, load_training_config

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Evalua los modelos comparativos en el set de Test")
    parser.add_argument("--data-config", type=str, default="configs/data/toy_example_test1.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base_test1.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default_test1.yaml")
    parser.add_argument("--exp-dir", type=str, default="experiments/comparative_run")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def main():
    args = parse_args()
    init(autoreset=True)
    
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)
    training_cfg, _ = load_training_config(args.training_config)
    device = torch.device(training_cfg.device if torch.cuda.is_available() else "cpu")

    print(Fore.GREEN + f"=== Evaluando Benchmark Multi-Horizonte ===" + Style.RESET_ALL)
    
    df = pd.read_csv(data_cfg.csv_path)
    time_col = data_cfg.time_column
    df = df.sort_values(time_col).reset_index(drop=True)
    
    df_train, df_val, df_test = split_dataframe_by_time(
        df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio
    )
    
    print(Fore.CYAN + f"Set de test extraido: {len(df_test)} filas." + Style.RESET_ALL)

    def df_to_arrays(df_part):
        ts = _timestamps_to_float(df_part[time_col])
        X = df_part[data_cfg.feature_columns].to_numpy(dtype="float32")
        y = df_part[data_cfg.target_columns].to_numpy(dtype="float32")
        return ts, X, y

    ts_train, X_train, y_train = df_to_arrays(df_train)
    ts_test, X_test, y_test = df_to_arrays(df_test)
    
    value_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_train_scaled = value_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_test_scaled = value_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)
    
    values_test = np.concatenate([X_test_scaled, y_test_scaled], axis=1)
    output_dim = y_train.shape[1]
    input_dim = X_train.shape[1]
    
    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length,
        target_offset_choices=data_cfg.target_offset_choices,
        target_offset_min=data_cfg.target_offset_min,
        target_offset_max=data_cfg.target_offset_max,
        stride=data_cfg.stride,
        min_history_length=data_cfg.min_history_length,
        num_targets=data_cfg.num_targets,
    )

    use_events = data_cfg.use_event_tokens
    
    if use_events:
        target_sensor_ids = [data_cfg.feature_columns.index(c) if c in data_cfg.feature_columns else input_dim for c in data_cfg.target_columns]
        seq_builder = SequenceBuilder(input_dim=1, target_token_value="zeros", use_sensor_ids=True, num_sensors=input_dim, num_target_tokens=output_dim, target_sensor_ids=target_sensor_ids)
        ds_test = EventTimeSeriesDataset(X_test_scaled, ts_test, y_test_scaled, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        model_input_dim = 1
    else:
        seq_builder = SequenceBuilder(input_dim=input_dim, target_token_value="zeros", use_sensor_ids=False, num_sensors=0, num_target_tokens=1)
        ds_test = TimeSeriesDataset(values_test, ts_test, ds_cfg, input_dim, output_dim, sequence_builder=seq_builder)
        model_input_dim = input_dim

    collate_fn = build_collate_fn(pad_to_max_length=True)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(ds_test, batch_size=data_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0

    custom_model = TimeSeriesTransformer(model_cfg)
    d_model_sota = model_cfg.d_model
    strats_base = STraTSNetwork(num_features=input_dim + 1, d_model=d_model_sota, num_classes=output_dim)
    strats_wrapped = MultiHorizonBaselineWrapper(strats_base, "strats", d_model_sota, output_dim, use_sensor_embedding=use_events)
    
    coformer_base = CompatibleTransformer(num_variates=input_dim if use_events else 1, d_model=d_model_sota, num_classes=output_dim)
    coformer_wrapped = MultiHorizonBaselineWrapper(coformer_base, "coformer", d_model_sota, output_dim, use_sensor_embedding=use_events)
    
    models = {
        "Custom": custom_model,
        "STraTS_Adapter": strats_wrapped,
        "CoFormer_Adapter": coformer_wrapped
    }
    
    results = []
    
    for name, model in models.items():
        ckpt_path = os.path.join(args.exp_dir, name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(Fore.RED + f"Falta checkpoint para {name}" + Style.RESET_ALL)
            continue
            
        print(f"Cargando {name} y evaluando...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Utilizamos Trainer fake para aprovechar su logica de evaluacion
        trainer = Trainer(model=model, train_loader=test_loader, val_loader=test_loader, config=training_cfg)
        metrics = trainer._evaluate(epoch=0)
        metrics["Modelo"] = name
        results.append(metrics)
        print(Fore.YELLOW + f"Metrics {name}: " + ", ".join([f"{k}={v:.4f}" for k,v in metrics.items() if k != "Modelo"]) + Style.RESET_ALL)
        
    df_res = pd.DataFrame(results)
    df_res = df_res[["Modelo"] + [c for c in df_res.columns if c != "Modelo"]]
    out_csv = os.path.join(args.exp_dir, "test_metrics_summary.csv")
    df_res.to_csv(out_csv, index=False)
    print(Fore.GREEN + f"\nGuardado resumen tabular en {out_csv}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
