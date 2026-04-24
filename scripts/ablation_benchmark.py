import os
import sys
import argparse
import copy
from colorama import init, Fore, Style
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from ts_transformer.data import (
    TimeSeriesDataset, EventTimeSeriesDataset, SequenceBuilder,
    build_collate_fn, StandardScaler, split_dataframe_by_time
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.training import TrainingConfig, Trainer
from ts_transformer.utils import load_data_config, load_model_config, load_training_config

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Ablación del Multi-Horizonte M")
    parser.add_argument("--data-config", type=str, default="configs/data/toy_example_test1.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base_test1.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default_test1.yaml")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def main():
    args = parse_args()
    init(autoreset=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = load_data_config(args.data_config)
    model_cfg_base = load_model_config(args.model_config)
    training_cfg, _ = load_training_config(args.training_config)

    df = pd.read_csv(data_cfg.csv_path)
    time_col = data_cfg.time_column
    df = df.sort_values(time_col).reset_index(drop=True)
    df_train, df_val, df_test = split_dataframe_by_time(df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio)

    def process_split(df_part):
        return _timestamps_to_float(df_part[time_col]), df_part[data_cfg.feature_columns].to_numpy(dtype="float32"), df_part[data_cfg.target_columns].to_numpy(dtype="float32")

    ts_train, X_train, y_train = process_split(df_train)
    ts_val, X_val, y_val = process_split(df_val)
    
    v_scal, t_scal = StandardScaler(), StandardScaler()
    X_train_s = v_scal.fit_transform(X_train)
    y_train_s = t_scal.fit_transform(y_train)
    X_val_s = v_scal.transform(X_val)
    y_val_s = t_scal.transform(y_val)
    
    v_tr = np.concatenate([X_train_s, y_train_s], axis=1)
    v_va = np.concatenate([X_val_s, y_val_s], axis=1)
    input_dim, output_dim = X_train.shape[1], y_train.shape[1]
    
    # Ablation Targets M
    M_values = [1, 5, 20]
    
    ablation_results = []
    exp_dir = "experiments/ablation"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(Fore.GREEN + "=== Iniciando Estudio Ablativo de M-Targets ===" + Style.RESET_ALL)
    
    for M in M_values:
        print(Fore.MAGENTA + f"\n[Ronda M={M}] Configurando Dataloader Multi-Horizonte..." + Style.RESET_ALL)
        
        ds_cfg = TimeSeriesDatasetConfig(
            history_length=data_cfg.history_length, target_offset_choices=data_cfg.target_offset_choices,
            target_offset_min=data_cfg.target_offset_min, target_offset_max=data_cfg.target_offset_max,
            stride=data_cfg.stride, min_history_length=data_cfg.min_history_length, 
            num_targets=M
        )
        
        # Datasets
        if data_cfg.use_event_tokens:
            tsid = [data_cfg.feature_columns.index(c) if c in data_cfg.feature_columns else input_dim for c in data_cfg.target_columns]
            sqb = SequenceBuilder(1, target_token_value="zeros", use_sensor_ids=True, num_sensors=input_dim, num_target_tokens=output_dim, target_sensor_ids=tsid)
            ds_tr = EventTimeSeriesDataset(X_train_s, ts_train, y_train_s, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
            ds_va = EventTimeSeriesDataset(X_val_s, ts_val, y_val_s, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
            mi_dim = 1
        else:
            sqb = SequenceBuilder(input_dim, target_token_value="zeros", use_sensor_ids=False, num_sensors=0, num_target_tokens=1)
            ds_tr = TimeSeriesDataset(v_tr, ts_train, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
            ds_va = TimeSeriesDataset(v_va, ts_val, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
            mi_dim = input_dim

        collate_fn = build_collate_fn(pad_to_max_length=True)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(ds_tr, batch_size=data_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(ds_va, batch_size=data_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        
        model_cfg_base.input_dim = mi_dim
        model_cfg_base.output_dim = output_dim
        model_cfg_base.use_sensor_embedding = bool(data_cfg.use_event_tokens)
        model_cfg_base.num_sensors = input_dim if data_cfg.use_event_tokens else 0
        
        d_model = model_cfg_base.d_model
        models = {
            "Custom": TimeSeriesTransformer(copy.deepcopy(model_cfg_base)),
            "STraTS_Adapter": MultiHorizonBaselineWrapper(STraTSNetwork(num_features=input_dim + 1, d_model=d_model, num_classes=output_dim), "strats", d_model, output_dim, use_sensor_embedding=data_cfg.use_event_tokens),
            "CoFormer_Adapter": MultiHorizonBaselineWrapper(
                CompatibleTransformer(
                    num_variates=input_dim if data_cfg.use_event_tokens else 1,
                    d_model=d_model,
                    n_heads=model_cfg_base.num_heads,
                    n_layers=model_cfg_base.num_layers,
                    dropout=model_cfg_base.dropout,
                    num_classes=output_dim,
                ),
                "coformer",
                d_model,
                output_dim,
                use_sensor_embedding=data_cfg.use_event_tokens,
            )
        }
        
        for name, model in models.items():
            print(f"Entrenando {name} con M={M}...")
            cfg_mod = copy.deepcopy(training_cfg)
            cfg_mod.checkpoint_dir = os.path.join(exp_dir, f"M_{M}", name)
            os.makedirs(cfg_mod.checkpoint_dir, exist_ok=True)
            cfg_mod.num_epochs = 5 # Early limit to scale speed for ablation
            
            trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg_mod)
            history = trainer.fit()
            
            best_rmse = min(history.get("val_rmse", [])) if "val_rmse" in history else np.nan
            ablation_results.append({"M_Targets": M, "Modelo": name, "Root Mean Squared Error (Val)": round(best_rmse,4)})
            
    # Guardar Ablacion
    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(os.path.join(exp_dir, "ablation_results.csv"), index=False)
    
    # Hacer pivot format
    table = df_abl.pivot(index='Modelo', columns='M_Targets', values='Root Mean Squared Error (Val)')
    table.to_csv(os.path.join(exp_dir, "ablation_grouped.csv"))
    print(Fore.GREEN + "\n==== RESULTADOS ABLACION ====" + Style.RESET_ALL)
    print(table.to_markdown())

if __name__ == "__main__":
    main()
