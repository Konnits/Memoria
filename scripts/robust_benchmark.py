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
from ts_transformer.utils import load_data_config, load_model_config, load_training_config, set_global_seed, setup_logging, get_logger

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Ejecución Robusta Benchmarking (Multi-Semilla)")
    parser.add_argument("--data-config", type=str, default="configs/data/toy_example_test1.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base_test1.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default_test1.yaml")
    parser.add_argument("--runs", type=int, default=3, help="Cantidad de simulaciones con semillas distintas.")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def build_models(model_cfg, use_events, model_input_dim, input_dim, output_dim):
    # Constructor de dict de modelos base por semilla
    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0

    custom = TimeSeriesTransformer(model_cfg)
    
    d_model = model_cfg.d_model
    s_base = STraTSNetwork(num_features=input_dim + 1, d_model=d_model, num_classes=output_dim)
    strats = MultiHorizonBaselineWrapper(s_base, "strats", d_model, output_dim, use_sensor_embedding=use_events)
    
    c_base = CompatibleTransformer(num_variates=input_dim if use_events else 1, d_model=d_model, num_classes=output_dim)
    coform = MultiHorizonBaselineWrapper(c_base, "coformer", d_model, output_dim, use_sensor_embedding=use_events)
    
    return {"Custom": custom, "STraTS_Adapter": strats, "CoFormer_Adapter": coform}

def main():
    args = parse_args()
    init(autoreset=True)
    setup_logging()
    logger = get_logger("robust_benchmark")

    logger.info(Fore.GREEN + f"=== Iniciando Pipeline Robusto ({args.runs} simulaciones) ===" + Style.RESET_ALL)
    
    data_cfg = load_data_config(args.data_config)
    model_cfg_base = load_model_config(args.model_config)
    training_cfg, default_seed = load_training_config(args.training_config)

    df = pd.read_csv(data_cfg.csv_path)
    time_col = data_cfg.time_column
    df = df.sort_values(time_col).reset_index(drop=True)
    df_train, df_val, df_test = split_dataframe_by_time(df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio)

    def process_split(df_part):
        ts = _timestamps_to_float(df_part[time_col])
        X = df_part[data_cfg.feature_columns].to_numpy(dtype="float32")
        y = df_part[data_cfg.target_columns].to_numpy(dtype="float32")
        return ts, X, y

    ts_train, X_train, y_train = process_split(df_train)
    ts_val, X_val, y_val = process_split(df_val)
    ts_test, X_test, y_test = process_split(df_test)
    
    v_scal = StandardScaler()
    t_scal = StandardScaler()
    X_train_s = v_scal.fit_transform(X_train)
    y_train_s = t_scal.fit_transform(y_train)
    X_val_s = v_scal.transform(X_val)
    y_val_s = t_scal.transform(y_val)
    X_test_s = v_scal.transform(X_test)
    y_test_s = t_scal.transform(y_test)

    v_tr = np.concatenate([X_train_s, y_train_s], axis=1)
    v_va = np.concatenate([X_val_s, y_val_s], axis=1)
    v_te = np.concatenate([X_test_s, y_test_s], axis=1)
    input_dim, output_dim = X_train.shape[1], y_train.shape[1]

    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length, target_offset_choices=data_cfg.target_offset_choices,
        target_offset_min=data_cfg.target_offset_min, target_offset_max=data_cfg.target_offset_max,
        stride=data_cfg.stride, min_history_length=data_cfg.min_history_length, 
        num_targets=data_cfg.num_targets
    )

    use_events = data_cfg.use_event_tokens

    if use_events:
        tsid = [data_cfg.feature_columns.index(c) if c in data_cfg.feature_columns else input_dim for c in data_cfg.target_columns]
        sqb = SequenceBuilder(input_dim=1, target_token_value="zeros", use_sensor_ids=True, num_sensors=input_dim, num_target_tokens=output_dim, target_sensor_ids=tsid)
        ds_tr = EventTimeSeriesDataset(X_train_s, ts_train, y_train_s, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        ds_va = EventTimeSeriesDataset(X_val_s, ts_val, y_val_s, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        ds_te = EventTimeSeriesDataset(X_test_s, ts_test, y_test_s, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        mi_dim = 1
    else:
        sqb = SequenceBuilder(input_dim=input_dim, target_token_value="zeros", use_sensor_ids=False, num_sensors=0, num_target_tokens=1)
        ds_tr = TimeSeriesDataset(v_tr, ts_train, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        ds_va = TimeSeriesDataset(v_va, ts_val, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        ds_te = TimeSeriesDataset(v_te, ts_test, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        mi_dim = input_dim

    collate_fn = build_collate_fn(pad_to_max_length=True)
    from torch.utils.data import DataLoader
    
    loader_kwargs = {"batch_size": data_cfg.batch_size, "collate_fn": collate_fn}
    train_loader = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ds_va, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ds_te, shuffle=False, **loader_kwargs)

    seeds = [default_seed + i*1337 for i in range(args.runs)]
    master_results = []
    exp_dir = "experiments/robust_benchmark"
    os.makedirs(exp_dir, exist_ok=True)

    device_used = torch.device(training_cfg.device if torch.cuda.is_available() else "cpu")

    for k, s in enumerate(seeds):
        logger.info(Fore.MAGENTA + f"\n[Ronda {k+1}/{args.runs}] Estableciendo Semilla: {s}" + Style.RESET_ALL)
        set_global_seed(s, deterministic=False)
        models = build_models(copy.deepcopy(model_cfg_base), use_events, mi_dim, input_dim, output_dim)

        for name, model in models.items():
            cfg_mod = copy.deepcopy(training_cfg)
            cfg_mod.checkpoint_dir = os.path.join(exp_dir, f"run_{k}", name)
            os.makedirs(cfg_mod.checkpoint_dir, exist_ok=True)
            
            logger.info(f"Entrenando {name} (Run {k+1})...")
            trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg_mod)
            history = trainer.fit()
            
            # Evaluación In-situ Test
            logger.info(f"Evaluando {name} (Run {k+1}) sobre TEST SET...")
            ckpt_path = os.path.join(cfg_mod.checkpoint_dir, "best_model.pt")
            checkpoint = torch.load(ckpt_path, map_location=device_used, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            test_metrics = trainer._evaluate(epoch=0)
            
            # Guardamos
            record = {"Modelo": name, "Run": k+1, "Seed": s}
            record.update({m: v for m,v in test_metrics.items()})
            master_results.append(record)

    df_r = pd.DataFrame(master_results)
    df_r.to_csv(os.path.join(exp_dir, "raw_metrics_runs.csv"), index=False)
    
    # Agrupamos por modelo y calculamos Mean \pm Std
    df_agg = df_r.groupby("Modelo").agg({c: ['mean', 'std'] for c in df_r.columns if c.startswith("val_")})
    df_nice = pd.DataFrame()
    for c in df_r.columns:
        if c.startswith("val_"):
            df_nice[c] = df_agg[c]['mean'].round(4).astype(str) + " ± " + df_agg[c]['std'].round(4).astype(str)
            
    df_nice.to_csv(os.path.join(exp_dir, "thesis_table_robust.csv"))
    logger.info(Fore.GREEN + "\n==== RESULTADOS ROBUSTOS MULTI-SEED ====" + Style.RESET_ALL)
    print(df_nice.to_markdown())

if __name__ == "__main__":
    main()
