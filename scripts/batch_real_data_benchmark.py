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
    parser = argparse.ArgumentParser(description="Benchmarking Lote en Datasets Reales")
    parser.add_argument("--base-data-config", type=str, default="configs/data/real_data1.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base_test1.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default_test1.yaml")
    parser.add_argument("--start-dataset", type=int, default=1)
    parser.add_argument("--end-dataset", type=int, default=17)
    parser.add_argument("--runs-per-dataset", type=int, default=1, help="Semillas aleatorias por dataset.")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def build_models(model_cfg, use_events, model_input_dim, input_dim, output_dim):
    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0

    custom = TimeSeriesTransformer(model_cfg)
    d_model = model_cfg.d_model
    
    s_base = STraTSNetwork(num_features=input_dim + 1, d_model=d_model, num_classes=output_dim)
    strats = MultiHorizonBaselineWrapper(s_base, "strats", d_model, output_dim, use_sensor_embedding=use_events)
    
    # c_base = CompatibleTransformer(num_variates=input_dim if use_events else 1, d_model=d_model, num_classes=output_dim)
    # coform = MultiHorizonBaselineWrapper(c_base, "coformer", d_model, output_dim, use_sensor_embedding=use_events)
    
    return {"Custom": custom, "STraTS_Adapter": strats}

def main():
    args = parse_args()
    init(autoreset=True)
    setup_logging()
    logger = get_logger("batch_benchmark")

    logger.info(Fore.GREEN + f"=== Iniciando Lote de Benchmark (Datasets {args.start_dataset} a {args.end_dataset}) ===" + Style.RESET_ALL)
    
    base_data_cfg = load_data_config(args.base_data_config)
    model_cfg_base = load_model_config(args.model_config)
    training_cfg, default_seed = load_training_config(args.training_config)

    device_used = torch.device(training_cfg.device if torch.cuda.is_available() else "cpu")
    
    master_results = []
    exp_dir = "experiments/real_data_batch"
    os.makedirs(exp_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, "benchmark_sensors.csv")
    
    # Intenta cargar un archivo existente si lo detienes y retomas
    completed_runs = set()
    if os.path.exists(out_csv):
        df_exist = pd.read_csv(out_csv)
        master_results = df_exist.to_dict('records')
        for val in master_results:
            completed_runs.add((val["Dataset_ID"], val["Seed"], val["Modelo"]))
        logger.info(Fore.YELLOW + f"[*] Se detectaron {len(completed_runs)} simulaciones previas en el CSV. Se reanudará desde el último punto de error." + Style.RESET_ALL)

    for ds_idx in range(args.start_dataset, args.end_dataset + 1):
        target_csv = f"data/processed/real_data_{ds_idx}.csv"
        
        if not os.path.exists(target_csv):
            logger.warning(Fore.YELLOW + f"Saltando Dataset {ds_idx} porque no existe: {target_csv}" + Style.RESET_ALL)
            continue
            
        logger.info(Fore.CYAN + f"\n======= Procesando Dataset {ds_idx}: {target_csv} =======" + Style.RESET_ALL)
        
        df = pd.read_csv(target_csv)

        # Revisamos que la columan timestamp es numerica en vez de string
        if not pd.api.types.is_numeric_dtype(df[base_data_cfg.time_column]):
            df[base_data_cfg.time_column] = pd.to_datetime(df[base_data_cfg.time_column]).apply(lambda x: x.timestamp())

        time_col = base_data_cfg.time_column
        df = df.sort_values(time_col).reset_index(drop=True)
        df_train, df_val, df_test = split_dataframe_by_time(df, time_column=time_col, train_ratio=base_data_cfg.train_ratio, val_ratio=base_data_cfg.val_ratio)

        def process_split(df_part):
            ts = _timestamps_to_float(df_part[time_col])
            X = df_part[base_data_cfg.feature_columns].to_numpy(dtype="float32")
            y = df_part[base_data_cfg.target_columns].to_numpy(dtype="float32")
            return ts, X, y

        ts_train, X_train, y_train = process_split(df_train)
        ts_val, X_val, y_val = process_split(df_val)
        ts_test, X_test, y_test = process_split(df_test)
        
        v_scal, t_scal = StandardScaler(), StandardScaler()
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
            history_length=base_data_cfg.history_length, target_offset_choices=base_data_cfg.target_offset_choices,
            target_offset_min=base_data_cfg.target_offset_min, target_offset_max=base_data_cfg.target_offset_max,
            stride=base_data_cfg.stride, min_history_length=base_data_cfg.min_history_length, 
            num_targets=base_data_cfg.num_targets
        )

        use_events = base_data_cfg.use_event_tokens

        if use_events:
            tsid = [base_data_cfg.feature_columns.index(c) if c in base_data_cfg.feature_columns else input_dim for c in base_data_cfg.target_columns]
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
        
        num_workers = 4
        loader_kwargs = {
            "batch_size": base_data_cfg.batch_size,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        train_loader = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(ds_va, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(ds_te, shuffle=False, **loader_kwargs)

        seeds = [default_seed + i*42 for i in range(args.runs_per_dataset)]

        for k, s in enumerate(seeds):
            logger.info(Fore.MAGENTA + f"\n  [Dataset {ds_idx} | Semilla {s}]" + Style.RESET_ALL)
            set_global_seed(s, deterministic=False)
            models = build_models(copy.deepcopy(model_cfg_base), use_events, mi_dim, input_dim, output_dim)

            for name, model in models.items():
                if (ds_idx, s, name) in completed_runs:
                    logger.info(f"  Saltando {name} (Ya fue completado previamente en el archivo maestro).")
                    continue
                    
                cfg_mod = copy.deepcopy(training_cfg)
                cfg_mod.checkpoint_dir = os.path.join(exp_dir, f"ds_{ds_idx}_seed_{s}", name)
                os.makedirs(cfg_mod.checkpoint_dir, exist_ok=True)
                
                logger.info(f"  Entrenando {name}...")
                trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg_mod)
                trainer.fit()
                
                logger.info(f"  Evaluando {name} en Muestra Retenida (Test)...")
                ckpt_path = os.path.join(cfg_mod.checkpoint_dir, "best_model.pt")
                checkpoint = torch.load(ckpt_path, map_location=device_used, weights_only=False)
                
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                test_metrics = trainer._evaluate(epoch=0)
                
                record = {"Dataset_ID": ds_idx, "Modelo": name, "Seed": s}
                record.update({m: v for m,v in test_metrics.items()})
                master_results.append(record)
                
                # Guardar CSV después de cada modelo para no perder progreso
                pd.DataFrame(master_results).to_csv(out_csv, index=False)
                logger.info(Fore.GREEN + f"[*] Progreso exportado a {out_csv} ({name} ds={ds_idx})" + Style.RESET_ALL)

    logger.info(Fore.GREEN + "\n==== BENCHMARK COMPLETADO ====" + Style.RESET_ALL)
    print(pd.DataFrame(master_results).to_markdown())

if __name__ == "__main__":
    main()
