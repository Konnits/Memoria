import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ts_transformer.data import TimeSeriesDataset, EventTimeSeriesDataset, SequenceBuilder, build_collate_fn, StandardScaler, split_dataframe_by_time
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesTransformer
from ts_transformer.utils import load_data_config, load_model_config, load_training_config

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser("Graficador Evaluativo para Series de Tiempos Irregulares")
    parser.add_argument("--data-config", type=str, default="configs/data/toy_example_test1.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/transformer_base_test1.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default_test1.yaml")
    parser.add_argument("--exp-dir", type=str, default="experiments/comparative_run")
    parser.add_argument("--num-plots", type=int, default=3, help="Cuántos plot generar.")
    return parser.parse_args()

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)
    training_cfg, _ = load_training_config(args.training_config)

    df = pd.read_csv(data_cfg.csv_path)
    time_col = data_cfg.time_column
    df = df.sort_values(time_col).reset_index(drop=True)
    _, _, df_test = split_dataframe_by_time(df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio)

    def process_split(df_part):
        return _timestamps_to_float(df_part[time_col]), df_part[data_cfg.feature_columns].to_numpy(dtype="float32"), df_part[data_cfg.target_columns].to_numpy(dtype="float32")
    
    # We must fit scalers on Train to invert correctly later, but we didn't export scalers.
    # We'll just do a quick fit on full train logic
    df_train, _, _ = split_dataframe_by_time(df, time_column=time_col, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio)
    _, X_train, y_train = process_split(df_train)
    target_scaler = StandardScaler()
    target_scaler.fit_transform(y_train)

    ts_test, X_test, y_test = process_split(df_test)
    X_test_scaled = StandardScaler().fit(X_train).transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    use_events = data_cfg.use_event_tokens

    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length, 
        target_offset_min=data_cfg.target_offset_min, 
        target_offset_max=data_cfg.target_offset_max,
        stride=data_cfg.stride, 
        min_history_length=data_cfg.min_history_length, 
        num_targets=data_cfg.num_targets
    )

    if use_events:
        tsid = [data_cfg.feature_columns.index(c) if c in data_cfg.feature_columns else input_dim for c in data_cfg.target_columns]
        sqb = SequenceBuilder(1, target_token_value="zeros", use_sensor_ids=True, num_sensors=input_dim, num_target_tokens=output_dim, target_sensor_ids=tsid)
        ds_test = EventTimeSeriesDataset(X_test_scaled, ts_test, y_test_scaled, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        model_input_dim = 1
    else:
        values_test = np.concatenate([X_test_scaled, y_test_scaled], axis=1)
        sqb = SequenceBuilder(input_dim, target_token_value="zeros", use_sensor_ids=False, num_sensors=0, num_target_tokens=1)
        ds_test = TimeSeriesDataset(values_test, ts_test, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
        model_input_dim = input_dim

    collate_fn = build_collate_fn(pad_to_max_length=True)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(ds_test, batch_size=args.num_plots, shuffle=True, collate_fn=collate_fn)

    # Models Initialization
    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0

    custom = TimeSeriesTransformer(model_cfg).to(device)
    custom.eval()
    
    d_model = model_cfg.d_model
    strats = MultiHorizonBaselineWrapper(STraTSNetwork(num_features=input_dim + 1, d_model=d_model, num_classes=output_dim), "strats", d_model, output_dim, use_sensor_embedding=use_events).to(device)
    strats.eval()

    coform = MultiHorizonBaselineWrapper(
        CompatibleTransformer(
            num_variates=input_dim if use_events else 1,
            d_model=d_model,
            n_heads=model_cfg.num_heads,
            n_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            num_classes=output_dim,
        ),
        "coformer",
        d_model,
        output_dim,
        use_sensor_embedding=use_events,
    ).to(device)
    coform.eval()

    models = {"Custom": custom, "CoFormer_Adapter": coform, "STraTS_Adapter": strats}
    
    # Load Weights
    for name, m in models.items():
        ckpt_path = os.path.join(args.exp_dir, name, "best_model.pt")
        if os.path.exists(ckpt_path):
            chkp = torch.load(ckpt_path, map_location=device, weights_only=False)
            m.load_state_dict(chkp["model_state_dict"] if "model_state_dict" in chkp else chkp)
        else:
            print(f"No checkpoint para {name}. Se omitirá visualización fiable.")

    # Inferencia de 1 batch
    batch = next(iter(test_loader))
    with torch.no_grad():
        x_val = batch["input_values"].to(device)
        x_ts  = batch["input_timestamps"].to(device)
        is_tgt = batch["is_target_mask"].to(device)
        s_id  = batch.get("input_sensor_ids", None)
        if s_id is not None: s_id = s_id.to(device)
        pad = batch.get("padding_mask", None)
        if pad is not None: pad = pad.to(device)

        preds = {}
        for name, m in models.items():
            out = m(x_val, x_ts, is_tgt, s_id, pad, return_dict=False)
            preds[name] = out.cpu().numpy()
            
    # Graficar
    os.makedirs("experiments/plots", exist_ok=True)
    target_values = batch["target_values"].numpy()
    target_times = batch["target_timestamp"].numpy()
    
    import matplotlib.style as style
    style.use("seaborn-v0_8-paper")

    for i in range(args.num_plots):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Recuperar la historia desde el tensor crudo
        mask_h = ~batch["is_target_mask"][i].numpy() & ~batch.get("padding_mask", torch.zeros_like(is_tgt))[i].numpy()
        hist_times = batch["input_timestamps"][i].numpy()[mask_h]
        if use_events:
            # Recomposicion de events
            hist_vals = batch["input_values"][i, mask_h, 0].numpy()
        else:
            hist_vals = batch["input_values"][i, mask_h, 0].numpy()
            
        ax.plot(hist_times, hist_vals, marker=".", linestyle="-", color="grey", alpha=0.5, label="Historia Normalizada")
        
        # Invertir escalas a escala normal (o mostrar normalizada)
        t_t = target_times[i]
        t_v = target_values[i, :, 0] if target_values.ndim == 3 else target_values[i, :]
        
        # Los M targets
        ax.scatter(t_t, t_v, color="black", s=100, marker="x", label="Verdad Terrestre (Truth)", zorder=5)

        colors = {"Custom": "red", "STraTS_Adapter": "green", "CoFormer_Adapter": "blue"}
        markers = {"Custom": "o", "STraTS_Adapter": "^", "CoFormer_Adapter": "s"}
        
        for name, prd_mat in preds.items():
            p_v = prd_mat[i, :, 0] if prd_mat.ndim == 3 else prd_mat[i, :]
            ax.plot(t_t, p_v, marker=markers[name], color=colors[name], linestyle="--", linewidth=2, alpha=0.9, label=f"Pred: {name}")

        ax.set_title(f"Muestra Multi-Temporal Test Set {i+1}", fontsize=14)
        ax.set_xlabel("Timestamp Relativo")
        ax.set_ylabel("Valor Normalizado")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"experiments/plots/pred_sample_{i+1}.png", dpi=300)
        plt.close()
    
    print(f"Exportados {args.num_plots} gráficos a experiments/plots/")

if __name__ == "__main__":
    main()
