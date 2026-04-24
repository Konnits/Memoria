import os
import copy
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any
from torch.utils.data import DataLoader
from colorama import Fore, Style

from ts_transformer.data import (
    TimeSeriesDataset, build_collate_fn, StandardScaler, split_dataframe_by_time
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.data.sequence_builder import AutoregressiveSequenceBuilder
from ts_transformer.training import Trainer
from ts_transformer.models.time_series_encoder_decoder import TimeSeriesEncoderDecoder

def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if np.issubdtype(col.dtype, np.datetime64):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()

class AdvancedARTimeSeriesDataset(TimeSeriesDataset):
    """Dataset proxy capaz de intercalar modos autoregresivos por step."""
    def __init__(self, mode: str, base_data_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_mode = mode
        
        self.contiguous_offsets = list(range(1, self.config.num_targets + 1))
        
        if base_data_cfg.target_offset_choices is not None:
            self.random_offset_pool = [int(o) for o in base_data_cfg.target_offset_choices]
        elif getattr(base_data_cfg, "target_offset_max", None) is not None:
            o_min = int(getattr(base_data_cfg, "target_offset_min", 1))
            o_max = int(base_data_cfg.target_offset_max)
            self.random_offset_pool = list(range(o_min, o_max + 1))
        else:
            self.random_offset_pool = [int(getattr(base_data_cfg, "target_offset", 1))]

        # Ahora que conocemos todos los offsets posibles, reconstruimos los índices
        # para que max_anchor sea seguro para CUALQUIER modo.
        max_possible_offset = max(max(self.contiguous_offsets), max(self.random_offset_pool))
        
        # Override temporal sólo para calcular índices
        old_choices = self.config.target_offset_choices
        self.config.target_offset_choices = [max_possible_offset]
        self._example_indices = self._build_example_indices()
        self.config.target_offset_choices = old_choices

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        use_contiguous = True
        if self.ar_mode == "Random":
            use_contiguous = False
        elif self.ar_mode == "Mixed":
            use_contiguous = np.random.rand() > 0.5
            
        original_choices = self.config.target_offset_choices
        original_min = self.config.target_offset_min
        original_max = self.config.target_offset_max
        
        # Override temporary para la generación del anchor
        self.config.target_offset_choices = self.contiguous_offsets if use_contiguous else self.random_offset_pool
        self.config.target_offset_min = None
        self.config.target_offset_max = None
        
        try:
            ret = super().__getitem__(idx)
        finally:
            self.config.target_offset_choices = original_choices
            self.config.target_offset_min = original_min
            self.config.target_offset_max = original_max

        # Garantizar que siempre haya 'num_targets' tokens, rellenando si es necesario.
        # Esto soluciona errores de batches asimétricos al usar Mixed o Random con pool < num_targets
        num_targets_req = getattr(self.config, "num_targets", 1)
        k_actual = ret["target_values"].shape[0]

        if k_actual < num_targets_req:
            diff = num_targets_req - k_actual
            
            # Pad target_values with zeros
            pad_vals = torch.zeros((diff, ret["target_values"].shape[1]), dtype=ret["target_values"].dtype)
            ret["target_values"] = torch.cat([ret["target_values"], pad_vals], dim=0)
            
            # Pad target_timestamp by incrementing the last available timestamp
            last_ts = ret["target_timestamp"][-1].item() if k_actual > 0 else ret["past_timestamps"][-1].item()
            pad_ts = torch.arange(1, diff + 1, dtype=ret["target_timestamp"].dtype) + last_ts
            ret["target_timestamp"] = torch.cat([ret["target_timestamp"], pad_ts], dim=0)

            # Pad target_loss_mask with zeros (so the model isn't penalized for dummy outputs)
            pad_mask = torch.zeros((diff, ret["target_loss_mask"].shape[1]), dtype=ret["target_loss_mask"].dtype)
            ret["target_loss_mask"] = torch.cat([ret["target_loss_mask"], pad_mask], dim=0)
            
        return ret

def prepare_ar_data(mode: str, base_data_cfg, ds_idx, logger):
    target_csv = f"data/processed/real_data_{ds_idx}.csv"
    if not os.path.exists(target_csv):
        return None

    df = pd.read_csv(target_csv)
    time_col = base_data_cfg.time_column

    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col]).apply(lambda x: x.timestamp())

    df = df.sort_values(time_col).reset_index(drop=True)
    df_train, df_val, df_test = split_dataframe_by_time(
        df, time_column=time_col,
        train_ratio=base_data_cfg.train_ratio,
        val_ratio=base_data_cfg.val_ratio,
    )

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
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    num_targets = getattr(base_data_cfg, "num_targets", 4)
    if num_targets <= 1:
        num_targets = 4

    ds_cfg = TimeSeriesDatasetConfig(
        history_length=base_data_cfg.history_length,
        target_offset_choices=None, # handled dynamically
        stride=base_data_cfg.stride,
        min_history_length=base_data_cfg.min_history_length,
        num_targets=num_targets
    )

    sqb = AutoregressiveSequenceBuilder(
        input_dim=input_dim,
        use_sensor_ids=False,
        num_sensors=0,
        num_target_tokens=1
    )

    ds_tr = AdvancedARTimeSeriesDataset(mode, base_data_cfg, v_tr, ts_train, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
    ds_va = AdvancedARTimeSeriesDataset(mode, base_data_cfg, v_va, ts_val, ds_cfg, input_dim, output_dim, sequence_builder=sqb)
    # Test siempre se evalua consistentemente sobre continuous para comparar causalidad
    ds_te = AdvancedARTimeSeriesDataset("Contiguous", base_data_cfg, v_te, ts_test, ds_cfg, input_dim, output_dim, sequence_builder=sqb)

    collate_fn = build_collate_fn(pad_to_max_length=True)

    loader_kwargs = dict(
        batch_size=base_data_cfg.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    train_loader = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ds_va, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ds_te, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "t_scal": t_scal
    }

def run_ar_finetuning(mode: str, model: torch.nn.Module, model_name: str, base_data_cfg, ds_idx, training_cfg, base_ckpt_dir, device, logger):
    if not isinstance(model, TimeSeriesEncoderDecoder):
        return None
        
    logger.info(Fore.BLUE + f"    [AR-FT {mode}] Iniciando Finetuning Autoregresivo para {model_name}..." + Style.RESET_ALL)
    
    data = prepare_ar_data(mode, base_data_cfg, ds_idx, logger)
    if data is None:
        return None

    # Estrategia de Fine-Tuning: La mitad de epocas
    epochs = training_cfg.num_epochs // 2
    if epochs < 1: epochs = 1
        
    ft_cfg = copy.deepcopy(training_cfg)
    ft_cfg.num_epochs = epochs
    ft_cfg.checkpoint_dir = os.path.join(base_ckpt_dir, f"{model_name}_FT_AR_{mode}")
    
    # ESTRATEGIA DE CONGELAMIENTO (Catastrophic Forgetting Defense)
    ft_cfg.freeze_encoder_epochs = max(1, epochs // 2)
    # ESTRATEGIA DE FINE-TUNING (Reducir LR dramáticamente tras descongelar)
    base_lr = getattr(ft_cfg.optimizer_config, "learning_rate", 1e-3)
    ft_cfg.unfreeze_lr = base_lr / 10.0
    
    os.makedirs(ft_cfg.checkpoint_dir, exist_ok=True)
    
    ar_model = copy.deepcopy(model)
    ar_model.config.use_causal_mask = True
    
    trainer = Trainer(
        model=ar_model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        config=ft_cfg,
    )

    t_start = time.time()
    trainer.fit()
    train_time = time.time() - t_start
    
    ckpt_path = os.path.join(ft_cfg.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ar_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

    ar_model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
         for batch in data["test_loader"]:
             is_t_mask = batch["is_target_mask"].to(device)
             iv = batch["input_values"].to(device)
             it = batch["input_timestamps"].to(device)
             tv = batch["target_values"].to(device)
             
             B, L_tot, _ = iv.shape
             hist_mask = ~is_t_mask
             num_hist = int(hist_mask.sum(dim=1)[0].item())
             num_targ = int(is_t_mask.sum(dim=1)[0].item())
             
             history_v = iv[hist_mask].view(B, num_hist, -1)
             history_t = it[hist_mask].view(B, num_hist)
             target_t = it[is_t_mask].view(B, num_targ)
             
             preds = ar_model.generate(
                 history_values=history_v,
                 history_timestamps=history_t,
                 target_timestamps=target_t
             )
             
             all_preds.append(preds.cpu())
             all_targets.append(tv.cpu())

    if len(all_preds) == 0: return None
    
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    
    from ts_transformer.training.metrics import compute_regression_metrics
    test_metrics = compute_regression_metrics(preds_cat.view(-1, 1), targets_cat.view(-1, 1), prefix="test_ar_")
    
    test_metrics["train_time_s"] = round(train_time, 2)
    test_metrics["epochs_run"] = epochs
    
    del ar_model
    torch.cuda.empty_cache()
    
    return test_metrics
