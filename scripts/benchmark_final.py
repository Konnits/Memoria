"""
benchmark_final.py — Benchmark integral para la tesis.

Ejecuta de forma resumible:
  1. Todos los 17 datasets reales.
  2. Múltiples semillas por dataset (default: 3).
  3. Modelos: Custom, STraTS_Adapter, CoFormer-Uni/CoFormer_Adapter,
              Persistence, Linear,
              NoTimeEncoding (ablación), NoTargetToken (ablación).
              EncDec-Opt es opcional con --include-encdec.
  4. Evaluación en SPLIT DE TEST real (separado de validación).
  5. Medición de costo (tiempo de entrenamiento, nº de parámetros).
  6. Análisis estadístico al final (Wilcoxon, bootstrap, tablas resumen).

Uso:
  python scripts/benchmark_final.py
  python scripts/benchmark_final.py --start-dataset 7 --end-dataset 17
  python scripts/benchmark_final.py --seeds 42 84 126
  python scripts/benchmark_final.py --skip-ablation   (sólo modelos principales)
  python scripts/benchmark_final.py --include-large
  python scripts/benchmark_final.py --include-encdec --encdec-num-targets 8
  python scripts/benchmark_final.py --include-encdec --enable-encdec-finetuning --encdec-ft-modes Mixed

Se puede detener con Ctrl+C y reanudar ejecutando el mismo comando.
El progreso se guarda en el CSV después de cada modelo entrenado.
"""

import os
import sys
import argparse
import copy
import time
from typing import Any
from colorama import init, Fore, Style
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from torch.utils.data import DataLoader

from ts_transformer.data import (
    TimeSeriesDataset, EventTimeSeriesDataset, SequenceBuilder,
    build_collate_fn, StandardScaler, split_dataframe_by_time
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesTransformer, TimeSeriesEncoderDecoder
from ts_transformer.training import TrainingConfig, Trainer
from ts_transformer.utils import (
    load_data_config, load_model_config, load_training_config,
    set_global_seed, setup_logging, get_logger
)

from state_art.strats.model import STraTSNetwork
from state_art.coformer.model import CompatibleTransformer
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper
from state_art.simple_baselines import (
    PersistenceModel, LinearBaselineModel,
    NoTimeEncodingTransformer, NoTargetTokenTransformer,
)

from ar_finetuning import run_ar_finetuning


# ======================================================================
# Argumentos CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Final — Tesis Memoria",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--base-data-config", type=str,
                   default="configs/data/real_data1.yaml")
    p.add_argument("--model-config", type=str,
                   default="configs/model/transformer_base_real_data1.yaml")
    p.add_argument("--training-config", type=str,
                   default="configs/training/default_real_data1.yaml",
                   help="Training config por defecto (modelos medianos)")
    p.add_argument("--training-config-small", type=str,
                   default="configs/training/training_small.yaml",
                   help="Training config para modelos pequeños (<100K params)")
    p.add_argument("--training-config-large", type=str,
                   default="configs/training/training_large.yaml",
                   help="Training config para modelos grandes (>1M params)")
    p.add_argument("--start-dataset", type=int, default=1)
    p.add_argument("--end-dataset", type=int, default=17)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126],
                   help="Lista de semillas aleatorias (default: 42 84 126)")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Si se activa, omite NoTimeEncoding y NoTargetToken")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Si se activa, omite sólo los baselines simples Persistence y Linear")
    p.add_argument("--include-large", action="store_true",
                   help="Incluye variantes Large; por defecto se omiten para mantener viable el benchmark")
    p.add_argument("--include-encdec", action="store_true",
                   help="Incluye variantes optimizadas Encoder-Decoder")
    p.add_argument("--encdec-num-targets", type=int, default=0,
                   help="Targets de entrenamiento para EncDec-Opt; 0 usa el valor base del data config")
    p.add_argument("--enable-encdec-finetuning", action="store_true",
                   help="Activa fine-tuning autoregresivo para variantes EncDec")
    p.add_argument("--encdec-ft-modes", type=str, nargs="+",
                   choices=["Contiguous", "Random", "Mixed"],
                   default=["Mixed"],
                   help="Modos AR a ejecutar cuando --enable-encdec-finetuning está activo")
    p.add_argument("--encdec-ft-num-targets", type=int, default=0,
                   help="Targets para fine-tuning AR; 0 usa los targets de entrenamiento de EncDec")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Lista opcional de modelos a ejecutar (filtra después de construirlos)")
    p.add_argument("--exp-dir", type=str, default="experiments/benchmark_final",
                   help="Directorio de salida")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override de workers del DataLoader (default: usa data config)")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Prefetch por worker si num_workers > 0")
    return p.parse_args()


# ======================================================================
# Utilidades
# ======================================================================
def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(col):
        return (col.view("int64") / 1e9).astype("float32")
    return col.astype("float32").to_numpy()


def _fmt_metric(value: Any) -> str:
    """Formatea métricas numéricas de forma robusta para logging."""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def count_parameters(model: torch.nn.Module) -> int:
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """Cuenta parámetros totales."""
    return sum(p.numel() for p in model.parameters())


def configure_cuda_runtime(device: torch.device, logger) -> None:
    """Activa ajustes de runtime orientados a throughput en GPU."""
    if device.type != "cuda":
        logger.info("  Runtime: CPU")
        return

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger.info("  Runtime CUDA optimizado: TF32 + cuDNN benchmark habilitados")


# Umbrales para selección automática de training config por tamaño de modelo.
_SMALL_THRESHOLD = 100_000    # < 100K params → training_small
_LARGE_THRESHOLD = 1_000_000  # > 1M params  → training_large


def _estimate_dataset_time_scale(
    timestamps: np.ndarray,
    fallback: float = 900.0,
) -> float:
    """
    Estima un time_scale robusto por dataset usando la mediana de deltas > 0.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size < 3:
        return float(fallback)

    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float(fallback)

    # Clamp conservador para evitar escalas patológicas.
    return float(np.clip(np.median(diffs), 1.0, 86_400.0))


def _build_balanced_subset(
    df: pd.DataFrame,
    required_models: list[str],
    dataset_col: str = "Dataset_ID",
    seed_col: str = "Seed",
    model_col: str = "Modelo",
) -> pd.DataFrame:
    """
    Filtra filas para quedarse solo con pares (dataset, seed) que tienen
    resultados para todos los modelos requeridos.
    """
    if not required_models:
        return df.copy()

    df_req = df[df[model_col].isin(required_models)].copy()
    if df_req.empty:
        return df_req

    n_required = len(required_models)
    pair_counts = (
        df_req.groupby([dataset_col, seed_col])[model_col]
        .nunique()
        .reset_index(name="n_models")
    )
    valid_pairs = pair_counts[pair_counts["n_models"] == n_required][[dataset_col, seed_col]]
    if valid_pairs.empty:
        return df_req.iloc[0:0].copy()

    valid_pairs = valid_pairs.assign(__keep__=1)
    df_balanced = df_req.merge(valid_pairs, on=[dataset_col, seed_col], how="inner")
    return df_balanced.drop(columns=["__keep__"])


def select_training_config(
    model: torch.nn.Module,
    cfg_small: TrainingConfig,
    cfg_default: TrainingConfig,
    cfg_large: TrainingConfig,
    model_name: str,
    logger,
) -> TrainingConfig:
    """
    Selecciona la configuración de entrenamiento apropiada según
    el número de parámetros del modelo.

    - < 100K params  → cfg_small  (lr=0.001, warmup=2, epochs=30)
    - 100K-1M params → cfg_default (alineado con epochs=30, warmup=2)
    - > 1M params    → cfg_large  (lr=0.0002, warmup=5, epochs=60)
    """
    n_params = count_parameters(model)
    if n_params < _SMALL_THRESHOLD:
        chosen = copy.deepcopy(cfg_small)
        tier = "small"
    elif n_params > _LARGE_THRESHOLD:
        chosen = copy.deepcopy(cfg_large)
        tier = "large"
    else:
        chosen = copy.deepcopy(cfg_default)
        tier = "default"

    # Custom (familia mediana) es el más propenso a inestabilidad con el
    # tier default: historia/offsets variables + MSE + lr=1e-3 puede hacerlo
    # divergir tras el warmup. Lo mantenemos en su tier por comparabilidad,
    # pero usamos una receta más suave.
    if model_name.startswith("Custom") and ("Small" not in model_name and "Large" not in model_name):
        chosen.loss_name = "huber"
        chosen.grad_clip_norm = min(float(chosen.grad_clip_norm), 0.5)
        chosen.optimizer_config.lr = min(float(chosen.optimizer_config.lr), 3e-4)
        chosen.optimizer_config.warmup_epochs = max(
            int(chosen.optimizer_config.warmup_epochs),
            5,
        )
        chosen.optimizer_config.weight_decay = max(
            float(chosen.optimizer_config.weight_decay),
            0.003,
        )

    if model_name.startswith("EncDec-Opt"):
        chosen.loss_name = "huber"
        chosen.grad_clip_norm = (
            0.5 if float(chosen.grad_clip_norm) <= 0.0
            else min(float(chosen.grad_clip_norm), 0.5)
        )
        chosen.optimizer_config.lr = min(float(chosen.optimizer_config.lr), 3e-4)
        chosen.optimizer_config.warmup_epochs = max(
            int(chosen.optimizer_config.warmup_epochs),
            5,
        )
        chosen.optimizer_config.weight_decay = max(
            float(chosen.optimizer_config.weight_decay),
            0.003,
        )

    logger.info(
        f"    [CONFIG] {model_name} ({n_params:,} params) → training tier '{tier}' "
        f"(lr={chosen.optimizer_config.lr}, epochs={chosen.num_epochs}, "
        f"warmup={chosen.optimizer_config.warmup_epochs}, "
        f"wd={chosen.optimizer_config.weight_decay})"
    )
    return chosen


# ======================================================================
# Construcción de modelos
# ======================================================================
def build_models(
    model_cfg,
    dataset_time_scale: float,
    use_events: bool,
    model_input_dim: int,
    input_dim: int,
    output_dim: int,
    skip_ablation: bool = False,
    skip_baselines: bool = False,
    include_large: bool = False,
    include_encdec: bool = False,
    encdec_num_targets: int | None = None,
):
    """
    Construye todos los modelos a evaluar.

    Returns
    -------
    dict[str, nn.Module]
        Nombre → modelo instanciado.
    dict[str, bool]
        Nombre → True si el modelo requiere entrenamiento.
    """
    model_cfg.input_dim = model_input_dim
    model_cfg.output_dim = output_dim
    model_cfg.use_sensor_embedding = bool(use_events)
    model_cfg.num_sensors = input_dim if use_events else 0
    model_cfg.time_scale = float(dataset_time_scale)

    d_model = model_cfg.d_model

    models = {}
    trainable = {}

    cfg_small = load_model_config("configs/model/transformer_small.yaml")
    cfg_small.input_dim = model_input_dim
    cfg_small.output_dim = output_dim
    cfg_small.use_sensor_embedding = bool(use_events)
    cfg_small.num_sensors = input_dim if use_events else 0
    cfg_small.time_scale = float(dataset_time_scale)

    cfg_large = None
    if include_large:
        cfg_large = load_model_config("configs/model/transformer_large.yaml")
        cfg_large.input_dim = model_input_dim
        cfg_large.output_dim = output_dim
        cfg_large.use_sensor_embedding = bool(use_events)
        cfg_large.num_sensors = input_dim if use_events else 0
        cfg_large.time_scale = float(dataset_time_scale)

    # --- Modelo propuesto ---
    models["Custom-Small"] = TimeSeriesTransformer(copy.deepcopy(cfg_small))
    trainable["Custom-Small"] = True
    models["Custom"] = TimeSeriesTransformer(copy.deepcopy(model_cfg))
    trainable["Custom"] = True

    def _make_readout_variant_config(base_cfg, *, readout_mode: str):
        cfg = copy.deepcopy(base_cfg)
        cfg.readout_mode = readout_mode
        return cfg

    models["Custom-AttnPool-Small"] = TimeSeriesTransformer(
        _make_readout_variant_config(
            cfg_small,
            readout_mode="target_plus_attention_pool",
        )
    )
    trainable["Custom-AttnPool-Small"] = True
    models["Custom-AttnPool"] = TimeSeriesTransformer(
        _make_readout_variant_config(
            model_cfg,
            readout_mode="target_plus_attention_pool",
        )
    )
    trainable["Custom-AttnPool"] = True
    if include_large and cfg_large is not None:
        models["Custom-Large"] = TimeSeriesTransformer(copy.deepcopy(cfg_large))
        trainable["Custom-Large"] = True

    # --- Variante Encoder-Decoder optimizada ---
    def _make_encdec_config(base_cfg, *, decoder_num_layers: int = 1):
        cfg = copy.deepcopy(base_cfg)
        cfg.decoder_num_layers = int(decoder_num_layers)
        return cfg

    if include_encdec:
        mt_suffix = (
            f"-MT{int(encdec_num_targets)}"
            if encdec_num_targets is not None and int(encdec_num_targets) > 0
            else ""
        )
        models[f"EncDec-Opt-Small{mt_suffix}"] = TimeSeriesEncoderDecoder(
            _make_encdec_config(cfg_small, decoder_num_layers=1)
        )
        trainable[f"EncDec-Opt-Small{mt_suffix}"] = True
        models[f"EncDec-Opt{mt_suffix}"] = TimeSeriesEncoderDecoder(
            _make_encdec_config(model_cfg, decoder_num_layers=1)
        )
        trainable[f"EncDec-Opt{mt_suffix}"] = True

    # --- Variantes temporales aprendidas ---
    # Separar encoding temporal y sesgo de atención evita confundir qué mejora
    # o empeora al comparar contra el baseline sinusoidal.
    def _make_temporal_variant_config(
        base_cfg,
        *,
        time_encoding_mode: str,
        use_temporal_attn_bias: bool,
        time_transform: str | None = None,
        temporal_bias_layers: int | None = None,
    ):
        """Crea una copia del config con la variante temporal solicitada."""
        cfg = copy.deepcopy(base_cfg)
        cfg.time_encoding_mode = time_encoding_mode
        cfg.use_temporal_attn_bias = use_temporal_attn_bias
        cfg.temporal_bias_layers = temporal_bias_layers
        if time_transform is not None:
            cfg.time_transform = time_transform
        return cfg

    models["Custom-Time2Vec-Small"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            cfg_small,
            time_encoding_mode="time2vec",
            use_temporal_attn_bias=False,
            time_transform="linear",
        )
    )
    trainable["Custom-Time2Vec-Small"] = True
    models["Custom-Time2Vec"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            model_cfg,
            time_encoding_mode="time2vec",
            use_temporal_attn_bias=False,
            time_transform="linear",
        )
    )
    trainable["Custom-Time2Vec"] = True

    models["Custom-TempBias-Small"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            cfg_small,
            time_encoding_mode="sinusoidal",
            use_temporal_attn_bias=True,
            temporal_bias_layers=1,
        )
    )
    trainable["Custom-TempBias-Small"] = True
    models["Custom-TempBias"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            model_cfg,
            time_encoding_mode="sinusoidal",
            use_temporal_attn_bias=True,
            temporal_bias_layers=1,
        )
    )
    trainable["Custom-TempBias"] = True

    models["Custom-Time2VecBias-Small"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            cfg_small,
            time_encoding_mode="time2vec",
            use_temporal_attn_bias=True,
            time_transform="linear",
            temporal_bias_layers=1,
        )
    )
    trainable["Custom-Time2VecBias-Small"] = True
    models["Custom-Time2VecBias"] = TimeSeriesTransformer(
        _make_temporal_variant_config(
            model_cfg,
            time_encoding_mode="time2vec",
            use_temporal_attn_bias=True,
            time_transform="linear",
            temporal_bias_layers=1,
        )
    )
    trainable["Custom-Time2VecBias"] = True

    # --- STraTS Adapter ---
    # num_features = input_dim + 1: el +1 es para el feature_id marcador
    # de target (= D) que permite a STraTS distinguir tripletas de historia
    # de tripletas target en el transformer (comparación justa).
    s_base = STraTSNetwork(
        num_features=input_dim + 1,
        d_model=d_model,
        num_classes=output_dim,
    )
    models["STraTS_Adapter"] = MultiHorizonBaselineWrapper(
        s_base, "strats", d_model, output_dim, use_sensor_embedding=use_events
    )
    trainable["STraTS_Adapter"] = True

    # --- CoFormer Adapter ---
    # En datasets univariados, CompatibleTransformer desactiva la rama
    # inter-variate y queda como una variante CoFormer-Uni más barata.
    coformer_num_variates = input_dim if use_events else 1
    c_base = CompatibleTransformer(
        num_variates=coformer_num_variates,
        d_model=d_model,
        n_heads=model_cfg.num_heads,
        n_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        num_classes=output_dim,
    )
    coformer_name = (
        "CoFormer-Uni" if coformer_num_variates == 1 else "CoFormer_Adapter"
    )
    models[coformer_name] = MultiHorizonBaselineWrapper(
        c_base, "coformer", d_model, output_dim, use_sensor_embedding=use_events
    )
    trainable[coformer_name] = True

    if not skip_baselines:
        # --- Persistence (no entrena) ---
        models["Persistence"] = PersistenceModel(
            input_dim=model_input_dim, output_dim=output_dim
        )
        trainable["Persistence"] = False

        # --- Linear Baseline ---
        models["Linear"] = LinearBaselineModel(
            input_dim=model_input_dim,
            output_dim=output_dim,
            d_model=64,
            max_history=50,
        )
        trainable["Linear"] = True

    if not skip_ablation:
        # --- Ablación: sin encoding temporal continuo ---
        models["NoTimeEncoding"] = NoTimeEncodingTransformer(copy.deepcopy(model_cfg))
        trainable["NoTimeEncoding"] = True

        # --- Ablación: sin target token ---
        models["NoTargetToken"] = NoTargetTokenTransformer(copy.deepcopy(model_cfg))
        trainable["NoTargetToken"] = True

    return models, trainable


# ======================================================================
# Entrenamiento y evaluación de un modelo
# ======================================================================
def train_and_evaluate(
    model,
    model_name: str,
    needs_training: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    training_cfg: TrainingConfig,
    checkpoint_dir: str,
    device: torch.device,
    logger,
) -> dict:
    """
    Entrena un modelo (si necesita entrenamiento), lo evalúa en test,
    y retorna un diccionario con métricas + costo.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg_mod = copy.deepcopy(training_cfg)
    cfg_mod.checkpoint_dir = checkpoint_dir

    n_params_trainable = count_parameters(model)
    n_params_total = count_total_parameters(model)

    if needs_training:
        # Entrenar
        logger.info(f"    Entrenando {model_name} ({n_params_trainable:,} params)...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg_mod,
        )

        t_start = time.time()
        history = trainer.fit()
        train_time = time.time() - t_start
        epochs_run = len(history.get("train_loss", []))

        # Cargar mejor checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
    else:
        # Modelo sin entrenamiento (e.g. Persistence)
        model = model.to(device)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg_mod,
        )
        train_time = 0.0
        epochs_run = 0

    # Evaluación en TEST (split real separado)
    logger.info(f"    Evaluando {model_name} en TEST...")
    test_metrics = trainer.evaluate_on_loader(test_loader, prefix="test_")

    # Evaluación en VAL para comparación
    val_metrics = trainer.evaluate_on_loader(val_loader, prefix="val_")

    record = {}
    record.update(test_metrics)
    record.update(val_metrics)
    record["train_time_s"] = round(train_time, 2)
    record["epochs_run"] = epochs_run
    record["n_params_trainable"] = n_params_trainable
    record["n_params_total"] = n_params_total

    return record


# ======================================================================
# Pipeline de datos
# ======================================================================
def prepare_data(
    base_data_cfg,
    ds_idx,
    logger,
    num_workers_override: int | None = None,
    prefetch_factor: int = 4,
    num_targets_override: int | None = None,
):
    """Carga y prepara un dataset concreto. Retorna loaders y metadatos."""
    target_csv = f"data/processed/real_data_{ds_idx}.csv"
    if not os.path.exists(target_csv):
        logger.warning(
            Fore.YELLOW
            + f"Saltando Dataset {ds_idx}: no existe {target_csv}"
            + Style.RESET_ALL
        )
        return None

    df = pd.read_csv(target_csv)
    time_col = base_data_cfg.time_column

    # Asegurar timestamps numéricos
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

    adaptive_time_scale = _estimate_dataset_time_scale(ts_train, fallback=900.0)

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

    num_targets = (
        int(num_targets_override)
        if num_targets_override is not None and int(num_targets_override) > 0
        else int(base_data_cfg.num_targets)
    )

    ds_cfg = TimeSeriesDatasetConfig(
        history_length=base_data_cfg.history_length,
        target_offset_choices=base_data_cfg.target_offset_choices,
        target_offset_min=base_data_cfg.target_offset_min,
        target_offset_max=base_data_cfg.target_offset_max,
        stride=base_data_cfg.stride,
        min_history_length=base_data_cfg.min_history_length,
        num_targets=num_targets,
    )

    use_events = base_data_cfg.use_event_tokens

    if use_events:
        tsid = [
            (
                base_data_cfg.feature_columns.index(c)
                if c in base_data_cfg.feature_columns
                else input_dim
            )
            for c in base_data_cfg.target_columns
        ]
        sqb = SequenceBuilder(
            input_dim=1,
            target_token_value="zeros",
            use_sensor_ids=True,
            num_sensors=input_dim,
            num_target_tokens=output_dim,
            target_sensor_ids=tsid,
        )
        ds_tr = EventTimeSeriesDataset(
            X_train_s, ts_train, y_train_s, ds_cfg, input_dim, output_dim,
            sequence_builder=sqb,
        )
        ds_va = EventTimeSeriesDataset(
            X_val_s, ts_val, y_val_s, ds_cfg, input_dim, output_dim,
            sequence_builder=sqb,
        )
        ds_te = EventTimeSeriesDataset(
            X_test_s, ts_test, y_test_s, ds_cfg, input_dim, output_dim,
            sequence_builder=sqb,
        )
        mi_dim = 1
    else:
        sqb = SequenceBuilder(
            input_dim=input_dim,
            target_token_value="zeros",
            use_sensor_ids=False,
            num_sensors=0,
            num_target_tokens=1,
        )
        ds_tr = TimeSeriesDataset(
            v_tr, ts_train, ds_cfg, input_dim, output_dim, sequence_builder=sqb
        )
        ds_va = TimeSeriesDataset(
            v_va, ts_val, ds_cfg, input_dim, output_dim, sequence_builder=sqb
        )
        ds_te = TimeSeriesDataset(
            v_te, ts_test, ds_cfg, input_dim, output_dim, sequence_builder=sqb
        )
        mi_dim = input_dim

    collate_fn = build_collate_fn(pad_to_max_length=True)

    cfg_workers = int(getattr(base_data_cfg, "num_workers", 0))
    num_workers = cfg_workers if num_workers_override is None else int(num_workers_override)
    num_workers = max(0, num_workers)

    loader_kwargs: dict[str, Any] = {
        "batch_size": base_data_cfg.batch_size,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": num_workers,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ds_va, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ds_te, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model_input_dim": mi_dim,
        "use_events": use_events,
        "n_train": len(ds_tr),
        "n_val": len(ds_va),
        "n_test": len(ds_te),
        "adaptive_time_scale": adaptive_time_scale,
        "num_targets": int(getattr(ds_tr, "k_targets", num_targets)),
    }


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    init(autoreset=True)
    setup_logging()
    logger = get_logger("benchmark_final")

    if args.enable_encdec_finetuning and not args.include_encdec:
        args.include_encdec = True
        logger.info("  --enable-encdec-finetuning activa también --include-encdec.")

    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, "benchmark_final.csv")

    enable_finetuning = bool(args.enable_encdec_finetuning)

    logger.info(
        Fore.GREEN
        + f"=== BENCHMARK FINAL ==="
        + Style.RESET_ALL
    )
    logger.info(f"  Datasets: {args.start_dataset} a {args.end_dataset}")
    logger.info(f"  Semillas: {args.seeds}")
    logger.info(f"  Ablación: {'NO' if args.skip_ablation else 'SÍ'}")
    logger.info(f"  Baselines simples: {'NO' if args.skip_baselines else 'SÍ'}")
    logger.info(f"  Modelos Large: {'SÍ' if args.include_large else 'NO'}")
    logger.info(f"  EncDec optimizado: {'SÍ' if args.include_encdec else 'NO'}")
    if args.include_encdec:
        logger.info(
            "  EncDec train targets: "
            + ("base data config" if args.encdec_num_targets <= 0 else str(args.encdec_num_targets))
        )
    logger.info(f"  Fine-tuning AR: {'SÍ' if enable_finetuning else 'NO'}")
    if enable_finetuning:
        logger.info(f"  Fine-tuning AR modos: {args.encdec_ft_modes}")
    logger.info(f"  Directorio: {exp_dir}")
    if args.models:
        logger.info(f"  Filtro de modelos: {args.models}")
    logger.info(
        "  DataLoader: "
        f"num_workers={'data_config' if args.num_workers is None else args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}"
    )

    # Cargar configs base
    base_data_cfg = load_data_config(args.base_data_config)
    model_cfg_base = load_model_config(args.model_config)
    training_cfg, _ = load_training_config(args.training_config)

    # Cargar configs especializadas por tamaño de modelo
    if os.path.exists(args.training_config_small):
        training_cfg_small, _ = load_training_config(args.training_config_small)
        logger.info(f"  Config small: {args.training_config_small}")
    else:
        training_cfg_small = training_cfg
        logger.info("  Config small: usando default (archivo no encontrado)")

    if os.path.exists(args.training_config_large):
        training_cfg_large, _ = load_training_config(args.training_config_large)
        logger.info(f"  Config large: {args.training_config_large}")
    else:
        training_cfg_large = training_cfg
        logger.info("  Config large: usando default (archivo no encontrado)")

    device = torch.device(
        training_cfg.device if torch.cuda.is_available() else "cpu"
    )
    configure_cuda_runtime(device, logger)

    # Cargar resultados previos para resume
    master_results: list[dict[str, Any]] = []
    completed_runs = set()
    if os.path.exists(out_csv):
        df_exist = pd.read_csv(out_csv)
        master_results = [
            {str(k): v for k, v in rec.items()}
            for rec in df_exist.to_dict("records")
        ]
        for row in master_results:
            completed_runs.add((row["Dataset_ID"], row["Seed"], row["Modelo"]))
        logger.info(
            Fore.YELLOW
            + f"  [RESUME] {len(completed_runs)} corridas previas detectadas."
            + Style.RESET_ALL
        )

    # Iterar sobre datasets
    for ds_idx in range(args.start_dataset, args.end_dataset + 1):
        logger.info(
            Fore.CYAN
            + f"\n{'='*60}"
            + f"\n  DATASET {ds_idx}"
            + f"\n{'='*60}"
            + Style.RESET_ALL
        )

        data = prepare_data(
            base_data_cfg,
            ds_idx,
            logger,
            num_workers_override=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if data is None:
            continue

        logger.info(
            f"  Muestras: train={data['n_train']}, "
            f"val={data['n_val']}, test={data['n_test']}"
        )
        logger.info(
            f"  time_scale adaptativo (train median dt): {data['adaptive_time_scale']:.3f}"
        )

        encdec_data = None
        encdec_train_targets = int(data["num_targets"])
        if args.include_encdec and args.encdec_num_targets > int(data["num_targets"]):
            encdec_train_targets = int(args.encdec_num_targets)
            logger.info(
                f"  EncDec-Opt entrenará con {encdec_train_targets} targets "
                f"y evaluará con {data['num_targets']}."
            )
            encdec_data = prepare_data(
                base_data_cfg,
                ds_idx,
                logger,
                num_workers_override=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                num_targets_override=encdec_train_targets,
            )
            if encdec_data is None:
                logger.warning(
                    Fore.YELLOW
                    + "  No se pudo preparar data extra para EncDec; se usará la data base."
                    + Style.RESET_ALL
                )
                encdec_data = None
                encdec_train_targets = int(data["num_targets"])
            else:
                encdec_train_targets = int(encdec_data["num_targets"])
                logger.info(f"  EncDec-Opt targets efectivos de entrenamiento: {encdec_train_targets}")
                if encdec_train_targets <= int(data["num_targets"]):
                    encdec_data = None

        for seed in args.seeds:
            logger.info(
                Fore.MAGENTA
                + f"\n  [Dataset {ds_idx} | Seed {seed}]"
                + Style.RESET_ALL
            )
            set_global_seed(seed, deterministic=False)

            # Construir modelos frescos para cada semilla
            model_cfg_runtime = copy.deepcopy(model_cfg_base)
            model_cfg_runtime.time_scale = float(data["adaptive_time_scale"])
            models, trainable = build_models(
                model_cfg_runtime,
                data["adaptive_time_scale"],
                data["use_events"],
                data["model_input_dim"],
                data["input_dim"],
                data["output_dim"],
                skip_ablation=args.skip_ablation,
                skip_baselines=args.skip_baselines,
                include_large=args.include_large,
                include_encdec=args.include_encdec,
                encdec_num_targets=(
                    encdec_train_targets
                    if encdec_train_targets > int(data["num_targets"])
                    else None
                ),
            )

            if args.models:
                requested_models = set(args.models)
                available_models = set(models.keys())
                available_aliases = available_models | {
                    name.split("-MT", 1)[0] for name in available_models
                }
                unknown_models = sorted(requested_models - available_aliases)
                if unknown_models:
                    logger.warning(
                        Fore.YELLOW
                        + f"    [MODELS] Modelos desconocidos ignorados: {unknown_models}"
                        + Style.RESET_ALL
                    )
                models = {
                    name: model
                    for name, model in models.items()
                    if name in requested_models or name.split("-MT", 1)[0] in requested_models
                }
                trainable = {
                    name: trainable[name]
                    for name in models.keys()
                }
                if not models:
                    logger.warning(
                        Fore.YELLOW
                        + "    [MODELS] Ningún modelo seleccionado para esta corrida; se omite la semilla."
                        + Style.RESET_ALL
                    )
                    continue

            for name, model in models.items():
                requires_base_training = (ds_idx, seed, name) not in completed_runs
                
                # Check for EncDec fine-tunings completion
                missing_fts = []
                if enable_finetuning and "EncDec" in name:
                    for mode in args.encdec_ft_modes:
                        if (ds_idx, seed, f"{name}_FT_AR_{mode}") not in completed_runs:
                            missing_fts.append(mode)

                if not requires_base_training and not missing_fts:
                    logger.info(f"    [SKIP] {name} y sus variantes (completados)")
                    continue
                    
                if (
                    enable_finetuning
                    and not requires_base_training
                    and "EncDec" in name
                    and missing_fts
                ):
                    logger.info(f"    [RESUME-FT] {name} base ya evaluado. Faltan fine-tunings: {missing_fts}")

                try:
                    ckpt_dir = os.path.join(
                        exp_dir, f"ds_{ds_idx}_seed_{seed}", name
                    )
                    if (
                        enable_finetuning
                        and not requires_base_training
                        and "EncDec" in name
                        and missing_fts
                    ):
                        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                        if not os.path.exists(ckpt_path):
                            logger.warning(
                                Fore.YELLOW
                                + f"    [RESUME-FT] {name} figura en CSV, pero falta best_model.pt; "
                                + "se reentrenará la base antes del fine-tuning."
                                + Style.RESET_ALL
                            )
                            requires_base_training = True

                    if requires_base_training:
                        is_encdec_opt = name.startswith("EncDec-Opt")
                        train_data_for_model = (
                            encdec_data
                            if is_encdec_opt and encdec_data is not None
                            else data
                        )
                        # Seleccionar config de training según tamaño del modelo
                        effective_cfg = select_training_config(
                            model, training_cfg_small, training_cfg,
                            training_cfg_large, name, logger,
                        )
                        record = train_and_evaluate(
                            model=model,
                            model_name=name,
                            needs_training=trainable[name],
                            train_loader=train_data_for_model["train_loader"],
                            val_loader=data["val_loader"],
                            test_loader=data["test_loader"],
                            training_cfg=effective_cfg,
                            checkpoint_dir=ckpt_dir,
                            device=device,
                            logger=logger,
                        )

                        record["Dataset_ID"] = ds_idx
                        record["Modelo"] = name
                        record["Seed"] = seed
                        record["n_train"] = train_data_for_model["n_train"]
                        record["n_val"] = data["n_val"]
                        record["n_test"] = data["n_test"]
                        record["train_num_targets"] = int(train_data_for_model["num_targets"])
                        record["eval_num_targets"] = int(data["num_targets"])

                        master_results.append(record)

                        # Guardar progreso inmediatamente
                        pd.DataFrame(master_results).to_csv(out_csv, index=False)
                        logger.info(
                            Fore.GREEN
                            + f"    [OK] {name} ds={ds_idx} seed={seed} → "
                            + f"test_mse={_fmt_metric(record.get('test_mse'))}, "
                            + f"time={record.get('train_time_s', 0):.1f}s"
                            + Style.RESET_ALL
                        )
                    else:
                        # Load the model's base weights for fine-tunings since base training was skipped
                        if enable_finetuning and "EncDec" in name and missing_fts:
                            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                            if os.path.exists(ckpt_path):
                                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                                logger.info(Fore.BLUE + f"    [STATE] Cargados pesos base de {name} para reanudar FT." + Style.RESET_ALL)

                    # ------ INICIO FINE-TUNING RECURSIVO ------
                    if enable_finetuning and "EncDec" in name and missing_fts:
                        for mode in missing_fts:
                            ft_num_targets = (
                                int(args.encdec_ft_num_targets)
                                if int(args.encdec_ft_num_targets) > 0
                                else (
                                    int(encdec_train_targets)
                                    if name.startswith("EncDec-Opt")
                                    else int(data["num_targets"])
                                )
                            )
                            ft_record_raw = run_ar_finetuning(
                                mode=mode,
                                model=model,
                                model_name=name,
                                base_data_cfg=base_data_cfg,
                                ds_idx=ds_idx,
                                training_cfg=training_cfg,
                                base_ckpt_dir=os.path.join(exp_dir, f"ds_{ds_idx}_seed_{seed}"),
                                device=device,
                                logger=logger,
                                num_targets_override=ft_num_targets,
                                num_workers_override=args.num_workers,
                                prefetch_factor=args.prefetch_factor,
                            )
                            if ft_record_raw:
                                ft_record: dict[str, Any] = dict(ft_record_raw)
                                # Reempaquetar con el nombre nuevo reflejando el modo
                                ft_record["Dataset_ID"] = ds_idx
                                ft_record["Modelo"] = f"{name}_FT_AR_{mode}"
                                ft_record["Seed"] = seed
                                ft_record["n_train"] = int(ft_record.get("n_train", data["n_train"]))
                                ft_record["n_val"] = int(ft_record.get("n_val", data["n_val"]))
                                ft_record["n_test"] = int(ft_record.get("n_test", data["n_test"]))
                                ft_record["train_num_targets"] = int(ft_record.get("train_num_targets", ft_num_targets))
                                ft_record["eval_num_targets"] = int(ft_record.get("eval_num_targets", ft_num_targets))
                                
                                # Adaptar el prefijo test_ar_
                                ft_record["test_mse"] = ft_record.pop("test_ar_mse", float("nan"))
                                ft_record["test_rmse"] = ft_record.pop("test_ar_rmse", float("nan"))
                                ft_record["test_mae"] = ft_record.pop("test_ar_mae", float("nan"))

                                master_results.append(ft_record)
                                pd.DataFrame(master_results).to_csv(out_csv, index=False)
                                logger.info(
                                    Fore.MAGENTA
                                    + f"    [FT-OK] {name}_FT_AR_{mode} ds={ds_idx} seed={seed} → "
                                    + f"test_mse={_fmt_metric(ft_record.get('test_mse'))}"
                                    + Style.RESET_ALL
                                )
                    # ------ FIN FINE-TUNING RECURSIVO ------

                except Exception as e:
                    logger.error(
                        Fore.RED
                        + f"    [ERROR] {name} ds={ds_idx} seed={seed}: {e}"
                        + Style.RESET_ALL
                    )
                    import traceback
                    traceback.print_exc()
                    continue

                # Liberar memoria GPU
                del model
                torch.cuda.empty_cache()

            # Liberar modelos dict
            del models
            torch.cuda.empty_cache()

    # =================================================================
    # Análisis estadístico final
    # =================================================================
    logger.info(
        Fore.GREEN + "\n" + "=" * 60 + "\n  ANÁLISIS ESTADÍSTICO" + "\n" + "=" * 60
        + Style.RESET_ALL
    )

    df_all = pd.DataFrame(master_results)
    if df_all.empty:
        logger.warning("No hay resultados para analizar.")
        return

    # Si hubo reanudaciones, conservar la última corrida por (dataset, seed, modelo).
    dedup_cols = ["Dataset_ID", "Seed", "Modelo"]
    if all(c in df_all.columns for c in dedup_cols):
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=dedup_cols, keep="last").copy()
        after = len(df_all)
        if after != before:
            logger.info(f"  [DEDUP] Filas únicas por (dataset, seed, modelo): {before} -> {after}")

    df_all.to_csv(out_csv, index=False)

    # Diagnóstico de cobertura por (dataset, seed): cuántos modelos hay realmente.
    coverage = (
        df_all.groupby(["Dataset_ID", "Seed"])["Modelo"]
        .nunique()
        .reset_index(name="n_models")
        .sort_values(["Dataset_ID", "Seed"])
    )
    coverage_path = os.path.join(exp_dir, "coverage_by_dataset_seed.csv")
    coverage.to_csv(coverage_path, index=False)
    logger.info(f"  Cobertura por dataset/seed guardada en {coverage_path}")

    preferred_models = [
        "Custom",
        "Custom-AttnPool",
        "Custom-Small",
        "Custom-AttnPool-Small",
        "Linear",
        "NoTimeEncoding",
        "STraTS_Adapter",
        "CoFormer-Uni",
        "CoFormer_Adapter",
        "EncDec-Opt",
        "EncDec-Opt-Small",
    ]
    available_models = sorted(df_all["Modelo"].unique().tolist())
    for model_name in available_models:
        if model_name.startswith("EncDec-Opt") and model_name not in preferred_models:
            preferred_models.append(model_name)
    required_models = [m for m in preferred_models if m in available_models]
    if len(required_models) < 2:
        required_models = available_models

    df_stats = _build_balanced_subset(df_all, required_models)
    if not df_stats.empty:
        balanced_path = os.path.join(exp_dir, "benchmark_final_balanced.csv")
        df_stats.to_csv(balanced_path, index=False)
        logger.info(
            f"  [BALANCED] Análisis estadístico en subset balanceado con modelos={required_models}; "
            f"filas={len(df_stats)}"
        )
        logger.info(f"  [BALANCED] CSV guardado en {balanced_path}")
    else:
        logger.warning(
            "  [BALANCED] No hay subset balanceado para los modelos objetivo; "
            "se usará el CSV completo para estadísticas."
        )
        df_stats = df_all

    # Importar análisis
    from statistical_analysis import (
        generate_summary_table,
        compute_pairwise_comparison,
        generate_full_report,
    )

    test_metrics = ["test_mse", "test_rmse", "test_mae"]
    available_metrics = [m for m in test_metrics if m in df_stats.columns]

    if available_metrics:
        report = generate_full_report(
            df_stats,
            reference_model="Custom",
            metrics=available_metrics,
        )
        print(report)

        # Guardar reporte
        report_path = os.path.join(exp_dir, "statistical_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"  Reporte guardado en {report_path}")

        # Tabla resumen
        summary = generate_summary_table(df_stats, available_metrics)
        summary_path = os.path.join(exp_dir, "summary_table.csv")
        summary.to_csv(summary_path, index=False)
        logger.info(f"  Tabla resumen guardada en {summary_path}")

        # Comparaciones emparejadas
        comparisons = []
        other_models = [
            m for m in df_stats["Modelo"].unique() if m != "Custom"
        ]
        for metric in available_metrics:
            for other in other_models:
                comp = compute_pairwise_comparison(
                    df_stats, "Custom", other, metric
                )
                comparisons.append(comp)

        comp_df = pd.DataFrame(comparisons)
        comp_path = os.path.join(exp_dir, "pairwise_comparisons.csv")
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"  Comparaciones guardadas en {comp_path}")

    # Tabla de costo
    cost_cols = ["Modelo", "n_params_trainable", "n_params_total", "train_time_s", "epochs_run"]
    avail_cost = [c for c in cost_cols if c in df_all.columns]
    if len(avail_cost) > 1:
        cost_summary = (
            df_all.groupby("Modelo")[
                [c for c in avail_cost if c != "Modelo"]
            ].mean().round(2)
        )
        cost_path = os.path.join(exp_dir, "cost_summary.csv")
        cost_summary.to_csv(cost_path)
        logger.info(f"  Costos guardados en {cost_path}")
        print("\nCOSTO PROMEDIO POR MODELO:")
        print(cost_summary.to_string())

    logger.info(
        Fore.GREEN
        + "\n==== BENCHMARK FINAL COMPLETADO ===="
        + Style.RESET_ALL
    )


if __name__ == "__main__":
    main()
