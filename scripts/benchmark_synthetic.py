"""Benchmark reproducible para las colecciones sintéticas univariadas y multivariadas.

Los Parquet deben generarse antes con::

    conda activate memoria
    python scripts/generate_synthetic_benchmarks.py --seed 2026 \
        --univariate-observations 1000000 --multivariate-observations 1000000 \
        --n-channels 6

Uso habitual::

    python scripts/benchmark_synthetic.py
    python scripts/benchmark_synthetic.py --kinds univariate --models Custom Persistence
    python scripts/benchmark_synthetic.py --models Custom EncDec-AR --model-sizes Small Medium Large
    python scripts/benchmark_synthetic.py --seeds 42 84 126
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from colorama import Fore, Style, init
from torch.utils.data import DataLoader, Dataset, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")])

from state_art.baselines_wrapper import MultiHorizonBaselineWrapper
from state_art.coformer.model import CompatibleTransformer
from state_art.simple_baselines import LastValueTimeMLP, PerTargetPersistenceModel
from state_art.strats.model import STraTSNetwork
from ts_transformer.data import (
    EventTimeSeriesDataset,
    SequenceBuilder,
    StandardScaler,
    TimeSeriesDataset,
    build_collate_fn,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.data.sequence_builder import AutoregressiveSequenceBuilder
from ts_transformer.models import TimeSeriesEncoderDecoder, TimeSeriesTransformer
from ts_transformer.training import Trainer
from ts_transformer.training.metrics import (
    compute_regression_metrics,
    compute_structured_regression_metrics,
)
from ts_transformer.utils import (
    get_logger,
    load_data_config,
    load_model_config,
    load_training_config,
    set_global_seed,
    setup_logging,
)


MODEL_NAMES = (
    "Custom",
    "Custom-TimeBias",
    "Custom-Time2Vec",
    "Custom-OrdinalTime",
    "Custom-NoRole",
    "EncDec-AR",
    "STraTS_Adapter",
    "CoFormer",
    "Persistence",
    "LastValueTimeMLP",
)
DEFAULT_MODEL_NAMES = tuple(
    model_name for model_name in MODEL_NAMES if model_name != "Custom-TimeBias"
)

MODEL_SIZE_PROFILES = {
    "Small": {"d_model": 32, "num_heads": 4, "num_layers": 2, "dim_feedforward": 128},
    "Medium": {"d_model": 64, "num_heads": 4, "num_layers": 2, "dim_feedforward": 256},
    "Large": {"d_model": 128, "num_heads": 8, "num_layers": 4, "dim_feedforward": 512},
}

OPTIMIZED_SIZE = "Optimized"
DEFAULT_RECIPES_CONFIG = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "synthetic_optuna_best.yaml"
)


@dataclass
class ModelRunSpec:
    model: torch.nn.Module
    trainable: bool
    autoregressive: bool = False
    training_family: str | None = None
    model_size: str = "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sobre datos sintéticos irregulares.")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--data-config", default="configs/data/synthetic_benchmark.yaml")
    parser.add_argument("--model-config", default="configs/model/synthetic_transformer.yaml")
    parser.add_argument("--training-config", default="configs/training/synthetic_benchmark.yaml")
    parser.add_argument(
        "--recipes-config",
        type=Path,
        default=DEFAULT_RECIPES_CONFIG,
        help="Recetas congeladas de Optuna para Custom y EncDec-AR.",
    )
    parser.add_argument("--exp-dir", type=Path, default=REPOSITORY_ROOT / "experiments" / "synthetic_benchmark")
    parser.add_argument("--kinds", nargs="+", choices=("univariate", "multivariate"), default=("univariate", "multivariate"))
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 84, 126))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        choices=(OPTIMIZED_SIZE, *MODEL_SIZE_PROFILES),
        default=(OPTIMIZED_SIZE,),
        help="Variantes de Custom y EncDec-AR (default: ganador Optimized de Optuna).",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--limit-datasets", type=int, default=None, help="Sólo para pruebas de humo.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida contratos de datos, entrenamiento y score sin iniciar el benchmark.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=2,
        help="Batch pequeño usado por --validate-only (default: 2).",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_time_scale(timestamps: np.ndarray) -> float:
    deltas = np.diff(timestamps.astype(np.float64, copy=False))
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(deltas) == 0:
        return 1.0
    return max(float(np.median(deltas)), 1e-8)


def model_name_for_size(family: str, size: str) -> str:
    """El ganador de Optuna conserva el nombre de familia; los perfiles llevan sufijo."""
    return family if size == OPTIMIZED_SIZE else f"{family}-{size}"


def model_size_from_name(model_name: str) -> str:
    for size in MODEL_SIZE_PROFILES:
        if model_name.endswith(f"-{size}"):
            return size
    if model_name in {"Custom", "Custom-TimeBias", "Custom-Time2Vec", "EncDec-AR"}:
        return OPTIMIZED_SIZE
    return "N/A"


def configure_model_size(config, size: str):
    sized_config = copy.deepcopy(config)
    for field_name, value in MODEL_SIZE_PROFILES[size].items():
        setattr(sized_config, field_name, value)
    return sized_config


def load_benchmark_recipes(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        recipes = yaml.safe_load(handle)
    if not isinstance(recipes, dict) or "families" not in recipes or "benchmark_task" not in recipes:
        raise ValueError(f"Configuración de recetas inválida: {path}")
    for family in ("Custom", "EncDec-AR"):
        if family not in recipes["families"]:
            raise ValueError(f"Falta la receta de {family} en {path}.")
    return recipes


def apply_model_recipe(config, recipes: dict[str, Any], family: str):
    configured = copy.deepcopy(config)
    for field_name, value in recipes["families"][family]["model"].items():
        setattr(configured, field_name, value)
    return configured


def configure_model_variant(config, recipes: dict[str, Any], family: str, size: str):
    if size == OPTIMIZED_SIZE:
        return apply_model_recipe(config, recipes, family)
    configured = configure_model_size(config, size)
    if family == "EncDec-AR":
        configured.use_causal_mask = True
        configured.decoder_num_layers = 1
    return configured


def configure_training_for_family(training_cfg, recipes: dict[str, Any], family: str | None):
    configured = copy.deepcopy(training_cfg)
    configured.optimizer_config.scheduler_T_max = configured.num_epochs
    if family is None:
        return configured
    optimizer = recipes["families"][family]["optimizer"]
    configured.optimizer_config.optimizer_name = optimizer["optimizer_name"]
    configured.optimizer_config.lr = float(optimizer["learning_rate"])
    configured.optimizer_config.weight_decay = float(optimizer["weight_decay"])
    configured.optimizer_config.warmup_epochs = int(optimizer["warmup_epochs"])
    configured.optimizer_config.betas = tuple(float(value) for value in optimizer["betas"])
    return configured


def apply_common_benchmark_task(data_cfg, recipes: dict[str, Any]):
    configured = copy.deepcopy(data_cfg)
    task = recipes["benchmark_task"]
    configured.history_length = int(task["history_length"])
    configured.min_history_length = None
    configured.target_offset_choices = [int(value) for value in task["target_offset_choices"]]
    configured.target_offset_min = None
    configured.target_offset_max = None
    configured.num_targets = int(task["num_targets"])
    return configured


def discover_datasets(data_root: Path, kinds: tuple[str, ...], presets: list[str] | None) -> list[tuple[str, str, Path]]:
    datasets: list[tuple[str, str, Path]] = []
    for kind in kinds:
        kind_root = data_root / kind
        if not kind_root.is_dir():
            raise FileNotFoundError(f"No existe el directorio de datos: {kind_root}")
        for preset_dir in sorted(path for path in kind_root.iterdir() if path.is_dir()):
            if presets is not None and preset_dir.name not in presets:
                continue
            for observations_path in sorted(preset_dir.glob("*/observations.parquet")):
                datasets.append((kind, preset_dir.name, observations_path))
    if not datasets:
        raise FileNotFoundError("No se encontraron archivos observations.parquet para los filtros solicitados.")
    return datasets


def read_observations(
    kind: str,
    observations_path: Path,
    *,
    include_clean: bool = False,
):
    available_columns = set(pq.ParquetFile(observations_path).schema.names)
    has_clean = "clean_value" in available_columns
    if kind == "univariate":
        columns = ["time", "value", "split", "event_index"]
        if include_clean and has_clean:
            columns.append("clean_value")
        frame = pd.read_parquet(observations_path, columns=columns)
        frame = frame.sort_values(["time", "event_index"], kind="stable")
        timestamps = frame["time"].to_numpy(dtype=np.float32)
        values = frame[["value"]].to_numpy(dtype=np.float32)
        splits = frame["split"].astype(str).to_numpy()
        result = (timestamps, values, splits, ["x00"])
        if include_clean:
            clean_values = (
                frame[["clean_value"]].to_numpy(dtype=np.float32)
                if has_clean else None
            )
            return (*result, clean_values)
        return result

    columns = ["time", "channel_index", "value", "split", "event_index"]
    if include_clean and has_clean:
        columns.append("clean_value")
    frame = pd.read_parquet(
        observations_path,
        columns=columns,
    ).sort_values(["time", "event_index"], kind="stable")
    split_counts = frame.groupby("time", observed=True)["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError(f"Un timestamp pertenece a más de un split en {observations_path}.")
    values_frame = frame.pivot(index="time", columns="channel_index", values="value").sort_index()
    values_frame = values_frame.reindex(columns=range(int(frame["channel_index"].max()) + 1))
    split_by_time = frame.drop_duplicates("time", keep="last").set_index("time")["split"]
    timestamps = values_frame.index.to_numpy(dtype=np.float32)
    values = values_frame.to_numpy(dtype=np.float32)
    splits = split_by_time.reindex(values_frame.index).astype(str).to_numpy()
    result = (
        timestamps,
        values,
        splits,
        [f"x{index:02d}" for index in range(values.shape[1])],
    )
    if include_clean:
        clean_values = None
        if has_clean:
            clean_frame = frame.pivot(
                index="time", columns="channel_index", values="clean_value"
            ).sort_index()
            clean_frame = clean_frame.reindex(
                index=values_frame.index, columns=values_frame.columns
            )
            clean_values = clean_frame.to_numpy(dtype=np.float32)
        return (*result, clean_values)
    return result


def split_indices(dataset: Dataset, splits: np.ndarray, split_name: str) -> list[int]:
    base = dataset
    if not isinstance(base, (TimeSeriesDataset, EventTimeSeriesDataset)):
        raise TypeError("El dataset debe ser TimeSeriesDataset o EventTimeSeriesDataset.")
    selected: list[int] = []
    for dataset_index, anchor in enumerate(base._example_indices):
        target_indices = [anchor + offset for offset in base.offsets]
        if np.all(splits[target_indices] == split_name):
            selected.append(dataset_index)
    if not selected:
        raise ValueError(f"No hay ejemplos completos para el split '{split_name}'.")
    return selected


def prepare_data(kind: str, observations_path: Path, data_cfg, num_workers_override: int | None, prefetch_factor: int) -> dict[str, Any]:
    timestamps, values, splits, channel_names, clean_values = read_observations(
        kind, observations_path, include_clean=True
    )
    return prepare_data_from_arrays(
        kind,
        timestamps,
        values,
        splits,
        channel_names,
        data_cfg,
        num_workers_override,
        prefetch_factor,
        clean_values=clean_values,
    )


def prepare_data_from_arrays(
    kind: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    splits: np.ndarray,
    channel_names: list[str],
    data_cfg,
    num_workers_override: int | None,
    prefetch_factor: int,
    clean_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Construye loaders desde observaciones ya cargadas en memoria."""
    train_rows = splits == "train"
    if not train_rows.any():
        raise ValueError("El dataset no contiene filas de entrenamiento.")

    scaler = StandardScaler().fit(values[train_rows])
    scaled_values = scaler.transform(values).astype(np.float32, copy=False)
    scaled_clean_values = (
        scaler.transform(clean_values).astype(np.float32, copy=False)
        if clean_values is not None else None
    )
    time_scale = estimate_time_scale(timestamps[train_rows])
    dataset_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length,
        target_offset_choices=data_cfg.target_offset_choices,
        target_offset_min=data_cfg.target_offset_min,
        target_offset_max=data_cfg.target_offset_max,
        target_offset=data_cfg.target_offset,
        stride=data_cfg.stride,
        num_targets=data_cfg.num_targets,
    )
    use_events = kind == "multivariate"
    input_dim = values.shape[1]
    if use_events:
        builder = SequenceBuilder(
            input_dim=1,
            target_token_value="zeros",
            use_sensor_ids=True,
            num_sensors=input_dim,
            num_target_tokens=input_dim,
            target_sensor_ids=list(range(input_dim)),
        )
        dataset = EventTimeSeriesDataset(
            scaled_values, timestamps, scaled_values, dataset_cfg, input_dim, input_dim,
            sequence_builder=builder,
        )
        model_input_dim = 1
    else:
        builder = SequenceBuilder(input_dim=input_dim, target_token_value="zeros")
        dataset = TimeSeriesDataset(
            scaled_values, timestamps, dataset_cfg, input_dim, input_dim,
            targets=scaled_values, sequence_builder=builder,
        )
        model_input_dim = input_dim

    clean_dataset = None
    if scaled_clean_values is not None:
        if use_events:
            clean_builder = SequenceBuilder(
                input_dim=1,
                target_token_value="zeros",
                use_sensor_ids=True,
                num_sensors=input_dim,
                num_target_tokens=input_dim,
                target_sensor_ids=list(range(input_dim)),
            )
            clean_dataset = EventTimeSeriesDataset(
                scaled_values,
                timestamps,
                scaled_clean_values,
                dataset_cfg,
                input_dim,
                input_dim,
                sequence_builder=clean_builder,
            )
        else:
            clean_builder = SequenceBuilder(
                input_dim=input_dim, target_token_value="zeros"
            )
            clean_dataset = TimeSeriesDataset(
                scaled_values,
                timestamps,
                dataset_cfg,
                input_dim,
                input_dim,
                targets=scaled_clean_values,
                sequence_builder=clean_builder,
            )

    num_workers = data_cfg.num_workers if num_workers_override is None else num_workers_override
    loader_kwargs: dict[str, Any] = {
        "batch_size": data_cfg.batch_size,
        "collate_fn": build_collate_fn(pad_to_max_length=True),
        "num_workers": max(0, int(num_workers)),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    subset_indices = {
        split_name: split_indices(dataset, splits, split_name)
        for split_name in ("train", "validation", "test")
    }
    subsets = {
        split_name: Subset(dataset, indices)
        for split_name, indices in subset_indices.items()
    }
    prepared = {
        "train_loader": DataLoader(subsets["train"], shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(subsets["validation"], shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(subsets["test"], shuffle=False, **loader_kwargs),
        "input_dim": input_dim,
        "model_input_dim": model_input_dim,
        "output_dim": input_dim,
        "use_events": use_events,
        "time_scale": time_scale,
        "counts": {name: len(subset) for name, subset in subsets.items()},
        "channel_names": channel_names,
        "has_clean_targets": clean_dataset is not None,
    }
    if clean_dataset is not None:
        prepared["clean_test_loader"] = DataLoader(
            Subset(clean_dataset, subset_indices["test"]),
            shuffle=False,
            **loader_kwargs,
        )
    return prepared


def add_autoregressive_train_loader(data: dict[str, Any]) -> None:
    """Añade teacher forcing al loader de entrenamiento sin alterar val/test."""
    train_subset = data["train_loader"].dataset
    base_dataset = train_subset.dataset
    if not isinstance(base_dataset, (TimeSeriesDataset, EventTimeSeriesDataset)):
        raise TypeError("El subset de entrenamiento no contiene un dataset de series temporales.")

    if data["use_events"]:
        builder = AutoregressiveSequenceBuilder(
            input_dim=1,
            target_token_value="zeros",
            use_sensor_ids=True,
            num_sensors=data["input_dim"],
            num_target_tokens=data["output_dim"],
            target_sensor_ids=list(range(data["output_dim"])),
        )
        ar_dataset = EventTimeSeriesDataset(
            base_dataset.values,
            base_dataset.timestamps,
            base_dataset.targets,
            copy.deepcopy(base_dataset.config),
            data["input_dim"],
            data["output_dim"],
            sequence_builder=builder,
        )
    else:
        builder = AutoregressiveSequenceBuilder(
            input_dim=data["input_dim"], target_token_value="zeros"
        )
        ar_dataset = TimeSeriesDataset(
            base_dataset.values,
            base_dataset.timestamps,
            copy.deepcopy(base_dataset.config),
            data["input_dim"],
            data["output_dim"],
            targets=base_dataset.targets,
            sequence_builder=builder,
        )

    base_loader = data["train_loader"]
    loader_kwargs: dict[str, Any] = {
        "batch_size": base_loader.batch_size,
        "shuffle": True,
        "collate_fn": base_loader.collate_fn,
        "num_workers": base_loader.num_workers,
        "pin_memory": base_loader.pin_memory,
    }
    if base_loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = base_loader.persistent_workers
        if base_loader.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = base_loader.prefetch_factor
    data["ar_train_loader"] = DataLoader(
        Subset(ar_dataset, train_subset.indices), **loader_kwargs
    )


class AutoregressiveTrainer(Trainer):
    """Trainer cuya validación y test usan generación recursiva, no teacher forcing."""

    @torch.inference_mode()
    def _evaluate_generated(self, loader: DataLoader, prefix: str) -> dict[str, float]:
        self.model.eval()
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        for batch in loader:
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            input_timestamps = batch["input_timestamps"].to(self.device, non_blocking=True)
            target_timestamps = batch["target_timestamps"].to(self.device, non_blocking=True)
            target_values = batch["target_values"].to(self.device, non_blocking=True)
            target_tokens = int(batch["is_target_mask"][0].sum().item())
            history_length = input_values.shape[1] - target_tokens
            input_sensor_ids = batch.get("input_sensor_ids")
            if input_sensor_ids is not None:
                input_sensor_ids = input_sensor_ids.to(self.device, non_blocking=True)
            padding_mask = batch.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device, non_blocking=True)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)

            predictions = self.model.generate(
                history_values=input_values[:, :history_length],
                history_timestamps=input_timestamps[:, :history_length],
                target_timestamps=target_timestamps,
                history_sensor_ids=(
                    input_sensor_ids[:, :history_length]
                    if input_sensor_ids is not None else None
                ),
                target_sensor_ids=(
                    input_sensor_ids[:, -target_tokens:]
                    if input_sensor_ids is not None else None
                ),
                history_padding_mask=(
                    padding_mask[:, :history_length]
                    if padding_mask is not None else None
                ),
                history_lengths=(lengths - target_tokens if lengths is not None else None),
            )
            all_predictions.append(predictions.cpu())
            all_targets.append(target_values.cpu())
            target_loss_mask = batch.get("target_loss_mask")
            if target_loss_mask is not None:
                all_masks.append(target_loss_mask.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0) if all_masks else None
        loss = self._compute_loss(predictions, targets, masks).item()
        if masks is not None and torch.any(masks > 0):
            valid = masks > 0
            predictions = predictions[valid].view(-1, 1)
            targets = targets[valid].view(-1, 1)
        metrics = compute_regression_metrics(predictions, targets, prefix=prefix)
        metrics.update(
            compute_structured_regression_metrics(
                torch.cat(all_predictions, dim=0),
                torch.cat(all_targets, dim=0),
                masks,
                prefix=prefix,
            )
        )
        metrics[f"{prefix}loss"] = loss
        return metrics

    @torch.inference_mode()
    def _evaluate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        metrics = self._evaluate_generated(self.val_loader, "val_")
        print(f"[Epoch {epoch:03d}] " + ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
        return metrics

    @torch.inference_mode()
    def evaluate_on_loader(self, loader: DataLoader, prefix: str = "test_") -> dict[str, float]:
        return self._evaluate_generated(loader, prefix)


def build_models(
    model_cfg,
    data: dict[str, Any],
    model_sizes: tuple[str, ...],
    recipes: dict[str, Any],
    model_seed: int | None = None,
) -> dict[str, ModelRunSpec]:
    config = copy.deepcopy(model_cfg)
    config.input_dim = data["model_input_dim"]
    config.output_dim = data["output_dim"]
    config.use_sensor_embedding = data["use_events"]
    config.num_sensors = data["input_dim"] if data["use_events"] else 0
    config.time_scale = data["time_scale"]
    def reset_initialization_seed() -> None:
        if model_seed is not None:
            set_global_seed(model_seed, deterministic=False)

    models: dict[str, ModelRunSpec] = {}
    reset_initialization_seed()
    models["Persistence"] = ModelRunSpec(
        PerTargetPersistenceModel(data["model_input_dim"], data["output_dim"]),
        trainable=False,
    )
    reset_initialization_seed()
    models["LastValueTimeMLP"] = ModelRunSpec(
        LastValueTimeMLP(
            data["model_input_dim"], data["output_dim"],
            num_sensors=data["input_dim"], time_scale=data["time_scale"],
        ),
        trainable=True,
    )

    for size in model_sizes:
        reset_initialization_seed()
        custom_config = configure_model_variant(config, recipes, "Custom", size)
        models[model_name_for_size("Custom", size)] = ModelRunSpec(
            TimeSeriesTransformer(custom_config),
            trainable=True,
            training_family="Custom" if size == OPTIMIZED_SIZE else None,
            model_size=size,
        )

    reset_initialization_seed()
    time2vec_config = apply_model_recipe(config, recipes, "Custom")
    time2vec_config.time_encoding_mode = "time2vec"
    models["Custom-Time2Vec"] = ModelRunSpec(
        TimeSeriesTransformer(time2vec_config),
        trainable=True,
        training_family="Custom",
        model_size=OPTIMIZED_SIZE,
    )
    reset_initialization_seed()
    time_bias_config = apply_model_recipe(config, recipes, "Custom")
    time_bias_config.use_temporal_attn_bias = True
    time_bias_config.temporal_bias_layers = 1
    models["Custom-TimeBias"] = ModelRunSpec(
        TimeSeriesTransformer(time_bias_config),
        trainable=True,
        training_family="Custom",
        model_size=OPTIMIZED_SIZE,
    )
    reset_initialization_seed()
    ordinal_config = apply_model_recipe(config, recipes, "Custom")
    ordinal_config.time_encoding_mode = "ordinal"
    ordinal_config.time_transform = "linear"
    models["Custom-OrdinalTime"] = ModelRunSpec(
        TimeSeriesTransformer(ordinal_config),
        trainable=True,
        training_family="Custom",
        model_size=OPTIMIZED_SIZE,
    )
    reset_initialization_seed()
    no_role_config = apply_model_recipe(config, recipes, "Custom")
    no_role_config.use_target_flag_embedding = False
    models["Custom-NoRole"] = ModelRunSpec(
        TimeSeriesTransformer(no_role_config),
        trainable=True,
        training_family="Custom",
        model_size=OPTIMIZED_SIZE,
    )
    for size in model_sizes:
        reset_initialization_seed()
        ar_config = configure_model_variant(config, recipes, "EncDec-AR", size)
        models[model_name_for_size("EncDec-AR", size)] = ModelRunSpec(
            TimeSeriesEncoderDecoder(ar_config),
            trainable=True,
            autoregressive=True,
            training_family="EncDec-AR" if size == OPTIMIZED_SIZE else None,
            model_size=size,
        )

    reset_initialization_seed()
    strats = STraTSNetwork(
        num_features=data["input_dim"] + 1,
        d_model=config.d_model,
        num_classes=data["output_dim"],
    )
    models["STraTS_Adapter"] = ModelRunSpec(
        MultiHorizonBaselineWrapper(
            strats, "strats", config.d_model, data["output_dim"],
            use_sensor_embedding=data["use_events"], time_scale=data["time_scale"],
        ),
        trainable=True,
    )
    n_variates = data["input_dim"] if data["use_events"] else 1
    reset_initialization_seed()
    coformer = CompatibleTransformer(
        num_variates=n_variates,
        d_model=config.d_model,
        n_heads=config.num_heads,
        n_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=data["output_dim"],
    )
    models["CoFormer"] = ModelRunSpec(
        MultiHorizonBaselineWrapper(
            coformer, "coformer", config.d_model, data["output_dim"],
            use_sensor_embedding=data["use_events"], time_scale=data["time_scale"],
        ),
        trainable=True,
    )
    return models


def selected_model_variants(
    models: dict[str, ModelRunSpec],
    requested_families: tuple[str, ...] | None,
) -> dict[str, ModelRunSpec]:
    if requested_families is None:
        return models

    selected: dict[str, ModelRunSpec] = {}
    for name, item in models.items():
        family = name
        for size in MODEL_SIZE_PROFILES:
            suffix = f"-{size}"
            if name.endswith(suffix):
                family = name.removesuffix(suffix)
                break
        if family in requested_families:
            selected[name] = item
    return selected


def train_and_evaluate(model: torch.nn.Module, trainable: bool, data: dict[str, Any], training_cfg, checkpoint_dir: Path, autoregressive: bool = False) -> dict[str, Any]:
    config = copy.deepcopy(training_cfg)
    config.checkpoint_dir = str(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_type = AutoregressiveTrainer if autoregressive else Trainer
    train_loader = data["ar_train_loader"] if autoregressive else data["train_loader"]
    trainer = trainer_type(model=model, train_loader=train_loader, val_loader=data["val_loader"], config=config)
    started = time.perf_counter()
    history = trainer.fit() if trainable else {}
    elapsed = time.perf_counter() - started
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if trainable and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    metrics = trainer.evaluate_on_loader(data["val_loader"], prefix="val_")
    metrics.update(trainer.evaluate_on_loader(data["test_loader"], prefix="test_"))
    if "clean_test_loader" in data:
        metrics.update(
            trainer.evaluate_on_loader(data["clean_test_loader"], prefix="test_clean_")
        )
    metrics.update({
        "train_time_s": round(elapsed, 2),
        "epochs_run": len(history.get("train_loss", [])),
        "n_params_trainable": count_parameters(model),
    })
    return metrics


def validate_model_contract(
    model_name: str,
    spec: ModelRunSpec,
    data: dict[str, Any],
    training_cfg,
    recipes: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta un paso de entrenamiento y un score de validación sobre un batch."""
    config = configure_training_for_family(
        training_cfg, recipes, spec.training_family
    )
    config.checkpoint_dir = None
    config.log_every_n_steps = 0
    config.use_amp = False
    trainer_type = AutoregressiveTrainer if spec.autoregressive else Trainer
    train_loader = data["ar_train_loader"] if spec.autoregressive else data["train_loader"]
    trainer = trainer_type(
        model=spec.model,
        train_loader=train_loader,
        val_loader=data["val_loader"],
        config=config,
    )

    train_loss = None
    if spec.trainable:
        train_loss = float(trainer._train_step(next(iter(train_loader))))
        if not np.isfinite(train_loss):
            raise ValueError(f"{model_name}: pérdida de entrenamiento no finita.")

    validation_batch = next(iter(data["val_loader"]))
    metrics = trainer.evaluate_on_loader([validation_batch], prefix="val_")
    required_metrics = ("val_mse", "val_rmse", "val_mae", "val_loss")
    if any(name not in metrics or not np.isfinite(metrics[name]) for name in required_metrics):
        raise ValueError(f"{model_name}: métricas faltantes o no finitas: {metrics}")
    if not np.isclose(metrics["val_rmse"] ** 2, metrics["val_mse"], rtol=1e-5, atol=1e-7):
        raise ValueError(
            f"{model_name}: RMSE^2 != MSE ({metrics['val_rmse']} vs {metrics['val_mse']})."
        )

    target_mask = validation_batch.get("target_loss_mask")
    n_valid_targets = (
        int((target_mask > 0).sum().item())
        if target_mask is not None
        else int(validation_batch["target_values"].numel())
    )
    return {
        "Model": model_name,
        "Model_Size": spec.model_size,
        "Autoregressive": spec.autoregressive,
        "Training_Recipe": spec.training_family or "base",
        "Train_Loss": train_loss,
        "val_mse": metrics["val_mse"],
        "val_rmse": metrics["val_rmse"],
        "val_mae": metrics["val_mae"],
        "n_valid_targets": n_valid_targets,
        "target_shape": str(tuple(validation_batch["target_values"].shape)),
        "n_params_trainable": count_parameters(spec.model),
        "Status": "PASS",
    }


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def main() -> None:
    args = parse_args()
    init(autoreset=True)
    setup_logging()
    logger = get_logger("benchmark_synthetic")
    recipes = load_benchmark_recipes(args.recipes_config)
    data_cfg = apply_common_benchmark_task(load_data_config(args.data_config), recipes)
    if args.validate_only:
        if args.validation_batch_size <= 0:
            raise ValueError("--validation-batch-size debe ser > 0.")
        data_cfg.batch_size = args.validation_batch_size
        data_cfg.num_workers = 0
    model_cfg = load_model_config(args.model_config)
    training_cfg, _ = load_training_config(args.training_config)
    if not torch.cuda.is_available():
        training_cfg.device = "cpu"

    datasets = discover_datasets(args.data_root, tuple(args.kinds), args.presets)
    if args.limit_datasets is not None:
        datasets = datasets[:args.limit_datasets]
    requested_models = (
        tuple(args.models) if args.models is not None else DEFAULT_MODEL_NAMES
    )
    selected_sizes = tuple(args.model_sizes)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arguments": serializable_arguments(args),
        "benchmark_task": recipes["benchmark_task"],
        "frozen_recipes": recipes["families"],
    }
    manifest_name = "preflight_manifest.json" if args.validate_only else "run_manifest.json"
    (args.exp_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.validate_only:
        validation_path = args.exp_dir / "model_contract_validation.csv"
        validation_records: list[dict[str, Any]] = []
    else:
        validation_path = None
        validation_records = []
    failed_runs: list[dict[str, Any]] = []
    failed_runs_path = args.exp_dir / "failed_runs.csv"
    if not args.validate_only:
        pd.DataFrame(
            columns=("Kind", "Preset", "Dataset_ID", "Seed", "Model", "Error")
        ).to_csv(failed_runs_path, index=False)
    output_path = args.exp_dir / "benchmark_synthetic.csv"
    existing = (
        pd.read_csv(output_path)
        if output_path.exists() and not args.validate_only
        else pd.DataFrame()
    )
    records: list[dict[str, Any]] = existing.to_dict("records")
    completed = {
        (str(row["Kind"]), str(row["Dataset_ID"]), int(row["Seed"]), str(row["Model"]))
        for row in records
    } if not existing.empty else set()
    logger.info(Fore.GREEN + "=== SYNTHETIC IRREGULAR BENCHMARK ===" + Style.RESET_ALL)
    logger.info(
        "Datasets=%d, seeds=%s, model families=%s, sizes=%s",
        len(datasets), args.seeds,
        requested_models,
        selected_sizes,
    )

    for kind, preset, observations_path in datasets:
        dataset_id = observations_path.parent.name
        logger.info(Fore.CYAN + "[%s | %s]" + Style.RESET_ALL, kind, dataset_id)
        data = prepare_data(kind, observations_path, data_cfg, args.num_workers, args.prefetch_factor)
        if "EncDec-AR" in requested_models:
            add_autoregressive_train_loader(data)
        logger.info("samples: train=%d val=%d test=%d, channels=%d, time_scale=%.8f", data["counts"]["train"], data["counts"]["validation"], data["counts"]["test"], data["input_dim"], data["time_scale"])
        if args.validate_only:
            set_global_seed(args.seeds[0], deterministic=False)
            models = selected_model_variants(
                build_models(
                    model_cfg, data, selected_sizes, recipes, model_seed=args.seeds[0]
                ),
                requested_models,
            )
            for model_name in list(models):
                spec = models.pop(model_name)
                set_global_seed(args.seeds[0], deterministic=False)
                logger.info("  validating %s", model_name)
                base_record = {
                    "Kind": kind,
                    "Preset": preset,
                    "Dataset_ID": dataset_id,
                    "n_channels": data["input_dim"],
                }
                try:
                    record = validate_model_contract(
                        model_name, spec, data, training_cfg, recipes
                    )
                    record.update(base_record)
                    logger.info(Fore.GREEN + "  [PASS] %s" + Style.RESET_ALL, model_name)
                except Exception as exc:
                    logger.exception("  [FAIL] %s %s", kind, model_name)
                    record = {
                        **base_record,
                        "Model": model_name,
                        "Model_Size": spec.model_size,
                        "Autoregressive": spec.autoregressive,
                        "Training_Recipe": spec.training_family or "base",
                        "Status": "FAIL",
                        "Error": f"{type(exc).__name__}: {exc}",
                    }
                validation_records.append(record)
                pd.DataFrame(validation_records).to_csv(validation_path, index=False)
                del spec.model
                del spec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        for seed in args.seeds:
            set_global_seed(seed, deterministic=False)
            models = selected_model_variants(
                build_models(model_cfg, data, selected_sizes, recipes, model_seed=seed),
                requested_models,
            )
            for model_name in list(models):
                spec = models.pop(model_name)
                model = spec.model
                run_key = (kind, dataset_id, seed, model_name)
                if run_key in completed:
                    logger.info("  [SKIP] %s seed=%d", model_name, seed)
                    del model, spec
                    continue
                checkpoint_dir = args.exp_dir / kind / dataset_id / f"seed_{seed}" / model_name
                logger.info("  training %s seed=%d (%d params)", model_name, seed, count_parameters(model))
                try:
                    set_global_seed(seed, deterministic=False)
                    model_training_cfg = configure_training_for_family(
                        training_cfg, recipes, spec.training_family
                    )
                    record = train_and_evaluate(
                        model, spec.trainable, data, model_training_cfg, checkpoint_dir,
                        autoregressive=spec.autoregressive,
                    )
                    record.update({
                        "Kind": kind,
                        "Preset": preset,
                        "Dataset_ID": dataset_id,
                        "Seed": seed,
                        "Model": model_name,
                        "Model_Size": spec.model_size,
                        "Training_Recipe": spec.training_family or "base",
                        "n_channels": data["input_dim"],
                        "n_train": data["counts"]["train"],
                        "n_val": data["counts"]["validation"],
                        "n_test": data["counts"]["test"],
                        "History_Length": data_cfg.history_length,
                        "Target_Offsets": json.dumps(data_cfg.target_offset_choices),
                        "Score_Name": recipes["benchmark_task"]["score_name"],
                        "Score_Space": recipes["benchmark_task"]["score_space"],
                        "Score_Value": record[recipes["benchmark_task"]["score_name"]],
                    })
                    records.append(record)
                    pd.DataFrame(records).to_csv(output_path, index=False)
                    completed.add(run_key)
                    logger.info(Fore.GREEN + "  [OK] %s mse=%0.6f" + Style.RESET_ALL, model_name, record["test_mse"])
                except Exception as exc:
                    logger.exception("  [ERROR] %s %s seed=%d", kind, model_name, seed)
                    failed_runs.append({
                        "Kind": kind,
                        "Preset": preset,
                        "Dataset_ID": dataset_id,
                        "Seed": seed,
                        "Model": model_name,
                        "Error": f"{type(exc).__name__}: {exc}",
                    })
                    pd.DataFrame(failed_runs).to_csv(failed_runs_path, index=False)
                finally:
                    del model, spec
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if args.validate_only:
        validation_frame = pd.DataFrame(validation_records)
        failures = validation_frame[validation_frame["Status"] != "PASS"]
        if not failures.empty:
            raise RuntimeError(
                f"Preflight falló en {len(failures)} combinaciones; revisa {validation_path}."
            )
        logger.info(
            Fore.GREEN + "Preflight completado: %d combinaciones válidas. Reporte: %s" + Style.RESET_ALL,
            len(validation_frame), validation_path,
        )
        return

    results = pd.DataFrame(records)
    if results.empty:
        if failed_runs:
            raise RuntimeError(
                f"Fallaron {len(failed_runs)} corridas; revisa {failed_runs_path}."
            )
        return
    results = results.drop_duplicates(["Kind", "Dataset_ID", "Seed", "Model"], keep="last")
    results.to_csv(output_path, index=False)
    summary = results.groupby(["Kind", "Preset", "Model"])[["test_mse", "test_rmse", "test_mae", "train_time_s", "n_params_trainable"]].mean().reset_index()
    summary.to_csv(args.exp_dir / "summary_by_kind_preset.csv", index=False)
    logger.info(Fore.GREEN + "Completed. Results: %s" + Style.RESET_ALL, output_path)
    if failed_runs:
        raise RuntimeError(
            f"Fallaron {len(failed_runs)} corridas; el benchmark se puede reanudar. "
            f"Revisa {failed_runs_path}."
        )


if __name__ == "__main__":
    main()
