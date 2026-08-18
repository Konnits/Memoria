"""Búsqueda Optuna reproducible para Custom y EncDec-AR en datos sintéticos.

Usa sólo validación sobre seis escenarios fijos; el split test queda reservado
para el benchmark final. Por defecto ejecuta 250 trials por familia (500 total)
y puede reanudarse desde la base SQLite creada en el directorio de salida.

Ejemplo::

    conda activate memoria
    python scripts/tune_synthetic_optuna.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.benchmark_synthetic import (
    AutoregressiveTrainer,
    add_autoregressive_train_loader,
    prepare_data_from_arrays,
    read_observations,
)
from ts_transformer.models import (
    TimeSeriesEncoderDecoder,
    TimeSeriesQueryCrossAttention,
    TimeSeriesTransformer,
)
from ts_transformer.training import Trainer
from ts_transformer.utils import (
    load_data_config,
    load_model_config,
    load_training_config,
    set_global_seed,
)


FAMILIES = ("Custom", "Custom-QueryCross", "EncDec-AR")
DEFAULT_DATASETS = (
    "univariate:bursty",
    "univariate:long_gaps",
    "univariate:hard_mixed",
    "multivariate:bursty",
    "multivariate:informative",
    "multivariate:hard_mixed",
)
ARCHITECTURES = {
    "small_2h": (32, 2),
    "small_4h": (32, 4),
    "medium_4h": (64, 4),
    "medium_8h": (64, 8),
    "wide_4h": (96, 4),
    "wide_8h": (96, 8),
    "large_8h": (128, 8),
}
HORIZON_PROFILES = {
    "short_2": (1, 16),
    "standard_4": (1, 4, 16, 64),
    "extended_8": (1, 2, 4, 8, 16, 32, 64, 128),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca hiperparámetros con Optuna sin usar el split test."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-config", default="configs/data/synthetic_benchmark.yaml")
    parser.add_argument("--model-config", default="configs/model/synthetic_transformer.yaml")
    parser.add_argument("--training-config", default="configs/training/synthetic_benchmark.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/optuna_synthetic_fixed_task"),
    )
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Pares kind:preset. Default: seis escenarios representativos.",
    )
    parser.add_argument(
        "--trials-per-family", type=int, default=250,
        help="Total acumulado por familia al reanudar (default: 250).",
    )
    parser.add_argument("--tuning-epochs", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--horizon-profile",
        choices=tuple(HORIZON_PROFILES),
        default="standard_4",
        help="Tarea fija para todo el estudio; nunca se samplea como hiperparámetro.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=512,
        help="Longitud histórica fija compartida por todos los trials.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_dataset_specs(values: tuple[str, ...] | list[str]) -> tuple[tuple[str, str], ...]:
    specs: list[tuple[str, str]] = []
    for value in values:
        try:
            kind, preset = value.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Dataset inválido {value!r}; usa kind:preset.") from exc
        if kind not in {"univariate", "multivariate"} or not preset:
            raise ValueError(f"Dataset inválido {value!r}; usa kind:preset.")
        specs.append((kind, preset))
    return tuple(specs)


def observations_path(data_root: Path, kind: str, preset: str) -> Path:
    candidates = sorted((data_root / kind / preset).glob("*/observations.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Se esperaba un observations.parquet para {kind}:{preset}; encontrados={candidates}"
        )
    return candidates[0]


def evenly_spaced_indices(indices: list[int], max_samples: int) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=max_samples, dtype=np.int64)
    return [indices[int(position)] for position in positions]


def limit_loader(loader: DataLoader, max_samples: int, batch_size: int, shuffle: bool) -> DataLoader:
    if not isinstance(loader.dataset, Subset):
        raise TypeError("El loader de tuning debe contener un torch.utils.data.Subset.")
    subset = loader.dataset
    indices = evenly_spaced_indices(list(subset.indices), max_samples)
    return DataLoader(
        Subset(subset.dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=loader.collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def sample_trial_config(
    trial: optuna.Trial,
    family: str,
    base_model_cfg,
    base_training_cfg,
    base_data_cfg,
    *,
    horizon_profile: str = "standard_4",
    history_length: int = 512,
):
    model_cfg = copy.deepcopy(base_model_cfg)
    training_cfg = copy.deepcopy(base_training_cfg)
    data_cfg = copy.deepcopy(base_data_cfg)

    architecture = trial.suggest_categorical("architecture", tuple(ARCHITECTURES))
    d_model, num_heads = ARCHITECTURES[architecture]
    model_cfg.d_model = d_model
    model_cfg.num_heads = num_heads
    model_cfg.num_layers = trial.suggest_int("encoder_layers", 2, 4)
    model_cfg.dim_feedforward = d_model * trial.suggest_categorical("ffn_multiplier", (2, 4, 6))
    model_cfg.dropout = trial.suggest_categorical("dropout", (0.0, 0.05, 0.1, 0.15, 0.2))
    model_cfg.time_encoding_mode = trial.suggest_categorical(
        "time_encoding_mode", ("sinusoidal", "time2vec")
    )
    model_cfg.time_transform = trial.suggest_categorical("time_transform", ("linear", "log1p"))

    if horizon_profile not in HORIZON_PROFILES:
        raise ValueError(f"horizon_profile desconocido: {horizon_profile}")
    if history_length <= 0:
        raise ValueError("history_length debe ser > 0.")
    offsets = HORIZON_PROFILES[horizon_profile]
    data_cfg.target_offset_choices = list(offsets)
    data_cfg.target_offset_min = None
    data_cfg.target_offset_max = None
    data_cfg.num_targets = len(offsets)
    data_cfg.history_length = int(history_length)
    data_cfg.min_history_length = None

    training_cfg.optimizer_config.lr = trial.suggest_float("learning_rate", 5e-5, 7e-4, log=True)
    training_cfg.optimizer_config.weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-2, log=True
    )
    training_cfg.optimizer_config.warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
    training_cfg.optimizer_config.scheduler_T_max = training_cfg.num_epochs
    training_cfg.optimizer_config.optimizer_name = "adamw"
    training_cfg.optimizer_config.betas = (0.9, 0.95)

    if family == "EncDec-AR":
        model_cfg.use_causal_mask = True
        model_cfg.decoder_num_layers = trial.suggest_int("decoder_layers", 1, 3)
    return model_cfg, training_cfg, data_cfg


def instantiate_model(family: str, model_cfg, data: dict[str, Any]) -> torch.nn.Module:
    config = copy.deepcopy(model_cfg)
    config.input_dim = data["model_input_dim"]
    config.output_dim = data["output_dim"]
    config.use_sensor_embedding = data["use_events"]
    config.num_sensors = data["input_dim"] if data["use_events"] else 0
    config.time_scale = data["time_scale"]
    if family == "Custom":
        return TimeSeriesTransformer(config)
    if family == "Custom-QueryCross":
        return TimeSeriesQueryCrossAttention(config)
    return TimeSeriesEncoderDecoder(config)


class SyntheticObjective:
    def __init__(
        self,
        family: str,
        raw_datasets: list[tuple[str, str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]],
        base_model_cfg,
        base_training_cfg,
        base_data_cfg,
        args: argparse.Namespace,
    ) -> None:
        self.family = family
        self.raw_datasets = raw_datasets
        self.base_model_cfg = base_model_cfg
        self.base_training_cfg = base_training_cfg
        self.base_data_cfg = base_data_cfg
        self.args = args

    def __call__(self, trial: optuna.Trial) -> float:
        model_cfg, training_cfg, data_cfg = sample_trial_config(
            trial,
            self.family,
            self.base_model_cfg,
            self.base_training_cfg,
            self.base_data_cfg,
            horizon_profile=self.args.horizon_profile,
            history_length=self.args.history_length,
        )
        trial.set_user_attr("fixed_horizon_profile", self.args.horizon_profile)
        trial.set_user_attr("fixed_history_length", int(self.args.history_length))
        training_cfg.num_epochs = self.args.tuning_epochs
        training_cfg.early_stopping_patience = self.args.early_stopping_patience
        training_cfg.early_stopping_min_delta = 1e-4
        training_cfg.restore_best_weights = True
        training_cfg.checkpoint_dir = None
        training_cfg.log_every_n_steps = 0
        training_cfg.save_best_on = "val_rmse"
        training_cfg.optimizer_config.scheduler_T_max = training_cfg.num_epochs
        if not torch.cuda.is_available():
            training_cfg.device = "cpu"

        scores: list[float] = []
        for dataset_index, (kind, preset, raw) in enumerate(self.raw_datasets):
            set_global_seed(self.args.seed + dataset_index, deterministic=False)
            data = prepare_data_from_arrays(
                kind, *raw, data_cfg, num_workers_override=0, prefetch_factor=0
            )
            if self.family == "EncDec-AR":
                add_autoregressive_train_loader(data)
                train_loader = limit_loader(
                    data["ar_train_loader"], self.args.max_train_samples,
                    self.args.batch_size, shuffle=True,
                )
                trainer_type = AutoregressiveTrainer
            else:
                train_loader = limit_loader(
                    data["train_loader"], self.args.max_train_samples,
                    self.args.batch_size, shuffle=True,
                )
                trainer_type = Trainer
            val_loader = limit_loader(
                data["val_loader"], self.args.max_val_samples,
                self.args.batch_size, shuffle=False,
            )
            model = instantiate_model(self.family, model_cfg, data)
            if dataset_index == 0:
                trial.set_user_attr(
                    "n_parameters",
                    sum(parameter.numel() for parameter in model.parameters()),
                )
            trainer = trainer_type(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=copy.deepcopy(training_cfg),
            )
            try:
                trainer.fit()
                if trainer._best_model_state_in_memory is not None:
                    trainer.model.load_state_dict(trainer._best_model_state_in_memory)
                score = float(trainer.evaluate_on_loader(val_loader, prefix="val_")["val_rmse"])
                scores.append(score)
                trial.set_user_attr(f"{kind}_{preset}_val_rmse", score)
                running_mean = float(np.mean(scores))
                trial.report(running_mean, step=dataset_index + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Podado tras {kind}:{preset}; mean val_rmse={running_mean:.6f}"
                    )
            finally:
                del trainer, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        score = float(np.mean(scores))
        trial.set_user_attr("mean_val_rmse", score)
        return score


def save_study_artifacts(study: optuna.Study, output_dir: Path, family: str) -> None:
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    frame.to_csv(output_dir / f"{family.lower()}_trials.csv", index=False)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    payload = {
        "family": family,
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_val_rmse": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    (output_dir / f"best_{family.lower()}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def print_trial_progress(family: str):
    """Crea un callback que resume el mejor trial acumulado tras cada iteración."""

    def callback(study: optuna.Study, finished_trial: optuna.trial.FrozenTrial) -> None:
        state = finished_trial.state.name
        score = "N/A" if finished_trial.value is None else f"{finished_trial.value:.6f}"
        completed = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        print(f"[{family}] trial={finished_trial.number} state={state} val_rmse={score}")
        if not completed:
            print(f"[{family}] todavía no hay trials completos.")
            return

        best = study.best_trial
        n_parameters = best.user_attrs.get("n_parameters", "N/A")
        print(
            f"[{family}] mejor acumulado: trial={best.number} "
            f"val_rmse={best.value:.6f} params={n_parameters}"
        )
        print(
            f"[{family}] mejores hiperparámetros: "
            f"{json.dumps(best.params, sort_keys=True)}"
        )

    return callback


def main() -> None:
    args = parse_args()
    if args.trials_per_family <= 0 or args.tuning_epochs <= 0 or args.history_length <= 0:
        raise ValueError("--trials-per-family y --tuning-epochs deben ser > 0.")
    specs = parse_dataset_specs(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(args.output_dir / 'optuna_studies.db').resolve().as_posix()}"
    base_data_cfg = load_data_config(args.data_config)
    base_model_cfg = load_model_config(args.model_config)
    base_training_cfg, _ = load_training_config(args.training_config)

    raw_datasets = []
    for kind, preset in specs:
        path = observations_path(args.data_root, kind, preset)
        raw_datasets.append((kind, preset, read_observations(kind, path)))
    print(f"Cached {len(raw_datasets)} tuning datasets in memory.")

    for family_index, family in enumerate(args.families):
        task_name = f"{args.horizon_profile}_h{args.history_length}"
        study_name = f"synthetic_{family.lower().replace('-', '_')}_{task_name}"
        sampler = optuna.samplers.TPESampler(seed=args.seed + family_index)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=2)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        remaining = max(0, args.trials_per_family - len(study.trials))
        print(f"[{family}] existing={len(study.trials)}, target={args.trials_per_family}, running={remaining}")
        if remaining:
            objective = SyntheticObjective(
                family,
                raw_datasets,
                base_model_cfg,
                base_training_cfg,
                base_data_cfg,
                args,
            )
            study.optimize(
                objective,
                n_trials=remaining,
                gc_after_trial=True,
                callbacks=[print_trial_progress(family)],
            )
        save_study_artifacts(study, args.output_dir, family)
        if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            print(f"[{family}] best val_rmse={study.best_value:.6f}, trial={study.best_trial.number}")


if __name__ == "__main__":
    main()
