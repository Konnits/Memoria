"""Optuna para arquitecturas continuas sobre forecasting en tiempo físico.

El objetivo usa historias de ``observations.parquet`` y consulta la señal
limpia de ``truth.parquet`` en horizontes continuos. La selección se realiza
exclusivamente con validación; el split test no se itera ni se evalúa durante
el tuning.

Ejemplo acotado::

    conda run -n memoria python scripts/tune_physical_optuna.py \
        --presets long_gaps --dataset-ids long_gaps_gseed3031_0000 \
        --trials 4 --epochs 2 --max-train-samples 128 \
        --max-val-samples 64 --max-observation-rows-per-split 20000
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")])

from scripts.benchmark_physical_models import (  # noqa: E402
    DEFAULT_CONFIG,
    IMPLEMENTATION_SOURCE_PATHS,
    PhysicalCollate,
    PreparedPhysicalData,
    base_model_config,
    discover_datasets,
    file_provenance,
    implementation_provenance,
    prepare_physical_data,
)
from ts_transformer.models.continuous_basis_decoder import (  # noqa: E402
    ContinuousBasisDecoderConfig,
    TimeSeriesContinuousBasisDecoder,
)
from ts_transformer.models.query_cross_attention import (  # noqa: E402
    QueryCrossAttentionConfig,
    TimeSeriesQueryCrossAttention,
)
from ts_transformer.training import Trainer  # noqa: E402
from ts_transformer.training.optimizers import OptimizerConfig  # noqa: E402
from ts_transformer.training.train_loop import TrainingConfig  # noqa: E402
from ts_transformer.utils.seed import set_global_seed  # noqa: E402


DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "optuna_physical"
SEARCH_SPACE_VERSION = 2
ARCHITECTURES: dict[str, tuple[int, int]] = {
    "d32_h2": (32, 2),
    "d32_h4": (32, 4),
    "d64_h4": (64, 4),
    "d64_h8": (64, 8),
    "d96_h4": (96, 4),
    "d96_h8": (96, 8),
    "d128_h8": (128, 8),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tuning de QueryCross/BasisDecoder con horizontes físicos continuos; "
            "el split test queda reservado."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kinds", nargs="+", choices=("univariate", "multivariate"), default=None
    )
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--dataset-ids", nargs="+", default=None)
    parser.add_argument(
        "--limit-datasets-per-kind",
        type=int,
        default=None,
        help=(
            "Cap explícito por modalidad para pruebas acotadas. Por defecto se "
            "usan todos los datasets que satisfacen --kinds/--presets/--dataset-ids."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Total acumulado de trials COMPLETE al reanudar.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-observation-rows-per-split",
        type=int,
        default=None,
        help="Cap opcional para smoke tests; no usar para resultados finales.",
    )
    return parser.parse_args(argv)


def load_physical_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Configuración física inválida: {path}")
    return payload


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA pero no está disponible.")
    return requested


def resolve_paths(args: argparse.Namespace, raw: Mapping[str, Any]) -> None:
    args.config = args.config.resolve()
    args.data_root = Path(args.data_root or raw["data"].get("root", "data"))
    if not args.data_root.is_absolute():
        args.data_root = (REPOSITORY_ROOT / args.data_root).resolve()
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.is_absolute():
        args.output_dir = (REPOSITORY_ROOT / args.output_dir).resolve()
    args.kinds = tuple(args.kinds or raw["data"].get("kinds", ("univariate", "multivariate")))
    args.device = resolve_device(args.device)
    for name in (
        "trials",
        "epochs",
        "early_stopping_patience",
        "max_train_samples",
        "max_val_samples",
        "batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} debe ser > 0.")
    if args.limit_datasets_per_kind is not None and args.limit_datasets_per_kind <= 0:
        raise ValueError("--limit-datasets-per-kind debe ser > 0.")
    if args.max_observation_rows_per_split is not None:
        if args.max_observation_rows_per_split < 100:
            raise ValueError("--max-observation-rows-per-split debe ser >= 100.")


def build_protocol_payload(
    args: argparse.Namespace,
    raw: Mapping[str, Any],
    dataset_specs: Sequence[tuple[str, str, Path, Path]],
) -> dict[str, Any]:
    """Descripción canónica de todo lo que cambia el objetivo Optuna."""
    task = raw["task"]
    return {
        "schema_version": 1,
        "search_space_version": SEARCH_SPACE_VERSION,
        "implementation": implementation_provenance(
            (*IMPLEMENTATION_SOURCE_PATHS, Path(__file__).resolve())
        ),
        "datasets": [
            {
                "kind": kind,
                "preset": preset,
                # La reanudación depende de los bytes efectivos. ``size`` y
                # ``mtime`` no bastan para distinguir datasets reemplazados o
                # copiados preservando atributos del filesystem.
                "observations": file_provenance(observations),
                "truth": file_provenance(truth),
            }
            for kind, preset, observations, truth in dataset_specs
        ],
        "task": {
            "horizons": [float(value) for value in task["horizons"]],
            "train_horizon_range": [
                float(value) for value in task["train_horizon_range"]
            ],
            "train_horizon_sampling": task.get(
                "train_horizon_sampling", "log_uniform"
            ),
            "queries_per_sample": int(task.get("queries_per_sample", 4)),
            "history_duration": float(task["history_duration"]),
            "max_history_events_univariate": int(
                task["max_history_events_univariate"]
            ),
            "max_history_events_multivariate": int(
                task["max_history_events_multivariate"]
            ),
            "history_subsampling": task.get("history_subsampling", "uniform_time"),
            "target_source": "truth.parquet",
            "target_match_mode": "linear",
        },
        "tuning": {
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_stopping_patience),
            "max_train_samples": int(args.max_train_samples),
            "max_val_samples": int(args.max_val_samples),
            "batch_size": int(args.batch_size),
            "max_observation_rows_per_split": args.max_observation_rows_per_split,
            "seed": int(args.seed),
            "selection_split": "validation",
            "selection_metric": "val_rmse",
            "test_access": "forbidden_in_objective",
        },
    }


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_study_protocol(
    study: optuna.Study,
    fingerprint: str,
    payload: Mapping[str, Any],
) -> None:
    """Impide reusar un study con datasets/tarea/espacio incompatibles."""
    existing = study.user_attrs.get("protocol_fingerprint")
    if existing is not None and existing != fingerprint:
        raise RuntimeError(
            "El study existente pertenece a otro protocolo: "
            f"esperado={fingerprint}, existente={existing}."
        )
    if existing is None and study.trials:
        raise RuntimeError(
            "El study contiene trials sin fingerprint; no es seguro reanudarlo."
        )
    study.set_user_attr("protocol_fingerprint", fingerprint)
    study.set_user_attr(
        "protocol_payload", json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def prepare_tuning_data(
    args: argparse.Namespace,
    raw: Mapping[str, Any],
    specs: Sequence[tuple[str, str, Path, Path]],
) -> list[PreparedPhysicalData]:
    task = raw["task"]
    prepared: list[PreparedPhysicalData] = []
    for index, spec in enumerate(specs):
        kind = spec[0]
        max_history_events = int(
            task[
                "max_history_events_univariate"
                if kind == "univariate"
                else "max_history_events_multivariate"
            ]
        )
        prepared.append(
            prepare_physical_data(
                spec,
                horizons=tuple(float(value) for value in task["horizons"]),
                history_duration=float(task["history_duration"]),
                max_history_events=max_history_events,
                history_subsampling=task.get("history_subsampling", "uniform_time"),
                max_samples={
                    "train": args.max_train_samples,
                    "validation": args.max_val_samples,
                    # Se construye por contrato del dataset, pero el objetivo
                    # nunca crea un loader ni lee muestras de test.
                    "test": 1,
                },
                max_observation_rows_per_split=args.max_observation_rows_per_split,
                protocol_seed=args.seed + index * 10_000,
                train_horizon_range=tuple(
                    float(value) for value in task["train_horizon_range"]
                ),
                train_horizon_sampling=task.get(
                    "train_horizon_sampling", "log_uniform"
                ),
                queries_per_sample=int(task.get("queries_per_sample", 4)),
            )
        )
    return prepared


def make_tuning_loaders(
    data: PreparedPhysicalData,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: str,
) -> tuple[DataLoader, DataLoader]:
    """Crea sólo train/validation; deliberadamente no toca ``data.test``."""
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": PhysicalCollate(),
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": False,
    }
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(data.train, shuffle=True, generator=generator, **common),
        DataLoader(data.validation, shuffle=False, **common),
    )


def sample_model_and_training(
    trial: optuna.Trial,
    data: PreparedPhysicalData,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, TrainingConfig]:
    architecture = trial.suggest_categorical("architecture", tuple(ARCHITECTURES))
    d_model, num_heads = ARCHITECTURES[architecture]
    decoder_architecture = trial.suggest_categorical(
        "decoder_architecture", ("query_cross", "continuous_basis")
    )
    encoder_layers = trial.suggest_int("encoder_layers", 1, 3)
    cross_layers = (
        trial.suggest_int("cross_layers", 1, 3)
        if decoder_architecture == "query_cross"
        else 1
    )
    ffn_multiplier = trial.suggest_categorical("ffn_multiplier", (2, 4, 6))
    dropout = trial.suggest_categorical("dropout", (0.0, 0.05, 0.1, 0.15))
    model_args = argparse.Namespace(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=encoder_layers,
        cross_layers=cross_layers,
        dim_feedforward=d_model * ffn_multiplier,
        dropout=dropout,
    )
    base = base_model_config(data, model_args)
    base.time_encoding_mode = trial.suggest_categorical(
        "time_encoding_mode", ("sinusoidal", "time2vec")
    )
    base.time_transform = trial.suggest_categorical(
        "time_transform", ("linear", "log1p")
    )
    use_history_time_encoding = trial.suggest_categorical(
        "use_history_time_encoding", (False, True)
    )
    use_ctssm = trial.suggest_categorical("use_ctssm", (False, True))
    if decoder_architecture == "query_cross":
        decoder_config: QueryCrossAttentionConfig | ContinuousBasisDecoderConfig = (
            QueryCrossAttentionConfig(
                num_cross_layers=cross_layers,
                temporal_feature_dim=1,
                lag_num_frequencies=trial.suggest_categorical(
                    "lag_num_frequencies", (2, 4, 8)
                ),
                lag_min_scale=trial.suggest_categorical(
                    "lag_min_scale", (0.125, 0.25, 0.5)
                ),
                lag_max_scale=trial.suggest_categorical(
                    "lag_max_scale", (16.0, 64.0, 128.0)
                ),
                use_history_time_encoding=use_history_time_encoding,
                use_sensor_relation_bias=data.kind == "multivariate",
                use_ctssm=use_ctssm,
            )
        )
        model: torch.nn.Module = TimeSeriesQueryCrossAttention(base, decoder_config)
    else:
        decoder_config = ContinuousBasisDecoderConfig(
            trend_degree=trial.suggest_int("basis_trend_degree", 1, 3),
            num_rbf_bases=trial.suggest_categorical(
                "basis_num_rbf", (4, 8, 16)
            ),
            num_fourier_frequencies=trial.suggest_categorical(
                "basis_num_fourier", (2, 4, 8)
            ),
            min_basis_scale=trial.suggest_categorical(
                "basis_min_scale", (0.125, 0.25, 0.5)
            ),
            max_basis_scale=trial.suggest_categorical(
                "basis_max_scale", (16.0, 64.0, 128.0)
            ),
            temporal_feature_dim=1,
            use_history_time_encoding=use_history_time_encoding,
            use_ctssm=use_ctssm,
        )
        model = TimeSeriesContinuousBasisDecoder(base, decoder_config)
    optimizer = OptimizerConfig(
        optimizer_name="adamw",
        lr=trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        scheduler_name="cosine",
        scheduler_T_max=args.epochs,
    )
    training = TrainingConfig(
        num_epochs=args.epochs,
        device=args.device,
        loss_name="mse",
        optimizer_config=optimizer,
        grad_clip_norm=1.0,
        log_every_n_steps=0,
        checkpoint_dir=None,
        save_best_on="val_rmse",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True,
        use_amp=args.device.startswith("cuda"),
    )
    return model, training


class PhysicalObjective:
    def __init__(
        self,
        datasets: Sequence[PreparedPhysicalData],
        args: argparse.Namespace,
    ) -> None:
        self.datasets = list(datasets)
        self.args = args

    def __call__(self, trial: optuna.Trial) -> float:
        scores: list[float] = []
        for dataset_index, data in enumerate(self.datasets):
            # La misma inicialización por dataset en todos los trials reduce
            # varianza de comparación; el trial sólo cambia hiperparámetros.
            seed = self.args.seed + dataset_index
            set_global_seed(seed, deterministic=False)
            train_loader, val_loader = make_tuning_loaders(
                data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                seed=seed,
                device=self.args.device,
            )
            model, training = sample_model_and_training(trial, data, self.args)
            if dataset_index == 0:
                trial.set_user_attr(
                    "n_parameters",
                    sum(parameter.numel() for parameter in model.parameters()),
                )
            trainer = Trainer(model, train_loader, val_loader, config=training)
            try:
                trainer.fit()
                if trainer._best_model_state_in_memory is not None:
                    trainer.model.load_state_dict(trainer._best_model_state_in_memory)
                # Única métrica de selección: validación. No existe test_loader
                # en este scope para evitar accesos accidentales.
                score = float(
                    trainer.evaluate_on_loader(val_loader, prefix="val_")["val_rmse"]
                )
                if not np.isfinite(score):
                    raise optuna.TrialPruned("val_rmse no finito.")
                scores.append(score)
                label = f"{data.kind}_{data.preset}_{data.dataset_id}_val_rmse"
                trial.set_user_attr(label, score)
                trial.report(float(np.mean(scores)), step=dataset_index + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Podado tras {label}; mean={np.mean(scores):.6f}"
                    )
            finally:
                del trainer, model, train_loader, val_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        result = float(np.mean(scores))
        trial.set_user_attr("mean_val_rmse", result)
        return result


def save_artifacts(
    study: optuna.Study,
    output_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    ).to_csv(output_dir / "physical_trials.csv", index=False)
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        return
    best = study.best_trial
    artifact = {
        "schema_version": 1,
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_val_rmse": float(best.value),
        "params": best.params,
        "user_attrs": best.user_attrs,
        "protocol": payload,
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    with (output_dir / "best_physical.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(artifact, handle, sort_keys=False, allow_unicode=True)


def progress_callback(
    study: optuna.Study, trial: optuna.trial.FrozenTrial
) -> None:
    value = "N/A" if trial.value is None else f"{trial.value:.6f}"
    print(f"trial={trial.number} state={trial.state.name} val_rmse={value}")
    completed = [
        item for item in study.trials
        if item.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        print(
            f"mejor={study.best_trial.number} "
            f"val_rmse={study.best_value:.6f} "
            f"params={json.dumps(study.best_params, sort_keys=True)}"
        )


def completed_trial_count(study: optuna.Study) -> int:
    """Cuenta sólo evaluaciones utilizables para el presupuesto del estudio."""
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )


def stop_after_completed_trials(
    target_completed: int,
):
    """Callback que detiene Optuna al alcanzar el presupuesto de trials válidos."""
    if target_completed < 1:
        raise ValueError("target_completed debe ser positivo.")

    def callback(
        study: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        if completed_trial_count(study) >= target_completed:
            study.stop()

    return callback


def main() -> None:
    args = parse_args()
    raw = load_physical_config(args.config)
    resolve_paths(args, raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_datasets(
        args.data_root,
        args.kinds,
        args.presets,
        args.limit_datasets_per_kind,
        args.dataset_ids,
    )
    payload = build_protocol_payload(args, raw, specs)
    fingerprint = protocol_fingerprint(payload)
    storage_url = f"sqlite:///{(args.output_dir / 'physical_optuna.db').as_posix()}"
    storage = optuna.storages.RDBStorage(
        storage_url,
        heartbeat_interval=60,
        grace_period=180,
    )
    study = optuna.create_study(
        study_name=f"physical_irregular_{fingerprint[:12]}",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    ensure_study_protocol(study, fingerprint, payload)
    completed = completed_trial_count(study)
    remaining = max(0, args.trials - completed)
    print(
        f"study={study.study_name} datasets={len(specs)} "
        f"complete={completed} total_records={len(study.trials)} "
        f"target_complete={args.trials} missing={remaining}"
    )
    if remaining:
        datasets = prepare_tuning_data(args, raw, specs)
        # Los trials PRUNED/FAIL/RUNNING no satisfacen el presupuesto científico.
        # Se permite un margen finito para podas/fallos y se detiene apenas hay
        # ``args.trials`` resultados COMPLETE.
        max_new_attempts = max(remaining * 5, remaining + 10)
        study.optimize(
            PhysicalObjective(datasets, args),
            n_trials=max_new_attempts,
            callbacks=[
                progress_callback,
                stop_after_completed_trials(args.trials),
            ],
            gc_after_trial=True,
        )
        completed = completed_trial_count(study)
        if completed < args.trials:
            raise RuntimeError(
                "No se alcanzó el presupuesto de trials COMPLETE tras "
                f"{max_new_attempts} intentos: complete={completed}, "
                f"objetivo={args.trials}. Revise trials FAIL/PRUNED."
            )
    save_artifacts(study, args.output_dir, payload)
    if any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        print(f"best_val_rmse={study.best_value:.6f}")
        print(f"best_params={json.dumps(study.best_params, sort_keys=True)}")
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()
