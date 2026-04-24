from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


@dataclass(frozen=True)
class SearchParameter:
    section: str
    path: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SearchConfig:
    name: str
    strategy: str
    metric: str
    mode: str
    seed: int
    max_trials: int | None
    parameters: tuple[SearchParameter, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"El YAML {path} no contiene un diccionario válido.")
    return data


def _save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_nested_value(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = obj
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def load_search_config(path: str | Path) -> SearchConfig:
    raw = _load_yaml(path)

    name = str(raw.get("name", "search"))
    strategy = str(raw.get("strategy", "grid")).lower()
    if strategy not in {"grid", "random"}:
        raise ValueError("strategy debe ser 'grid' o 'random'.")

    mode = str(raw.get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ValueError("mode debe ser 'min' o 'max'.")

    metric = str(raw.get("metric", "val_rmse"))
    seed = int(raw.get("seed", 42))
    max_trials = raw.get("max_trials", None)
    if max_trials is not None:
        max_trials = int(max_trials)

    params_raw = raw.get("parameters", [])
    if not isinstance(params_raw, list) or not params_raw:
        raise ValueError("La búsqueda necesita una lista no vacía en 'parameters'.")

    params: list[SearchParameter] = []
    for item in params_raw:
        if not isinstance(item, dict):
            raise ValueError("Cada entrada de 'parameters' debe ser un diccionario.")
        section = str(item["section"]).lower()
        if section not in {"model", "training", "data"}:
            raise ValueError("section debe ser model, training o data.")
        dotted_path = str(item["path"])
        values = item.get("values", None)
        if not isinstance(values, list) or not values:
            raise ValueError(f"El parámetro {dotted_path!r} debe tener una lista 'values' no vacía.")
        params.append(
            SearchParameter(
                section=section,
                path=dotted_path,
                values=tuple(values),
            )
        )

    return SearchConfig(
        name=name,
        strategy=strategy,
        metric=metric,
        mode=mode,
        seed=seed,
        max_trials=max_trials,
        parameters=tuple(params),
    )


def generate_trials(search_cfg: SearchConfig) -> list[dict[str, Any]]:
    rng = random.Random(search_cfg.seed)
    all_trials = []

    if search_cfg.strategy == "grid":
        value_spaces = [param.values for param in search_cfg.parameters]
        for combination in itertools.product(*value_spaces):
            trial = {
                f"{param.section}.{param.path}": value
                for param, value in zip(search_cfg.parameters, combination)
            }
            all_trials.append(trial)
    else:
        if search_cfg.max_trials is None or search_cfg.max_trials <= 0:
            raise ValueError("Con strategy='random' debes definir max_trials > 0.")
        seen: set[str] = set()
        max_unique = math.prod(len(param.values) for param in search_cfg.parameters)
        target_trials = min(search_cfg.max_trials, max_unique)

        while len(all_trials) < target_trials:
            trial = {}
            for param in search_cfg.parameters:
                trial[f"{param.section}.{param.path}"] = rng.choice(param.values)
            signature = json.dumps(trial, sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            all_trials.append(trial)

    if search_cfg.max_trials is not None and search_cfg.strategy == "grid":
        all_trials = all_trials[: search_cfg.max_trials]

    return all_trials


def apply_trial_overrides(
    base_model_cfg: dict[str, Any],
    base_training_cfg: dict[str, Any],
    base_data_cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_cfg = copy.deepcopy(base_model_cfg)
    training_cfg = copy.deepcopy(base_training_cfg)
    data_cfg = copy.deepcopy(base_data_cfg)

    mapping = {
        "model": model_cfg,
        "training": training_cfg,
        "data": data_cfg,
    }

    for full_key, value in overrides.items():
        section, dotted_path = full_key.split(".", 1)
        _set_nested_value(mapping[section], dotted_path, value)

    return model_cfg, training_cfg, data_cfg


def _extract_best_metric(history: dict[str, Any], metric: str, mode: str) -> float:
    if metric not in history:
        raise KeyError(f"La métrica {metric!r} no existe en history.yaml.")

    values = history[metric]
    if not isinstance(values, list) or not values:
        raise ValueError(f"La métrica {metric!r} no tiene valores válidos.")

    numeric_values = [float(v) for v in values]
    return min(numeric_values) if mode == "min" else max(numeric_values)


def _build_python_command() -> list[str]:
    return [sys.executable, "-m", "ts_transformer.train"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta una búsqueda de hiperparámetros sobre train.py reutilizando los YAML base."
    )
    parser.add_argument("--search-config", type=str, required=True, help="YAML con el espacio de búsqueda.")
    parser.add_argument("--model-config", type=str, required=True, help="YAML base de modelo.")
    parser.add_argument("--training-config", type=str, required=True, help="YAML base de training.")
    parser.add_argument("--data-config", type=str, required=True, help="YAML base de datos.")
    parser.add_argument("--stage", type=str, choices=("pretrain", "finetune"), default="finetune")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/hparam_search",
        help="Carpeta raíz donde se escribirán trials y resumen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    search_cfg = load_search_config(args.search_config)
    base_model_cfg = _load_yaml(args.model_config)
    base_training_cfg = _load_yaml(args.training_config)
    base_data_cfg = _load_yaml(args.data_config)

    trials = generate_trials(search_cfg)
    if not trials:
        raise RuntimeError("No se generaron trials para la búsqueda.")

    output_dir = Path(args.output_dir) / search_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for trial_index, overrides in enumerate(trials, start=1):
        trial_name = f"trial_{trial_index:03d}"
        trial_dir = output_dir / trial_name
        config_dir = trial_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        model_cfg, training_cfg, data_cfg = apply_trial_overrides(
            base_model_cfg,
            base_training_cfg,
            base_data_cfg,
            overrides,
        )
        training_cfg["checkpoint_dir"] = str(trial_dir).replace("\\", "/")

        model_cfg_path = config_dir / "model.yaml"
        training_cfg_path = config_dir / "training.yaml"
        data_cfg_path = config_dir / "data.yaml"
        _save_yaml(model_cfg_path, model_cfg)
        _save_yaml(training_cfg_path, training_cfg)
        _save_yaml(data_cfg_path, data_cfg)

        command = _build_python_command() + [
            "--stage", args.stage,
            "--model-config", str(model_cfg_path),
            "--training-config", str(training_cfg_path),
            "--data-config", str(data_cfg_path),
        ]
        if args.init_checkpoint:
            command += ["--init-checkpoint", args.init_checkpoint]

        print(f"[Search] Ejecutando {trial_name} con overrides: {overrides}")
        subprocess.run(command, check=True)

        history_path = trial_dir / "history.yaml"
        history = _load_yaml(history_path)
        metric_value = _extract_best_metric(history, search_cfg.metric, search_cfg.mode)

        row = {
            "trial": trial_name,
            "metric": search_cfg.metric,
            "metric_value": metric_value,
        }
        row.update(overrides)
        summary_rows.append(row)

        is_better = False
        if best_row is None:
            is_better = True
        elif search_cfg.mode == "min" and metric_value < float(best_row["metric_value"]):
            is_better = True
        elif search_cfg.mode == "max" and metric_value > float(best_row["metric_value"]):
            is_better = True

        if is_better:
            best_row = row

    summary_path = output_dir / "results.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    best_path = output_dir / "best_trial.yaml"
    _save_yaml(best_path, {"search": search_cfg.name, "best_trial": best_row or {}})

    print(f"[Search] Resultados guardados en {summary_path}")
    print(f"[Search] Mejor trial guardado en {best_path}")


if __name__ == "__main__":
    main()