"""Microbenchmark reproducible del cuello DataLoader del pipeline físico.

El probe usa un dataset real ya generado, prepara los ejemplos una sola vez y
entrena QueryCross desde la misma seed para cada combinación batch/workers. No
es parte de la campaña final ni modifica su configuración congelada.

Ejemplo usado para perfilar el cierre de tesis::

    conda run -n memoria python scripts/probe_physical_loader_performance.py \
        --kind multivariate --preset hard_mixed \
        --dataset-id hard_mixed_0000 --batch-sizes 32 128 \
        --num-workers 0 2 4 --epochs 3 --max-train-samples 512
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.benchmark_physical_models import (  # noqa: E402
    DEFAULT_CONFIG,
    build_model,
    discover_datasets,
    load_config,
    make_loaders,
    prepare_physical_data,
    training_config,
)
from ts_transformer.training import Trainer  # noqa: E402
from ts_transformer.utils.seed import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perfila batch/num_workers con QueryCross sobre datos físicos reales."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kind", choices=("univariate", "multivariate"), default="multivariate")
    parser.add_argument("--preset", default="hard_mixed")
    parser.add_argument("--dataset-id", default="hard_mixed_0000")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(32, 128))
    parser.add_argument("--num-workers", nargs="+", type=int, default=(0, 2, 4))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Mide preparación/diagnósticos exactos sin entrenar escenarios.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Archivo opcional para conservar el resultado del probe.",
    )
    return parser.parse_args()


def _positive(values: list[int] | tuple[int, ...], name: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} debe contener enteros > 0.")


def _clear_dataset_caches(dataset: Any, visited: set[int] | None = None) -> None:
    """Reinicia sólo caches lazy para que cada escenario comience en frío."""

    if visited is None:
        visited = set()
    identity = id(dataset)
    if identity in visited:
        return
    visited.add(identity)
    for name in ("_dense_history_cache", "_event_history_cache"):
        cache = getattr(dataset, name, None)
        if cache is not None:
            cache.clear()
    nested = getattr(dataset, "dataset", None)
    if nested is not None:
        _clear_dataset_caches(nested, visited)
    for child in getattr(dataset, "datasets", ()):
        _clear_dataset_caches(child, visited)


def _shutdown_loader(loader: Any) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def _benchmark_sensor_diagnostics(data: Any) -> dict[str, Any] | None:
    """Compara la ruta histórica y la preindexada sobre ventanas reales."""

    wrapper = data.test
    base = getattr(wrapper, "dataset", wrapper)
    selected = list(getattr(wrapper, "indices", range(len(wrapper))))
    if not hasattr(base, "_sensor_diagnostics") or base.event_sensor_ids is None:
        return None
    cases: list[tuple[int, int, float]] = []
    for index in selected:
        anchor = int(base._example_indices[index])
        origin = base._forecast_origin(index, anchor)
        start, stop = base._history_bounds(index, anchor, origin)
        cases.append((start, stop, origin))

    brute_force = []
    start_time = time.perf_counter()
    for start, stop, origin in cases:
        brute_force.append(
            base._sensor_diagnostics(
                base.timestamps[start:stop],
                base.event_sensor_ids[start:stop],
                origin,
            )
        )
    brute_force_seconds = time.perf_counter() - start_time

    preindexed = []
    start_time = time.perf_counter()
    for start, stop, origin in cases:
        preindexed.append(
            base._sensor_diagnostics(
                base.timestamps[start:stop],
                base.event_sensor_ids[start:stop],
                origin,
                history_start=start,
                history_stop=stop,
            )
        )
    preindexed_seconds = time.perf_counter() - start_time
    for expected, actual in zip(brute_force, preindexed):
        if expected.keys() != actual.keys() or any(
            not torch.equal(expected[key], actual[key]) for key in expected
        ):
            raise AssertionError("Los diagnósticos preindexados no son equivalentes.")
    return {
        "windows": len(cases),
        "brute_force_seconds": brute_force_seconds,
        "preindexed_seconds": preindexed_seconds,
        "speedup": brute_force_seconds / preindexed_seconds,
        "exactly_equal": True,
    }


def _training_namespace(
    raw: dict[str, Any], args: argparse.Namespace, *, batch_size: int, workers: int
) -> argparse.Namespace:
    model = raw["model"]
    training = raw["training"]
    return argparse.Namespace(
        d_model=int(model["d_model"]),
        num_heads=int(model["num_heads"]),
        num_layers=int(model["num_layers"]),
        cross_layers=int(model["cross_layers"]),
        dim_feedforward=int(model["dim_feedforward"]),
        dropout=float(model["dropout"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        epochs=int(args.epochs),
        batch_size=int(batch_size),
        num_workers=int(workers),
        device=str(args.device),
        deterministic=False,
    )


def main() -> None:
    args = parse_args()
    _positive(list(args.batch_sizes), "batch-sizes")
    if not args.num_workers or any(value < 0 for value in args.num_workers):
        raise ValueError("num-workers debe contener enteros >= 0.")
    _positive(
        [args.epochs, args.max_train_samples, args.max_val_samples, args.max_test_samples],
        "epochs/max-samples",
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("El probe solicitó CUDA, pero CUDA no está disponible.")

    raw = load_config(args.config)
    data_root = Path(raw["data"].get("root", REPOSITORY_ROOT / "data"))
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    matches = discover_datasets(
        data_root,
        (args.kind,),
        (args.preset,),
        None,
        (args.dataset_id,),
    )
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaba un dataset y se encontraron {len(matches)}.")

    task = raw["task"]
    sampling = raw["sampling"]
    max_history_events = int(
        task[
            "max_history_events_univariate"
            if args.kind == "univariate"
            else "max_history_events_multivariate"
        ]
    )
    prepare_start = time.perf_counter()
    data = prepare_physical_data(
        matches[0],
        horizons=tuple(float(value) for value in task["horizons"]),
        history_duration=float(task["history_duration"]),
        max_history_events=max_history_events,
        history_subsampling=str(task.get("history_subsampling", "uniform_time")),
        max_samples={
            "train": args.max_train_samples,
            "validation": args.max_val_samples,
            "test": args.max_test_samples,
        },
        max_observation_rows_per_split=None,
        protocol_seed=int(sampling.get("protocol_seed", 2026)),
        train_horizon_range=tuple(
            float(value) for value in task["train_horizon_range"]
        ),
        train_horizon_sampling=str(task.get("train_horizon_sampling", "log_uniform")),
        queries_per_sample=int(task.get("queries_per_sample", len(task["horizons"]))),
    )
    prepare_seconds = time.perf_counter() - prepare_start
    sensor_diagnostics = _benchmark_sensor_diagnostics(data)

    results: list[dict[str, Any]] = []
    batch_sizes = () if args.diagnostics_only else args.batch_sizes
    for batch_size in batch_sizes:
        for workers in args.num_workers:
            for split in (data.train, data.validation, data.test):
                _clear_dataset_caches(split)
            scenario_args = _training_namespace(
                raw, args, batch_size=batch_size, workers=workers
            )
            set_global_seed(args.seed, deterministic=False)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            wall_start = time.perf_counter()
            train_loader, val_loader, test_loader = make_loaders(
                data,
                batch_size=batch_size,
                num_workers=workers,
                seed=args.seed,
                device=args.device,
            )
            model = build_model("QueryCross", data, scenario_args)
            if model is None:
                raise RuntimeError("QueryCross no pudo construirse.")
            trainer = Trainer(
                model,
                train_loader,
                val_loader,
                config=training_config(
                    scenario_args, checkpoint_dir=None, model_name="QueryCross"
                ),
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            fit_start = time.perf_counter()
            history = trainer.fit()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - fit_start
            for loader in (train_loader, val_loader, test_loader):
                _shutdown_loader(loader)
            scenario_wall_seconds = time.perf_counter() - wall_start
            peak_mib = (
                torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20
                if args.device.startswith("cuda")
                else 0.0
            )
            result = {
                "batch_size": int(batch_size),
                "num_workers": int(workers),
                "epochs": int(args.epochs),
                "fit_seconds": fit_seconds,
                "scenario_wall_seconds": scenario_wall_seconds,
                "samples_per_fit_second": (
                    len(data.train) * args.epochs / fit_seconds
                ),
                "last_train_loss": float(history["train_loss"][-1]),
                "last_val_rmse": float(history["val_rmse"][-1]),
                "peak_cuda_memory_mib": peak_mib,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            del trainer, model, train_loader, val_loader, test_loader
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "scope": "diagnostic_probe_not_final_evaluation",
        "dataset": {
            "kind": data.kind,
            "preset": data.preset,
            "dataset_id": data.dataset_id,
            "max_history_events": data.max_history_events,
            "train_samples": len(data.train),
            "validation_samples": len(data.validation),
        },
        "fast_cuda_kernels": True,
        "deterministic": False,
        "seed": args.seed,
        "prepare_seconds": prepare_seconds,
        "sensor_diagnostics": sensor_diagnostics,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
