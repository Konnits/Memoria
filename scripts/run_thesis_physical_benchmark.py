"""Orquestador reproducible del benchmark físico final de la tesis.

El protocolo selecciona unidades exactas, conserva ``main`` y ``stress`` en
directorios distintos, ejecuta controles de identificabilidad y sólo genera el
reporte cuando todos los runs esperados están completos. Los hiperparámetros
son los valores fijos del YAML; este módulo no lanza búsquedas ni los modifica a
partir de métricas de test.

Uso recomendado (desde la raíz del repositorio)::

    conda run -n memoria python scripts/run_thesis_physical_benchmark.py preflight
    conda run -n memoria python scripts/run_thesis_physical_benchmark.py dry-run
    conda run -n memoria python scripts/run_thesis_physical_benchmark.py smoke
    conda run -n memoria python scripts/run_thesis_physical_benchmark.py run
    conda run -n memoria python scripts/run_thesis_physical_benchmark.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")])

from scripts.benchmark_physical_models import (  # noqa: E402
    MODEL_NAMES,
    atomic_write_json,
    completed_run_result,
    file_provenance,
    is_gaussian_model,
    sha256_file,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "thesis_physical_final.yaml"
)
PHYSICAL_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_physical_models.py"
IDENTIFIABILITY_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "temporal_identifiability_benchmark.py"
)
EXPECTED_SEEDS = (42, 84, 126)
COHORT_NAMES = ("main", "stress")
LATEX_TABLE_FILENAMES = (
    "protocol_summary.tex",
    "model_results.tex",
    "temporal_ablations.tex",
    "gaussian_calibration.tex",
)
RESULT_KEYS = ("Cohort", "Kind", "Preset", "Dataset_ID", "Seed", "Model")
SUMMARY_METRICS = (
    "test_rmse_z",
    "test_mae_z",
    "test_rmse",
    "test_mae",
    "test_nll_z",
    "test_crps_z",
    "test_mean_scale_z",
    "test_coverage_90",
    "test_coverage_95",
)


class ProtocolError(RuntimeError):
    """El protocolo o sus artefactos no cumplen el contrato final congelado."""


@dataclass(frozen=True, order=True)
class DatasetUnit:
    kind: str
    preset: str
    dataset_id: str
    generator_seed: int

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.preset, self.dataset_id


@dataclass(frozen=True)
class CommandSpec:
    cohort: str
    protocol: str
    command: tuple[str, ...]
    output_dir: Path


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_final_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ProtocolError(f"Configuración final inválida: {path}")
    raw["_config_path"] = path
    validate_final_config(raw)
    return raw


def cohort_units(config: Mapping[str, Any], cohort: str) -> tuple[DatasetUnit, ...]:
    if cohort not in COHORT_NAMES:
        raise ProtocolError(f"Cohorte desconocida: {cohort}")
    section = config["dataset_cohorts"][cohort]
    generator_seed = int(section["generator_seed"])
    return tuple(
        DatasetUnit(
            kind=str(item["kind"]),
            preset=str(item["preset"]),
            dataset_id=str(item["dataset_id"]),
            generator_seed=generator_seed,
        )
        for item in section["units"]
    )


def validate_final_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_status") != "thesis_physical_evaluation_v1_frozen":
        raise ProtocolError("El protocolo final debe estar congelado como evaluación física.")
    claim_boundary = str(config.get("claim_boundary", ""))
    if "retrospective" not in claim_boundary or "not_external_validation" not in claim_boundary:
        raise ProtocolError("Falta el límite retrospectivo/no externo de las conclusiones.")
    seeds = tuple(int(value) for value in config.get("training", {}).get("seeds", ()))
    if seeds != EXPECTED_SEEDS:
        raise ProtocolError(f"Las seeds finales deben ser exactamente {EXPECTED_SEEDS}.")
    if not bool(config.get("training", {}).get("deterministic", False)):
        raise ProtocolError("El benchmark final debe declarar deterministic: true.")
    if int(config["training"].get("batch_size", -1)) != 32:
        raise ProtocolError("El batch_size final congelado debe ser 32.")
    if str(config["training"].get("device")) != "cuda":
        raise ProtocolError("La ejecución final exige device: cuda.")
    if int(config["training"].get("num_workers", -1)) != 0:
        raise ProtocolError("La ejecución reproducible exige num_workers: 0.")
    if (
        config.get("task", {}).get("history_subsampling")
        not in {"uniform_time", "uniform_index"}
        or config.get("task", {}).get("cache_deterministic_history") is not True
    ):
        raise ProtocolError("El protocolo final exige cache histórico determinista.")
    checkpoint = config.get("training", {}).get("checkpoint_selection", {})
    if checkpoint != {"point": "val_rmse", "gaussian": "val_nll"}:
        raise ProtocolError("Checkpoint final: point=val_rmse y gaussian=val_nll.")
    models = tuple(config.get("models", ()))
    if not models or len(models) != len(set(models)) or not set(models) <= set(MODEL_NAMES):
        raise ProtocolError("La lista final de modelos es vacía, duplicada o desconocida.")
    if config.get("hyperparameter_policy") != "fixed_before_final_execution":
        raise ProtocolError("Los hiperparámetros finales deben estar fijados antes del run.")
    if tuple(config.get("reporting", {}).get("latex_tables", ())) != LATEX_TABLE_FILENAMES:
        raise ProtocolError(
            "Los cuatro nombres LaTeX deben coincidir exactamente con el contrato."
        )
    if int(config.get("execution", {}).get("num_workers_required", -1)) != 0:
        raise ProtocolError("execution.num_workers_required debe ser 0.")
    preflight = config.get("preflight", {})
    if (
        preflight.get("required_conda_environment") != "memoria"
        or preflight.get("require_clean_worktree") is not True
        or preflight.get("full_pytest") is not True
        or preflight.get("require_git_diff_check") is not True
    ):
        raise ProtocolError("El preflight P0 (memoria/clean/pytest/diff-check) es obligatorio.")
    required_ablations = {
        "real",
        "all_equal",
        "permuted_gaps",
        "regular_grid",
        "query_only",
        "history_only",
    }
    if not required_ablations <= set(config.get("evaluation", {}).get("timestamp_ablations", ())):
        raise ProtocolError("Faltan ablaciones temporales finales.")

    expected_counts = {"main": 16, "stress": 2}
    for cohort, expected_count in expected_counts.items():
        units = cohort_units(config, cohort)
        declared = int(config["dataset_cohorts"][cohort]["expected_units"])
        if len(units) != declared or declared != expected_count:
            raise ProtocolError(
                f"{cohort} debe contener exactamente {expected_count} unidades."
            )
        if len({unit.key for unit in units}) != len(units):
            raise ProtocolError(f"{cohort} contiene unidades duplicadas.")
        for preset in sorted({unit.preset for unit in units}):
            kinds = {unit.kind for unit in units if unit.preset == preset}
            if kinds != {"univariate", "multivariate"}:
                raise ProtocolError(f"{cohort}/{preset} no forma un par de kinds.")

    main = cohort_units(config, "main")
    if any(
        unit.generator_seed != 2026
        or unit.dataset_id != f"{unit.preset}_0000"
        or "gseed" in unit.dataset_id
        for unit in main
    ):
        raise ProtocolError("main debe ser la realización canónica *_0000 de seed 2026.")
    stress = cohort_units(config, "stress")
    if any(
        unit.generator_seed != 3031
        or unit.preset != "long_gaps"
        or unit.dataset_id != "long_gaps_gseed3031_0000"
        for unit in stress
    ):
        raise ProtocolError("stress debe ser long_gaps_gseed3031_0000 (seed 3031).")


def data_root(config: Mapping[str, Any]) -> Path:
    return _repository_path(config["data"]["root"]).resolve()


def output_root(config: Mapping[str, Any], override: Path | None = None) -> Path:
    return Path(override or _repository_path(config["output_dir"])).resolve()


def cohort_output_dir(
    config: Mapping[str, Any], cohort: str, root: Path
) -> Path:
    return root / str(config["dataset_cohorts"][cohort]["output_subdir"])


def unit_paths(root: Path, unit: DatasetUnit) -> dict[str, Path]:
    directory = root / unit.kind / unit.preset / unit.dataset_id
    return {
        "directory": directory,
        "observations": directory / "observations.parquet",
        "truth": directory / "truth.parquet",
        "metadata": directory / "metadata.json",
    }


def _source_record(path: Path, hash_files: bool) -> dict[str, Any]:
    stat = path.stat()
    if hash_files:
        return file_provenance(path) | {"mtime_ns": int(stat.st_mtime_ns)}
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
    }


def _validate_parquet_contract(path: Path, required: set[str]) -> list[str]:
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(required - columns)
    if missing:
        raise ProtocolError(f"{path} no contiene columnas requeridas: {missing}")
    return sorted(columns)


def cuda_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Valida CUDA, nombre esperado y VRAM antes de reservar datos/modelos."""
    execution = config.get("execution", {})
    if not bool(execution.get("require_cuda", False)):
        raise ProtocolError("El protocolo final debe exigir CUDA explícitamente.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ProtocolError("CUDA no está disponible; no se inicia la campaña final.")
    minimum_bytes = float(execution.get("minimum_gpu_memory_gib", 0)) * 2**30
    minimum_free_bytes = (
        float(execution.get("minimum_free_gpu_memory_gib", 0)) * 2**30
    )
    expected_name = str(execution.get("expected_gpu_name_contains", "")).lower()
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_memory, _ = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_gib": float(properties.total_memory / 2**30),
                "free_memory_bytes": int(free_memory),
                "free_memory_gib": float(free_memory / 2**30),
            }
        )
    matching = [
        item
        for item in devices
        if expected_name in str(item["name"]).lower()
        and int(item["total_memory_bytes"]) >= minimum_bytes
        and int(item["free_memory_bytes"]) >= minimum_free_bytes
    ]
    if not matching:
        raise ProtocolError(
            "No hay una GPU compatible con el protocolo: "
            f"nombre contiene '{execution.get('expected_gpu_name_contains')}' y "
            f"VRAM total/libre >= {execution.get('minimum_gpu_memory_gib')}/"
            f"{execution.get('minimum_free_gpu_memory_gib')} GiB. Detectadas: {devices}"
        )
    return devices


def runtime_versions() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pyarrow": str(pyarrow.__version__),
        "pytorch": str(torch.__version__),
        "pytorch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "pyyaml": str(yaml.__version__),
    }


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"Falló git {' '.join(arguments)}: {exc}") from exc
    return completed.stdout.replace("\r\n", "\n").strip()


def repository_state() -> dict[str, Any]:
    status = _git_output("status", "--short", "--untracked-files=all")
    return {
        "git_head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(status),
        "status_entry_count": len(status.splitlines()) if status else 0,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _run_preflight_command(command: Sequence[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + "\n" + completed.stderr).replace("\r\n", "\n").strip()
    record = {
        "label": label,
        "command": list(command),
        "returncode": int(completed.returncode),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_tail": output.splitlines()[-20:],
    }
    if completed.returncode != 0:
        raise ProtocolError(
            f"Preflight '{label}' falló ({completed.returncode}):\n"
            + "\n".join(record["output_tail"])
        )
    return record


def runtime_preflight(
    config: Mapping[str, Any],
    *,
    strict: bool,
    run_checks: bool,
    focal_tests: bool = False,
) -> dict[str, Any]:
    preflight = config.get("preflight", {})
    required_env = str(preflight.get("required_conda_environment", "memoria"))
    active_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if active_env != required_env:
        raise ProtocolError(
            f"Entorno conda activo '{active_env}'; se exige '{required_env}'."
        )
    initial_repository = repository_state()
    if strict and bool(preflight.get("require_clean_worktree", True)) and initial_repository["worktree_dirty"]:
        raise ProtocolError(
            "El preflight final exige un worktree limpio y un HEAD identificable."
        )
    checks: list[dict[str, Any]] = []
    if run_checks:
        if bool(preflight.get("require_git_diff_check", True)):
            checks.append(
                _run_preflight_command(
                    ("git", "diff", "--check", "HEAD"), "git_diff_check"
                )
            )
        if focal_tests:
            tests = [str(value) for value in preflight.get("smoke_focal_tests", ())]
            checks.append(
                _run_preflight_command(
                    (sys.executable, "-m", "pytest", *tests, "-q"),
                    "pytest_focal",
                )
            )
        elif strict and bool(preflight.get("full_pytest", True)):
            checks.append(
                _run_preflight_command(
                    (sys.executable, "-m", "pytest", "-q"), "pytest_full"
                )
            )
    final_repository = repository_state()
    if strict and final_repository["worktree_dirty"]:
        raise ProtocolError("Las pruebas del preflight dejaron cambios en el worktree.")
    return {
        "conda_environment": active_env,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "strict": strict,
        "repository": final_repository,
        "checks": checks,
        "cuda_devices": cuda_inventory(config),
        "versions": runtime_versions(),
    }


def implementation_paths() -> tuple[Path, ...]:
    return tuple(sorted((REPOSITORY_ROOT / "src/ts_transformer").rglob("*.py"))) + (
        PHYSICAL_SCRIPT,
        IDENTIFIABILITY_SCRIPT,
        Path(__file__).resolve(),
    )


def preflight_manifest(
    config: Mapping[str, Any],
    *,
    root: Path,
    cohorts: Sequence[str] = COHORT_NAMES,
    hash_files: bool = True,
    strict: bool = False,
    run_checks: bool = False,
    focal_tests: bool = False,
) -> dict[str, Any]:
    selected = tuple(dict.fromkeys(cohorts))
    if not selected or not set(selected) <= set(COHORT_NAMES):
        raise ProtocolError("--cohorts debe seleccionar main y/o stress.")
    runtime = runtime_preflight(
        config,
        strict=strict,
        run_checks=run_checks,
        focal_tests=focal_tests,
    )
    data_directory = data_root(config)
    records: list[dict[str, Any]] = []
    for cohort in selected:
        for unit in cohort_units(config, cohort):
            paths = unit_paths(data_directory, unit)
            missing = [str(path) for name, path in paths.items() if name != "directory" and not path.is_file()]
            if missing:
                raise ProtocolError("Faltan fuentes del dataset: " + ", ".join(missing))
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("dataset_id") != unit.dataset_id or metadata.get("kind") != unit.kind:
                raise ProtocolError(f"Metadata inconsistente en {paths['metadata']}")
            observation_required = {"time", "value", "split"}
            truth_required = {"time", "clean_value", "split"}
            if unit.kind == "multivariate":
                observation_required.add("channel_index")
                truth_required.add("channel_index")
            records.append(
                {
                    "cohort": cohort,
                    "kind": unit.kind,
                    "preset": unit.preset,
                    "dataset_id": unit.dataset_id,
                    "generator_seed": unit.generator_seed,
                    "generator_seed_source": "thesis_physical_final.yaml",
                    "observations_columns": _validate_parquet_contract(
                        paths["observations"], observation_required
                    ),
                    "truth_columns": _validate_parquet_contract(paths["truth"], truth_required),
                    "sources": {
                        name: _source_record(path, hash_files)
                        for name, path in paths.items()
                        if name != "directory"
                    },
                }
            )

    config_path = Path(config["_config_path"])
    temporal_config = _repository_path(config["identifiability"]["config"])
    manifest = {
        "schema_version": 1,
        "status": "preflight_passed",
        "config": _source_record(config_path, hash_files),
        "implementation": [
            _source_record(path, hash_files) for path in implementation_paths()
        ],
        "identifiability_config": _source_record(temporal_config, hash_files),
        "data_root": str(data_directory),
        "output_root": str(root.resolve()),
        "cohorts": list(selected),
        "seeds": list(EXPECTED_SEEDS),
        "models": list(config["models"]),
        "timestamp_ablations": list(config["evaluation"]["timestamp_ablations"]),
        "deterministic": True,
        "runtime": runtime,
        "dataset_generator_provenance": {
            "main": {"generator_seed": 2026, "dataset_id_rule": "<preset>_0000"},
            "stress": {
                "generator_seed": 3031,
                "dataset_id_rule": "long_gaps_gseed3031_0000",
            },
        },
        "datasets": records,
        "n_dataset_units": len(records),
    }
    manifest["manifest_fingerprint"] = _mapping_sha256(manifest)
    return manifest


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_preflight_manifest(manifest: Mapping[str, Any], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preflight_manifest.json"
    atomic_write_json(path, manifest)
    return path


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _assert_source_compatible(
    record: Mapping[str, Any], *, verify_sha256: bool
) -> None:
    path = _record_path(record)
    if not path.is_file():
        raise ProtocolError(f"Fuente del preflight ausente: {path}")
    stat = path.stat()
    if int(record.get("size", -1)) != stat.st_size or int(
        record.get("mtime_ns", -1)
    ) != stat.st_mtime_ns:
        raise ProtocolError(f"Fuente cambió desde el preflight: {path}")
    if verify_sha256 and sha256_file(path) != record.get("sha256"):
        raise ProtocolError(f"SHA-256 cambió desde el preflight: {path}")


def require_compatible_preflight(
    config: Mapping[str, Any], *, root: Path, cohorts: Sequence[str]
) -> dict[str, Any]:
    path = root / "preflight_manifest.json"
    if not path.is_file():
        raise ProtocolError(
            f"Falta {path}. Ejecute primero el subcomando preflight desde HEAD limpio."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Manifest de preflight inválido: {path}") from exc
    fingerprint = manifest.get("manifest_fingerprint")
    payload = dict(manifest)
    payload.pop("manifest_fingerprint", None)
    if fingerprint != _mapping_sha256(payload):
        raise ProtocolError("El manifest de preflight fue modificado o truncado.")
    if manifest.get("status") != "preflight_passed" or set(
        manifest.get("cohorts", ())
    ) != set(cohorts):
        raise ProtocolError("El preflight no corresponde a las cohortes solicitadas.")
    if Path(str(manifest.get("output_root", ""))).resolve() != root.resolve():
        raise ProtocolError("El output root no coincide con el preflight.")
    expected_units = {
        (cohort, *unit.key)
        for cohort in cohorts
        for unit in cohort_units(config, cohort)
    }
    recorded_units = {
        (
            str(item.get("cohort")),
            str(item.get("kind")),
            str(item.get("preset")),
            str(item.get("dataset_id")),
        )
        for item in manifest.get("datasets", ())
    }
    if recorded_units != expected_units:
        raise ProtocolError("La selección de datasets cambió desde el preflight.")
    runtime = manifest.get("runtime", {})
    if not bool(runtime.get("strict", False)):
        raise ProtocolError("El run final requiere un preflight estricto.")
    labels = {item.get("label") for item in runtime.get("checks", ())}
    if not {"git_diff_check", "pytest_full"} <= labels:
        raise ProtocolError("El preflight no registra pytest completo y git diff --check.")
    current_repository = repository_state()
    recorded_repository = runtime.get("repository", {})
    if current_repository["worktree_dirty"] or (
        current_repository["git_head"] != recorded_repository.get("git_head")
    ):
        raise ProtocolError("HEAD/worktree ya no coincide con el preflight limpio.")
    active_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if active_env != runtime.get("conda_environment") or active_env != "memoria":
        raise ProtocolError("El entorno conda ya no coincide con el preflight.")
    if runtime_versions() != runtime.get("versions"):
        raise ProtocolError("Las versiones numéricas cambiaron desde el preflight.")
    cuda_inventory(config)

    _assert_source_compatible(manifest["config"], verify_sha256=True)
    _assert_source_compatible(manifest["identifiability_config"], verify_sha256=True)
    for record in manifest.get("implementation", ()):
        _assert_source_compatible(record, verify_sha256=True)
    # Los archivos de datos ya tienen SHA-256 completo en el manifest; tamaño y
    # mtime permiten invalidarlo en O(n) sin releer decenas de GB justo al run.
    for dataset in manifest.get("datasets", ()):
        for record in dataset.get("sources", {}).values():
            _assert_source_compatible(record, verify_sha256=False)
    return manifest


def _selection_arguments(units: Sequence[DatasetUnit]) -> list[str]:
    kinds = sorted({unit.kind for unit in units})
    presets = sorted({unit.preset for unit in units})
    dataset_ids = sorted({unit.dataset_id for unit in units})
    return [
        "--kinds",
        *kinds,
        "--presets",
        *presets,
        "--dataset-ids",
        *dataset_ids,
    ]


def build_run_commands(
    config: Mapping[str, Any],
    *,
    root: Path,
    cohorts: Sequence[str] = COHORT_NAMES,
    include_identifiability: bool = True,
    smoke: bool = False,
) -> list[CommandSpec]:
    config_path = Path(config["_config_path"])
    data_directory = data_root(config)
    models = [str(value) for value in config["models"]]
    commands: list[CommandSpec] = []

    if smoke:
        smoke_cfg = config["smoke"]
        cohort = str(smoke_cfg["cohort"])
        allowed_presets = set(smoke_cfg["presets"])
        units = tuple(
            unit for unit in cohort_units(config, cohort) if unit.preset in allowed_presets
        )
        if not units:
            raise ProtocolError("La selección smoke quedó vacía.")
        physical_output = root / "smoke" / "physical_models"
        physical = [
            sys.executable,
            str(PHYSICAL_SCRIPT),
            "--config",
            str(config_path),
            "--data-root",
            str(data_directory),
            "--output-dir",
            str(physical_output),
            *_selection_arguments(units),
            "--models",
            *models,
            "--seeds",
            *(str(value) for value in smoke_cfg["seeds"]),
            "--epochs",
            str(smoke_cfg["epochs"]),
            "--max-train-samples",
            str(smoke_cfg["max_train_samples"]),
            "--max-val-samples",
            str(smoke_cfg["max_val_samples"]),
            "--max-test-samples",
            str(smoke_cfg["max_test_samples"]),
            "--max-observation-rows-per-split",
            str(smoke_cfg["max_observation_rows_per_split"]),
            "--batch-size",
            "32",
            "--device",
            "cuda",
            "--num-workers",
            "0",
            "--deterministic",
            "--force-rerun",
        ]
        commands.append(CommandSpec("smoke", "physical_models", tuple(physical), physical_output))
        if include_identifiability and bool(config["identifiability"].get("enabled", True)):
            temporal_output = root / "smoke" / "temporal_identifiability"
            temporal = [
                sys.executable,
                str(IDENTIFIABILITY_SCRIPT),
                "--config",
                str(_repository_path(config["identifiability"]["config"])),
                "--data-root",
                str(data_directory),
                "--output-dir",
                str(temporal_output),
                *_selection_arguments(units),
                "--max-train-anchors",
                str(smoke_cfg["max_train_anchors"]),
                "--max-eval-anchors",
                str(smoke_cfg["max_eval_anchors"]),
            ]
            commands.append(
                CommandSpec("smoke", "temporal_identifiability", tuple(temporal), temporal_output)
            )
        return commands

    for cohort in cohorts:
        units = cohort_units(config, cohort)
        cohort_dir = cohort_output_dir(config, cohort, root)
        physical_output = cohort_dir / "physical_models"
        physical = [
            sys.executable,
            str(PHYSICAL_SCRIPT),
            "--config",
            str(config_path),
            "--data-root",
            str(data_directory),
            "--output-dir",
            str(physical_output),
            *_selection_arguments(units),
            "--models",
            *models,
            "--seeds",
            *(str(value) for value in EXPECTED_SEEDS),
            "--batch-size",
            "32",
            "--device",
            "cuda",
            "--num-workers",
            "0",
            "--deterministic",
        ]
        commands.append(CommandSpec(cohort, "physical_models", tuple(physical), physical_output))
        if include_identifiability and bool(config["identifiability"].get("enabled", True)):
            temporal_output = cohort_dir / "temporal_identifiability"
            temporal = [
                sys.executable,
                str(IDENTIFIABILITY_SCRIPT),
                "--config",
                str(_repository_path(config["identifiability"]["config"])),
                "--data-root",
                str(data_directory),
                "--output-dir",
                str(temporal_output),
                *_selection_arguments(units),
            ]
            commands.append(
                CommandSpec(cohort, "temporal_identifiability", tuple(temporal), temporal_output)
            )
    return commands


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def execute_commands(commands: Sequence[CommandSpec]) -> None:
    for index, spec in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {spec.cohort}/{spec.protocol}", flush=True)
        print(command_text(spec.command), flush=True)
        subprocess.run(spec.command, cwd=REPOSITORY_ROOT, check=True)


def _expected_result_keys(
    config: Mapping[str, Any], cohort: str
) -> set[tuple[str, str, str, int, str]]:
    return {
        (unit.kind, unit.preset, unit.dataset_id, seed, str(model))
        for unit in cohort_units(config, cohort)
        for seed in EXPECTED_SEEDS
        for model in config["models"]
    }


def _verify_run_against_preflight(
    *,
    config: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    preflight: Mapping[str, Any],
    cohort: str,
    kind: str,
    preset: str,
    dataset_id: str,
    model: str,
) -> None:
    frozen_task = config["task"]
    actual_task = run_configuration.get("task", {})
    ordered_units = sorted(cohort_units(config, cohort))
    unit_index = [unit.key for unit in ordered_units].index(
        (kind, preset, dataset_id)
    )
    expected_task = {
        "horizons": list(frozen_task["horizons"]),
        "train_horizon_range": list(frozen_task["train_horizon_range"]),
        "train_horizon_sampling": frozen_task["train_horizon_sampling"],
        "queries_per_sample": int(frozen_task["queries_per_sample"]),
        "history_duration": float(frozen_task["history_duration"]),
        "max_history_events": int(
            frozen_task[
                "max_history_events_univariate"
                if kind == "univariate"
                else "max_history_events_multivariate"
            ]
        ),
        "history_subsampling": frozen_task["history_subsampling"],
        "cache_deterministic_history": True,
        "max_observation_rows_per_split": None,
        "protocol_seed": int(config["sampling"]["protocol_seed"])
        + unit_index * 10_000,
    }
    for key, expected in expected_task.items():
        observed = actual_task.get(key)
        if observed != expected:
            raise ProtocolError(
                f"task.{key}={observed!r}; el protocolo congelado exige {expected!r}."
            )

    expected_sampling = {
        "max_train_samples": int(config["sampling"]["max_train_samples"]),
        "max_val_samples": int(config["sampling"]["max_val_samples"]),
        "max_test_samples": int(config["sampling"]["max_test_samples"]),
        "batch_size": 32,
        "num_workers": 0,
    }
    actual_sampling = run_configuration.get("sampling", {})
    if any(actual_sampling.get(key) != value for key, value in expected_sampling.items()):
        raise ProtocolError(
            f"Sampling efectivo {actual_sampling} != congelado {expected_sampling}."
        )

    frozen_training = config["training"]
    actual_training = run_configuration.get("training", {})
    expected_training = {
        "num_epochs": int(frozen_training["epochs"]),
        "device": "cuda",
        "loss_name": "mse",
        "early_stopping_patience": int(frozen_training["early_stopping_patience"]),
        "restore_best_weights": True,
        "use_amp": True,
        "enable_cuda_runtime_optimizations": False,
    }
    if any(actual_training.get(key) != value for key, value in expected_training.items()):
        raise ProtocolError("Configuración de entrenamiento efectiva no congelada.")
    optimizer = actual_training.get("optimizer_config", {})
    expected_optimizer = {
        "optimizer_name": "adamw",
        "lr": float(frozen_training["learning_rate"]),
        "weight_decay": float(frozen_training["weight_decay"]),
        "scheduler_name": "cosine",
        "scheduler_T_max": int(frozen_training["epochs"]),
    }
    if any(optimizer.get(key) != value for key, value in expected_optimizer.items()):
        raise ProtocolError("Optimizador/scheduler efectivo no coincide con el YAML.")

    frozen_model = config["model"]
    actual_model = run_configuration.get("base_model", {})
    expected_model = {
        "d_model": int(frozen_model["d_model"]),
        "num_heads": int(frozen_model["num_heads"]),
        "num_layers": int(frozen_model["num_layers"]),
        "decoder_num_layers": int(frozen_model["cross_layers"]),
        "dim_feedforward": int(frozen_model["dim_feedforward"]),
        "dropout": float(frozen_model["dropout"]),
        "prediction_head": "gaussian" if is_gaussian_model(model) else "point",
    }
    if any(actual_model.get(key) != value for key, value in expected_model.items()):
        raise ProtocolError("Arquitectura efectiva no coincide con el YAML congelado.")

    if run_configuration.get("protocol_config", {}).get("sha256") != preflight.get(
        "config", {}
    ).get("sha256"):
        raise ProtocolError("Run generado con otro YAML final.")
    expected_implementation = {
        str(record["path"]): str(record["sha256"])
        for record in preflight.get("implementation", ())
    }
    run_implementation = run_configuration.get("implementation", {})
    observed_implementation = {
        str(record["path"]): str(record["sha256"])
        for record in run_implementation.get("sources", ())
    }
    if observed_implementation != expected_implementation:
        raise ProtocolError("Run generado con otra implementación efectiva.")
    if run_implementation.get("environment") != preflight.get("runtime", {}).get(
        "versions"
    ):
        raise ProtocolError("Run generado con otras versiones numéricas.")
    repository = run_implementation.get("repository", {})
    expected_head = preflight.get("runtime", {}).get("repository", {}).get("git_head")
    if repository.get("git_commit") != expected_head or repository.get("worktree_dirty") is not False:
        raise ProtocolError("Run no procede del HEAD limpio aprobado por preflight.")

    dataset_record = next(
        (
            item
            for item in preflight.get("datasets", ())
            if (
                item.get("cohort"),
                item.get("kind"),
                item.get("preset"),
                item.get("dataset_id"),
            )
            == (cohort, kind, preset, dataset_id)
        ),
        None,
    )
    if dataset_record is None:
        raise ProtocolError("Dataset del run no fue aprobado por preflight.")
    expected_data = {
        str(record["path"]): str(record["sha256"])
        for name, record in dataset_record["sources"].items()
        if name in {"observations", "truth"}
    }
    observed_data = {
        str(record["path"]): str(record["sha256"])
        for record in run_configuration.get("dataset", {}).get("sources", ())
    }
    if observed_data != expected_data:
        raise ProtocolError("Run generado sobre otros bytes de datos.")
    if run_configuration.get("task", {}).get("cache_deterministic_history") is not True:
        raise ProtocolError("Run sin cache histórico determinista registrado.")
    evaluation = run_configuration.get("evaluation", {})
    expected_evaluation = {
        "save_predictions": True,
        "validate_only": False,
        "device": "cuda",
        "checkpoints": True,
        "deterministic": True,
    }
    if any(evaluation.get(key) != value for key, value in expected_evaluation.items()):
        raise ProtocolError("Ejecución/evaluación efectiva no es la final congelada.")
    if tuple(evaluation.get("timestamp_ablations", ())) != tuple(
        preflight.get("timestamp_ablations", ())
    ):
        raise ProtocolError("Run con un conjunto distinto de ablaciones.")
    if model != "Persistence":
        expected_metric = "val_nll" if is_gaussian_model(model) else "val_rmse"
        if run_configuration.get("training", {}).get("save_best_on") != expected_metric:
            raise ProtocolError(f"{model}: checkpoint no usa {expected_metric}.")


def audit_physical_completion(
    config: Mapping[str, Any],
    cohort: str,
    root: Path,
    *,
    preflight: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    physical_root = cohort_output_dir(config, cohort, root) / "physical_models"
    expected = _expected_result_keys(config, cohort)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for kind, preset, dataset_id, seed, model in sorted(expected):
        run_dir = physical_root / kind / preset / dataset_id / f"seed_{seed}" / model
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.is_file():
            failures.append(f"ausente: {run_config_path}")
            continue
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            fingerprint = str(run_config["fingerprint"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            failures.append(f"inválido: {run_config_path}")
            continue
        try:
            _verify_run_against_preflight(
                config=config,
                run_configuration=run_config["configuration"],
                preflight=preflight,
                cohort=cohort,
                kind=kind,
                preset=preset,
                dataset_id=dataset_id,
                model=model,
            )
        except (KeyError, TypeError, ProtocolError) as exc:
            failures.append(f"provenance incompatible: {run_dir}: {exc}")
            continue
        result = completed_run_result(
            run_dir, fingerprint, require_predictions=True
        )
        if result is None:
            failures.append(f"incompleto o SHA inválido: {run_dir}")
            continue
        observed = (
            str(result.get("Kind")),
            str(result.get("Preset")),
            str(result.get("Dataset_ID")),
            int(result.get("Seed", -1)),
            str(result.get("Model")),
        )
        if observed != (kind, preset, dataset_id, seed, model):
            failures.append(f"identidad incorrecta: {run_dir}")
            continue
        rows.append({"Cohort": cohort} | result)
    if failures:
        preview = "\n".join(failures[:20])
        raise ProtocolError(
            f"{cohort}: {len(failures)} runs incompletos de {len(expected)}.\n{preview}"
        )

    frame = pd.DataFrame(rows)
    actual = {
        (str(row.Kind), str(row.Preset), str(row.Dataset_ID), int(row.Seed), str(row.Model))
        for row in frame.itertuples()
    }
    if actual != expected or len(frame) != len(expected):
        raise ProtocolError(f"{cohort}: el conjunto de resultados no es exacto.")
    summary_path = physical_root / "benchmark_physical_models.csv"
    if not summary_path.is_file():
        raise ProtocolError(f"Falta resumen consolidado: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary_keys = {
        (str(row.Kind), str(row.Preset), str(row.Dataset_ID), int(row.Seed), str(row.Model))
        for row in summary.itertuples()
    }
    if summary_keys != expected or len(summary) != len(expected):
        raise ProtocolError(f"{cohort}: benchmark_physical_models.csv no es exacto.")
    return frame, {
        "cohort": cohort,
        "expected_runs": len(expected),
        "complete_runs": len(frame),
        "complete": True,
        "summary": file_provenance(summary_path),
    }


def audit_identifiability_completion(
    config: Mapping[str, Any],
    cohort: str,
    root: Path,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    output = cohort_output_dir(config, cohort, root) / "temporal_identifiability"
    metadata_path = output / "run_metadata.json"
    if not metadata_path.is_file():
        raise ProtocolError(f"Falta auditoría de identificabilidad: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {unit.key for unit in cohort_units(config, cohort)}
    observed = {
        (str(item["kind"]), str(item["preset"]), str(item["dataset_id"]))
        for item in metadata.get("datasets", [])
    }
    if observed != expected or int(metadata.get("n_dataset_units", -1)) != len(expected):
        raise ProtocolError(f"{cohort}: selección temporal incompleta o mezclada.")
    if metadata.get("config", {}).get("sha256") != preflight.get(
        "identifiability_config", {}
    ).get("sha256"):
        raise ProtocolError(f"{cohort}: auditoría temporal usa otro YAML.")
    expected_script = next(
        (
            record
            for record in preflight.get("implementation", ())
            if str(record.get("path", "")).endswith(
                "scripts/temporal_identifiability_benchmark.py"
            )
        ),
        None,
    )
    if expected_script is None or metadata.get("implementation", {}).get(
        "sha256"
    ) != expected_script.get("sha256"):
        raise ProtocolError(f"{cohort}: implementación temporal obsoleta.")
    temporal_runtime = metadata.get("runtime", {})
    expected_versions = preflight.get("runtime", {}).get("versions", {})
    observed_versions = temporal_runtime.get("environment", {})
    if any(
        observed_versions.get(key) != expected_versions.get(key)
        for key in observed_versions
        if key != "python_executable"
    ):
        raise ProtocolError(f"{cohort}: entorno numérico temporal obsoleto.")
    temporal_repository = temporal_runtime.get("repository", {})
    expected_head = preflight.get("runtime", {}).get("repository", {}).get(
        "git_head"
    )
    if (
        temporal_repository.get("git_commit") != expected_head
        or temporal_repository.get("worktree_dirty") is not False
    ):
        raise ProtocolError(f"{cohort}: auditoría temporal no proviene del HEAD limpio.")
    expected_dataset_sha = {
        (str(item["kind"]), str(item["preset"]), str(item["dataset_id"])): {
            str(record["path"]): str(record["sha256"])
            for name, record in item["sources"].items()
            if name in {"observations", "truth"}
        }
        for item in preflight.get("datasets", ())
        if item.get("cohort") == cohort
    }
    for item in metadata.get("datasets", ()):
        key = (str(item["kind"]), str(item["preset"]), str(item["dataset_id"]))
        observed_sha = {
            str(record["path"]): str(record["sha256"])
            for record in item.get("sources", ())
        }
        if observed_sha != expected_dataset_sha.get(key):
            raise ProtocolError(f"{cohort}: provenance temporal de datos obsoleta.")
    artifacts = metadata.get("artifacts", {})
    required_artifacts = {
        "counterfactual_examples.parquet",
        "control_predictions.parquet",
        "control_metrics.csv",
        "control_metrics_by_dataset.csv",
        "metrics_by_temporal_stratum.csv",
        "timestamp_precision.csv",
        "timestamp_ablation_manifest.csv",
    }
    if not isinstance(artifacts, Mapping) or not required_artifacts <= set(artifacts):
        raise ProtocolError(f"{cohort}: artefactos temporales incompletos.")
    for relative, record in artifacts.items():
        path = output / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ProtocolError(f"Artefacto temporal inválido: {path}")
    return {"cohort": cohort, "complete": True, "metadata": file_provenance(metadata_path)}


def aggregate_seed_to_dataset(
    rows: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> pd.DataFrame:
    keys = ["Cohort", "Kind", "Preset", "Dataset_ID", "Model"]
    metric_columns = [
        column
        for column in rows.columns
        if column in SUMMARY_METRICS or column.startswith("rmse_z_")
    ]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True, dropna=False):
        seeds = tuple(sorted(int(value) for value in group["Seed"].unique()))
        if seeds != tuple(sorted(expected_seeds)) or len(group) != len(expected_seeds):
            raise ProtocolError(f"Seeds incompletas para {key}: {seeds}")
        record: dict[str, Any] = dict(zip(keys, key))
        record["n_seeds"] = len(seeds)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(finite.mean()) if finite.size else math.nan
            record[f"{metric}_seed_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else math.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def pair_kinds_to_preset(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["Cohort", "Preset", "Model"]
    # RMSE/MAE en unidades originales sólo son comparables dentro de un dataset;
    # se conservan en seed_to_dataset.csv y no se promedian entre escalas.
    non_pairable = {"test_rmse_mean", "test_mae_mean"}
    mean_columns = [
        column
        for column in dataset_rows
        if column.endswith("_mean") and column not in non_pairable
    ]
    records: list[dict[str, Any]] = []
    for key, group in dataset_rows.groupby(keys, sort=True, dropna=False):
        if set(group["Kind"]) != {"univariate", "multivariate"} or len(group) != 2:
            raise ProtocolError(f"Par de kinds incompleto para {key}.")
        record: dict[str, Any] = dict(zip(keys, key))
        record["n_kinds"] = 2
        for metric in mean_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size not in {0, 2}:
                raise ProtocolError(f"{key}/{metric} está presente en un solo kind.")
            record[metric] = float(finite.mean()) if finite.size else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def macro_across_presets(preset_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["Cohort", "Model"]
    mean_columns = [column for column in preset_rows if column.endswith("_mean")]
    records: list[dict[str, Any]] = []
    for key, group in preset_rows.groupby(keys, sort=True, dropna=False):
        record: dict[str, Any] = dict(zip(keys, key))
        record["n_presets"] = int(group["Preset"].nunique())
        for metric in mean_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            base = metric[: -len("_mean")]
            record[f"{base}_macro"] = float(finite.mean()) if finite.size else math.nan
            record[f"{base}_preset_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else math.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def temporal_ablation_summary(
    preset_rows: pd.DataFrame, ablations: Sequence[str]
) -> pd.DataFrame:
    """Resume deltas emparejados dentro de cada preset, nunca filas sueltas."""
    records: list[dict[str, Any]] = []
    for row in preset_rows.itertuples(index=False):
        real = float(getattr(row, "rmse_z_real_mean"))
        for ablation in ablations:
            value = float(getattr(row, f"rmse_z_{ablation}_mean"))
            records.append(
                {
                    "Cohort": row.Cohort,
                    "Preset": row.Preset,
                    "Model": row.Model,
                    "Ablation": ablation,
                    "rmse_z": value,
                    "delta_vs_real": value - real,
                }
            )
    paired = pd.DataFrame.from_records(records)
    return (
        paired.groupby(["Cohort", "Model", "Ablation"], sort=True, as_index=False)
        .agg(
            rmse_z_macro=("rmse_z", "mean"),
            rmse_z_preset_sd=("rmse_z", "std"),
            delta_vs_real_macro=("delta_vs_real", "mean"),
            delta_vs_real_preset_sd=("delta_vs_real", "std"),
            n_presets=("Preset", "nunique"),
        )
    )


def aggregate_horizon_metrics(
    config: Mapping[str, Any], cohort: str, root: Path
) -> pd.DataFrame:
    physical_root = cohort_output_dir(config, cohort, root) / "physical_models"
    frames: list[pd.DataFrame] = []
    for unit in cohort_units(config, cohort):
        for seed in EXPECTED_SEEDS:
            for model in config["models"]:
                path = (
                    physical_root
                    / unit.kind
                    / unit.preset
                    / unit.dataset_id
                    / f"seed_{seed}"
                    / str(model)
                    / "metrics.csv"
                )
                frame = pd.read_csv(path)
                selected = frame[(frame["Ablation"] == "real") & (frame["Scope"] == "horizon")].copy()
                selected["Cohort"] = cohort
                frames.append(selected)
    rows = pd.concat(frames, ignore_index=True)
    first_keys = ["Cohort", "Kind", "Preset", "Dataset_ID", "Model", "Level"]
    seeded = (
        rows.groupby(first_keys, sort=True, as_index=False)
        .agg(rmse_z_mean=("rmse_z", "mean"), n_seeds=("Seed", "nunique"))
    )
    if not (seeded["n_seeds"] == len(EXPECTED_SEEDS)).all():
        raise ProtocolError(f"{cohort}: horizonte con seeds incompletas.")
    paired_records = []
    for key, group in seeded.groupby(["Cohort", "Preset", "Model", "Level"], sort=True):
        if set(group["Kind"]) != {"univariate", "multivariate"} or len(group) != 2:
            raise ProtocolError(f"Par temporal incompleto: {key}")
        paired_records.append(
            dict(zip(["Cohort", "Preset", "Model", "Level"], key))
            | {"rmse_z_mean": float(group["rmse_z_mean"].mean())}
        )
    paired = pd.DataFrame(paired_records)
    return (
        paired.groupby(["Cohort", "Model", "Level"], sort=True, as_index=False)
        .agg(
            rmse_z_macro=("rmse_z_mean", "mean"),
            rmse_z_preset_sd=("rmse_z_mean", "std"),
            n_presets=("Preset", "nunique"),
        )
    )


def consolidate_stratified_metrics(
    config: Mapping[str, Any], cohort: str, root: Path
) -> pd.DataFrame:
    """Índice dataset-level de estratos; no mezcla canales ni kinds incompatibles."""
    scopes = {
        "horizon",
        "density_bin",
        "max_gap_bin",
        "last_observation_age_bin",
        "channel_density_bin",
        "channel_max_gap_bin",
        "channel_last_age_bin",
    }
    physical_root = cohort_output_dir(config, cohort, root) / "physical_models"
    frames: list[pd.DataFrame] = []
    for unit in cohort_units(config, cohort):
        for seed in EXPECTED_SEEDS:
            for model in config["models"]:
                path = (
                    physical_root
                    / unit.kind
                    / unit.preset
                    / unit.dataset_id
                    / f"seed_{seed}"
                    / str(model)
                    / "metrics.csv"
                )
                frame = pd.read_csv(path)
                selected = frame[frame["Scope"].isin(scopes)].copy()
                selected["Cohort"] = cohort
                frames.append(selected)
    rows = pd.concat(frames, ignore_index=True)
    keys = [
        "Cohort",
        "Kind",
        "Preset",
        "Dataset_ID",
        "Model",
        "Ablation",
        "Scope",
        "Level",
    ]
    metrics = [
        column
        for column in (
            "rmse_z",
            "mae_z",
            "bias_z",
            "rmse",
            "mae",
            "bias",
            "nll_z",
            "crps_z",
            "coverage_90",
            "coverage_95",
            "mean_scale_z",
        )
        if column in rows
    ]
    aggregate = {column: (column, "mean") for column in metrics}
    aggregate["n_per_seed_mean"] = ("n", "mean")
    aggregate["n_seeds"] = ("Seed", "nunique")
    result = rows.groupby(keys, sort=True, as_index=False, dropna=False).agg(**aggregate)
    if not (result["n_seeds"] == len(EXPECTED_SEEDS)).all():
        raise ProtocolError(f"{cohort}: estratos con seeds incompletas.")
    return result


def _latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(old, new)
    return text


def _metric_text(mean: Any, sd: Any | None = None, digits: int = 3) -> str:
    try:
        mean_value = float(mean)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(mean_value):
        return "--"
    if sd is None:
        return f"{mean_value:.{digits}f}"
    try:
        sd_value = float(sd)
    except (TypeError, ValueError):
        sd_value = math.nan
    if not np.isfinite(sd_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} $\\pm$ {sd_value:.{digits}f}"


def _table(
    *, caption: str, label: str, columns: Sequence[str], rows: Iterable[Sequence[str]], alignment: str | None = None
) -> str:
    alignment = alignment or "l" + "c" * (len(columns) - 1)
    body = [
        "% Generado por scripts/run_thesis_physical_benchmark.py; no editar a mano.",
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(body)


def write_latex_tables(
    *,
    latex_dir: Path,
    completion: Sequence[Mapping[str, Any]],
    main_macro: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> dict[str, Path]:
    latex_dir.mkdir(parents=True, exist_ok=True)
    status_rows = [
        (
            _latex_escape(item["cohort"]),
            str(item["dataset_units"]),
            str(item["generator_seed"]),
            str(item["expected_runs"]),
            str(item["complete_runs"]),
        )
        for item in completion
    ]
    contents: dict[str, str] = {
        "protocol_summary.tex": _table(
            caption="Cobertura y completitud del benchmark físico final.",
            label="tab:physical-final-status",
            columns=("Cohorte", "Datasets", "Seed generador", "Esperadas", "Completas"),
            rows=status_rows,
        )
    }

    point = main_macro[~main_macro["Model"].map(is_gaussian_model)].sort_values(
        "test_rmse_z_macro"
    )
    point_rows = []
    for row in point.itertuples():
        point_rows.append(
            (
                _latex_escape(row.Model),
                _metric_text(row.test_rmse_z_macro, row.test_rmse_z_preset_sd),
                _metric_text(row.test_mae_z_macro, row.test_mae_z_preset_sd),
                str(row.n_presets),
            )
        )
    contents["model_results.tex"] = _table(
        caption=(
            "Desempeño puntual macro: primero se promedian semillas por dataset, "
            "luego los kinds emparejados y finalmente los presets."
        ),
        label="tab:physical-final-point",
        columns=("Modelo", "RMSE-z", "MAE-z", "Presets"),
        rows=point_rows,
    )

    temporal_tables = []
    temporal_models = sorted(temporal_summary["Model"].unique())
    # Un tabular único de 72 filas no puede partir página. Tres modelos por
    # bloque mantienen cada float acotado sin requerir el paquete longtable.
    for block_index in range(0, len(temporal_models), 3):
        model_block = temporal_models[block_index : block_index + 3]
        selected = temporal_summary[temporal_summary["Model"].isin(model_block)]
        temporal_rows = [
            (
                _latex_escape(row.Model),
                _latex_escape(row.Ablation),
                _metric_text(row.rmse_z_macro, row.rmse_z_preset_sd),
                _metric_text(row.delta_vs_real_macro, row.delta_vs_real_preset_sd),
            )
            for row in selected.sort_values(["Model", "Ablation"]).itertuples()
        ]
        temporal_tables.append(
            _table(
                caption=(
                    "Ablaciones temporales emparejadas (bloque "
                    f"{block_index // 3 + 1}). Cada delta se calcula dentro del "
                    "preset después de promediar semillas y emparejar kinds."
                ),
                label=(
                    "tab:physical-final-temporal-ablations-"
                    f"{block_index // 3 + 1}"
                ),
                columns=("Modelo", "Ablación", "RMSE-z", r"$\Delta$ vs. real"),
                rows=temporal_rows,
            )
        )
    contents["temporal_ablations.tex"] = "\n\n".join(temporal_tables)
    horizon_values = sorted(
        {float(value) for value in horizon_summary["Level"].unique()}
    )
    horizon_rows = []
    for model, group in horizon_summary.groupby("Model", sort=True):
        indexed = {
            float(row.Level): row for row in group.itertuples(index=False)
        }
        horizon_rows.append(
            (_latex_escape(model),)
            + tuple(
                _metric_text(
                    indexed[value].rmse_z_macro,
                    indexed[value].rmse_z_preset_sd,
                )
                for value in horizon_values
            )
        )
    contents["temporal_ablations.tex"] += "\n\n" + _table(
        caption="RMSE-z macro por horizonte físico en la cohorte principal.",
        label="tab:physical-final-horizons",
        columns=("Modelo",)
        + tuple(rf"$h={value:g}$" for value in horizon_values),
        rows=horizon_rows,
    )

    gaussian = main_macro[main_macro["Model"].map(is_gaussian_model)].sort_values("Model")
    probabilistic_rows = []
    for row in gaussian.itertuples():
        probabilistic_rows.append(
            (
                _latex_escape(row.Model),
                _metric_text(row.test_nll_z_macro, row.test_nll_z_preset_sd),
                _metric_text(row.test_crps_z_macro, row.test_crps_z_preset_sd),
                _metric_text(row.test_mean_scale_z_macro, row.test_mean_scale_z_preset_sd),
                _metric_text(row.test_coverage_90_macro, digits=3),
                _metric_text(row.test_coverage_95_macro, digits=3),
            )
        )
    contents["gaussian_calibration.tex"] = _table(
        caption="Métricas probabilísticas macro de las cabezas gaussianas.",
        label="tab:physical-final-probabilistic",
        columns=("Modelo", "NLL-z", "CRPS-z", r"$\bar\sigma_z$", "Cov. 90", "Cov. 95"),
        rows=probabilistic_rows,
    )

    if tuple(contents) != LATEX_TABLE_FILENAMES:
        raise ProtocolError("El generador no produjo exactamente las cuatro tablas pactadas.")
    written: dict[str, Path] = {}
    for name in LATEX_TABLE_FILENAMES:
        path = latex_dir / name
        path.write_text(contents[name], encoding="utf-8", newline="\n")
        written[name] = path
    return written


def generate_report(
    config: Mapping[str, Any],
    *,
    root: Path,
    cohorts: Sequence[str] = COHORT_NAMES,
    require_identifiability: bool = True,
) -> dict[str, Any]:
    if set(cohorts) != set(COHORT_NAMES):
        raise ProtocolError("El reporte final exige main y stress completos y separados.")
    if not require_identifiability:
        raise ProtocolError("El reporte final no permite omitir identificabilidad.")
    preflight = require_compatible_preflight(config, root=root, cohorts=cohorts)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    completion_records = []
    macro_by_cohort: dict[str, pd.DataFrame] = {}
    preset_by_cohort: dict[str, pd.DataFrame] = {}
    source_records: dict[str, Any] = {}
    for cohort in cohorts:
        rows, completion = audit_physical_completion(
            config, cohort, root, preflight=preflight
        )
        if require_identifiability and bool(config["identifiability"].get("enabled", True)):
            completion["identifiability"] = audit_identifiability_completion(
                config, cohort, root, preflight=preflight
            )
        units = cohort_units(config, cohort)
        completion.update(
            dataset_units=len(units),
            generator_seed=units[0].generator_seed,
        )
        completion_records.append(completion)
        seed_rows = aggregate_seed_to_dataset(rows)
        preset_rows = pair_kinds_to_preset(seed_rows)
        macro_rows = macro_across_presets(preset_rows)
        seed_path = reports_dir / f"{cohort}_seed_to_dataset.csv"
        preset_path = reports_dir / f"{cohort}_paired_preset.csv"
        macro_path = reports_dir / f"{cohort}_model_macro.csv"
        seed_rows.to_csv(seed_path, index=False)
        preset_rows.to_csv(preset_path, index=False)
        macro_rows.to_csv(macro_path, index=False)
        macro_by_cohort[cohort] = macro_rows
        preset_by_cohort[cohort] = preset_rows
        source_records[cohort] = {
            "seed_to_dataset": file_provenance(seed_path),
            "paired_preset": file_provenance(preset_path),
            "model_macro": file_provenance(macro_path),
        }
        strata = consolidate_stratified_metrics(config, cohort, root)
        strata_path = reports_dir / f"{cohort}_stratified_seed_to_dataset.csv"
        strata.to_csv(strata_path, index=False)
        source_records[cohort]["stratified_seed_to_dataset"] = file_provenance(
            strata_path
        )
    if "main" not in macro_by_cohort:
        raise ProtocolError("Las tablas finales requieren la cohorte main completa.")
    temporal_ablations = tuple(config["evaluation"]["timestamp_ablations"])
    temporal_summary = temporal_ablation_summary(
        preset_by_cohort["main"], temporal_ablations
    )
    temporal_summary_path = reports_dir / "main_temporal_ablation_deltas.csv"
    temporal_summary.to_csv(temporal_summary_path, index=False)
    horizon_summary = aggregate_horizon_metrics(config, "main", root)
    horizon_summary_path = reports_dir / "main_horizon_macro.csv"
    horizon_summary.to_csv(horizon_summary_path, index=False)
    tables = write_latex_tables(
        latex_dir=root / "latex",
        completion=completion_records,
        main_macro=macro_by_cohort["main"],
        temporal_summary=temporal_summary,
        horizon_summary=horizon_summary,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "config": file_provenance(Path(config["_config_path"])),
        "aggregation_order": list(config["reporting"]["aggregation_order"]),
        "completion": completion_records,
        "aggregates": source_records
        | {
            "main_temporal_ablation_deltas": file_provenance(temporal_summary_path),
            "main_horizon_macro": file_provenance(horizon_summary_path),
        },
        "latex_tables": {name: file_provenance(path) for name, path in tables.items()},
    }
    manifest["report_fingerprint"] = _mapping_sha256(manifest)
    atomic_write_json(root / "report_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "dry-run", "smoke", "run", "report", "all"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cohorts", nargs="+", choices=COHORT_NAMES, default=list(COHORT_NAMES))
    parser.add_argument(
        "--skip-identifiability",
        action="store_true",
        help="Ejecuta/reporta sólo modelos físicos; no usar para el cierre de tesis.",
    )
    parser.add_argument(
        "--no-data-hash",
        action="store_true",
        help="Sólo para dry-run rápido; preflight/run final siempre calculan SHA-256.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_final_config(args.config)
    root = output_root(config, args.output_dir)
    cohorts = tuple(dict.fromkeys(args.cohorts))
    include_identifiability = not args.skip_identifiability
    if args.skip_identifiability and args.command in {"report", "all"}:
        raise ProtocolError(
            "--skip-identifiability no es válido para report/all del cierre final."
        )

    if args.command == "dry-run":
        manifest = preflight_manifest(
            config,
            root=root,
            cohorts=cohorts,
            hash_files=not args.no_data_hash,
            strict=False,
            run_checks=False,
        )
        print(
            f"Preflight lógico OK: {manifest['n_dataset_units']} unidades; "
            f"SHA-256={'sí' if not args.no_data_hash else 'omitido sólo en dry-run'}."
        )
        for spec in build_run_commands(
            config,
            root=root,
            cohorts=cohorts,
            include_identifiability=include_identifiability,
        ):
            print(f"[{spec.cohort}/{spec.protocol}] {command_text(spec.command)}")
        return

    if args.command == "smoke":
        # El smoke valida contratos y columnas, pero evita leer gigabytes sólo
        # para hashear; los runs conservan su propia provenance SHA-256.
        preflight_manifest(
            config,
            root=root,
            cohorts=("main",),
            hash_files=False,
            strict=False,
            run_checks=True,
            focal_tests=True,
        )
        execute_commands(
            build_run_commands(
                config,
                root=root,
                include_identifiability=include_identifiability,
                smoke=True,
            )
        )
        print(f"Smoke completo: {root / 'smoke'}")
        return

    if args.command in {"preflight", "all"}:
        if args.no_data_hash:
            raise ProtocolError("--no-data-hash sólo está permitido con dry-run.")
        manifest = preflight_manifest(
            config,
            root=root,
            cohorts=cohorts,
            hash_files=True,
            strict=True,
            run_checks=True,
        )
        path = write_preflight_manifest(manifest, root)
        print(f"Preflight completo: {path}")
        if args.command == "preflight":
            return

    if args.command in {"run", "all"}:
        require_compatible_preflight(config, root=root, cohorts=cohorts)
        execute_commands(
            build_run_commands(
                config,
                root=root,
                cohorts=cohorts,
                include_identifiability=include_identifiability,
            )
        )
        if args.command == "run":
            print("Runs terminados. Ejecute el subcomando report para consolidar.")
            return

    if args.command in {"report", "all"}:
        manifest = generate_report(
            config,
            root=root,
            cohorts=cohorts,
            require_identifiability=include_identifiability,
        )
        print(
            "Reporte completo: "
            f"{root / 'report_manifest.json'} ({manifest['report_fingerprint']})"
        )


if __name__ == "__main__":
    main()
