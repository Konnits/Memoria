from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from scripts.tune_physical_optuna import (
    build_protocol_payload,
    completed_trial_count,
    ensure_study_protocol,
    make_tuning_loaders,
    parse_args,
    protocol_fingerprint,
    sample_model_and_training,
    stop_after_completed_trials,
)
from ts_transformer.models import TimeSeriesContinuousBasisDecoder


def test_protocol_fingerprint_changes_when_the_physical_task_changes(tmp_path: Path) -> None:
    observations = tmp_path / "observations.parquet"
    truth = tmp_path / "truth.parquet"
    observations.write_bytes(b"observations")
    truth.write_bytes(b"truth")
    args = argparse.Namespace(
        epochs=2,
        early_stopping_patience=1,
        max_train_samples=8,
        max_val_samples=4,
        batch_size=2,
        max_observation_rows_per_split=None,
        seed=7,
    )
    raw = {
        "task": {
            "horizons": [0.25, 1.0],
            "train_horizon_range": [0.25, 1.0],
            "train_horizon_sampling": "log_uniform",
            "queries_per_sample": 2,
            "history_duration": 4.0,
            "max_history_events_univariate": 32,
            "max_history_events_multivariate": 64,
            "history_subsampling": "uniform_time",
        }
    }
    specs = [("univariate", "probe", observations, truth)]
    first_payload = build_protocol_payload(args, raw, specs)
    assert first_payload["implementation"]["fingerprint"]
    assert set(first_payload["datasets"][0]["observations"]) == {
        "path",
        "size",
        "sha256",
    }
    assert len(first_payload["datasets"][0]["observations"]["sha256"]) == 64
    assert any(
        item["path"].endswith("scripts/tune_physical_optuna.py")
        for item in first_payload["implementation"]["sources"]
    )
    yaml.safe_dump(first_payload)
    first = protocol_fingerprint(first_payload)
    raw["task"]["history_duration"] = 8.0
    second = protocol_fingerprint(build_protocol_payload(args, raw, specs))
    assert first != second


def test_protocol_fingerprint_changes_when_dataset_bytes_change(tmp_path: Path) -> None:
    observations = tmp_path / "observations.parquet"
    truth = tmp_path / "truth.parquet"
    observations.write_bytes(b"observations")
    truth.write_bytes(b"truth")
    args = argparse.Namespace(
        epochs=1,
        early_stopping_patience=1,
        max_train_samples=8,
        max_val_samples=4,
        batch_size=2,
        max_observation_rows_per_split=None,
        seed=7,
    )
    raw = {
        "task": {
            "horizons": [0.25],
            "train_horizon_range": [0.25, 1.0],
            "history_duration": 4.0,
            "max_history_events_univariate": 32,
            "max_history_events_multivariate": 64,
        }
    }
    specs = [("univariate", "probe", observations, truth)]
    first = protocol_fingerprint(build_protocol_payload(args, raw, specs))

    # Mismo path y mismo tamaño: la identidad debe depender del contenido.
    observations.write_bytes(b"OBSERVATIONS")
    second_payload = build_protocol_payload(args, raw, specs)

    assert second_payload["datasets"][0]["observations"]["size"] == 12
    assert first != protocol_fingerprint(second_payload)


def test_physical_tuning_uses_all_matching_datasets_by_default() -> None:
    assert parse_args([]).limit_datasets_per_kind is None
    assert (
        parse_args(["--limit-datasets-per-kind", "1"]).limit_datasets_per_kind
        == 1
    )


def test_study_rejects_an_incompatible_protocol() -> None:
    study = optuna.create_study(direction="minimize")
    ensure_study_protocol(study, "abc", {"task": 1})
    ensure_study_protocol(study, "abc", {"task": 1})
    with pytest.raises(RuntimeError, match="otro protocolo"):
        ensure_study_protocol(study, "def", {"task": 2})


def test_tuning_loaders_do_not_access_test_split() -> None:
    class TestForbidden:
        train = TensorDataset(torch.arange(4))
        validation = TensorDataset(torch.arange(2))

        @property
        def test(self):
            raise AssertionError("El objetivo de tuning no debe acceder a test.")

    # Este test usa un collate trivial porque TensorDataset no implementa el
    # contrato temporal; lo relevante es que construir los loaders no toca test.
    original_init = make_tuning_loaders.__globals__["PhysicalCollate"]
    make_tuning_loaders.__globals__["PhysicalCollate"] = lambda: (lambda batch: batch)
    try:
        train, validation = make_tuning_loaders(
            TestForbidden(),
            batch_size=2,
            num_workers=0,
            seed=3,
            device="cpu",
        )
    finally:
        make_tuning_loaders.__globals__["PhysicalCollate"] = original_init
    assert len(train.dataset) == 4
    assert len(validation.dataset) == 2


def test_physical_search_can_instantiate_the_continuous_basis_branch() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "architecture": "d32_h2",
            "decoder_architecture": "continuous_basis",
            "encoder_layers": 1,
            "ffn_multiplier": 2,
            "dropout": 0.0,
            "time_encoding_mode": "sinusoidal",
            "time_transform": "linear",
            "use_history_time_encoding": True,
            "use_ctssm": False,
            "basis_trend_degree": 2,
            "basis_num_rbf": 4,
            "basis_num_fourier": 2,
            "basis_min_scale": 0.25,
            "basis_max_scale": 16.0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        }
    )
    data = SimpleNamespace(kind="univariate", n_channels=1)
    args = argparse.Namespace(epochs=1, device="cpu", early_stopping_patience=1)

    model, training = sample_model_and_training(trial, data, args)

    assert isinstance(model, TimeSeriesContinuousBasisDecoder)
    assert training.num_epochs == 1
    assert "cross_layers" not in trial.params


def test_completion_budget_ignores_running_and_pruned_trials() -> None:
    study = optuna.create_study(direction="minimize")
    study.ask()  # Simula un trial RUNNING huérfano de una ejecución interrumpida.

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 1:
            raise optuna.TrialPruned("probe")
        return float(trial.number)

    study.optimize(
        objective,
        n_trials=10,
        callbacks=[stop_after_completed_trials(2)],
    )

    assert completed_trial_count(study) == 2
    assert sum(t.state == optuna.trial.TrialState.RUNNING for t in study.trials) == 1
    assert sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials) == 1
    assert len(study.trials) == 4
