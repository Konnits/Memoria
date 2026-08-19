from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_thesis_physical_benchmark as thesis_runner
from scripts.run_thesis_physical_benchmark import (
    DEFAULT_CONFIG,
    CommandSpec,
    DatasetUnit,
    EXPECTED_SEEDS,
    LATEX_TABLE_FILENAMES,
    ProtocolError,
    aggregate_seed_to_dataset,
    build_run_commands,
    cohort_units,
    consolidate_physical_shards,
    consolidate_stratified_metrics,
    cuda_inventory,
    execute_commands,
    generate_report,
    implementation_paths,
    load_final_config,
    macro_across_presets,
    pair_kinds_to_preset,
    require_compatible_preflight,
    temporal_ablation_summary,
    write_latex_tables,
    _verify_identifiability_metadata_contract,
    _verify_run_against_preflight,
)


def test_frozen_protocol_selects_exact_main_and_stress_units() -> None:
    config = load_final_config(DEFAULT_CONFIG)
    main = cohort_units(config, "main")
    stress = cohort_units(config, "stress")

    assert config["protocol_status"] == "thesis_physical_evaluation_v3_frozen"
    assert config["output_dir"] == "experiments/thesis_physical_benchmark_v3"
    assert "retrospective" in config["claim_boundary"]
    assert tuple(config["training"]["seeds"]) == EXPECTED_SEEDS
    assert config["training"]["batch_size"] == 32
    assert config["training"]["device"] == "cuda"
    assert config["training"]["num_workers"] == 0
    assert config["training"]["deterministic"] is False
    assert config["execution"]["fast_kernels"] is True
    assert config["execution"]["physical_execution_mode"] == "sequential"
    assert config["execution"]["parallel_physical_processes"] == 1
    assert config["task"]["cache_deterministic_history"] is True
    assert len(main) == 16
    assert len(stress) == 2
    assert [
        unit.key for unit in sorted(main, key=lambda item: item.protocol_index)
    ] == sorted(unit.key for unit in main)
    assert sorted(unit.protocol_index for unit in main) == list(range(16))
    assert sorted(unit.protocol_index for unit in stress) == [0, 1]
    assert {unit.shard for unit in (*main, *stress)} == {0}
    assert all(unit.generator_seed == 2026 for unit in main)
    assert all(unit.dataset_id == f"{unit.preset}_0000" for unit in main)
    assert all("gseed" not in unit.dataset_id for unit in main)
    assert {unit.kind for unit in stress} == {"univariate", "multivariate"}
    assert {unit.generator_seed for unit in stress} == {3031}
    assert {unit.dataset_id for unit in stress} == {
        "long_gaps_gseed3031_0000"
    }
    assert tuple(config["reporting"]["latex_tables"]) == LATEX_TABLE_FILENAMES
    assert config["preflight"]["required_conda_environment"] == "memoria"
    assert config["preflight"]["require_clean_worktree"] is True
    assert config["preflight"]["full_pytest"] is True
    source_files = set((DEFAULT_CONFIG.parents[2] / "src" / "ts_transformer").rglob("*.py"))
    assert source_files <= set(implementation_paths())


def test_frozen_protocol_rejects_parallel_physical_execution() -> None:
    config = load_final_config(DEFAULT_CONFIG)
    config["execution"]["parallel_physical_processes"] = 2

    with pytest.raises(ProtocolError, match="parallel_physical_processes: 1"):
        thesis_runner.validate_final_config(config)


def test_run_requires_a_prior_compatible_preflight(tmp_path: Path) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    with pytest.raises(ProtocolError, match="Ejecute primero"):
        require_compatible_preflight(
            config, root=tmp_path, cohorts=("main", "stress")
        )


def test_final_report_cannot_skip_identifiability(tmp_path: Path) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    with pytest.raises(ProtocolError, match="no permite omitir"):
        generate_report(
            config,
            root=tmp_path,
            cohorts=("main", "stress"),
            require_identifiability=False,
        )


def _nominal_effective_configuration(config: dict) -> dict:
    task = config["task"]
    training = config["training"]
    return {
        "task": {
            "horizons": list(task["horizons"]),
            "train_horizon_range": list(task["train_horizon_range"]),
            "train_horizon_sampling": task["train_horizon_sampling"],
            "queries_per_sample": task["queries_per_sample"],
            "history_duration": task["history_duration"],
            "max_history_events": task["max_history_events_multivariate"],
            "history_subsampling": task["history_subsampling"],
            "cache_deterministic_history": True,
            "max_observation_rows_per_split": None,
            "protocol_seed": config["sampling"]["protocol_seed"],
            "protocol_index": 0,
        },
        "sampling": {
            "max_train_samples": config["sampling"]["max_train_samples"],
            "max_val_samples": config["sampling"]["max_val_samples"],
            "max_test_samples": config["sampling"]["max_test_samples"],
            "batch_size": 32,
            "num_workers": 0,
        },
        "training": {
            "num_epochs": training["epochs"],
            "device": "cuda",
            "loss_name": "mse",
            "early_stopping_patience": training["early_stopping_patience"],
            "restore_best_weights": True,
            "use_amp": True,
            "enable_cuda_runtime_optimizations": True,
            "optimizer_config": {
                "optimizer_name": "adamw",
                "lr": training["learning_rate"],
                "weight_decay": training["weight_decay"],
                "scheduler_name": "cosine",
                "scheduler_T_max": training["epochs"],
            },
        },
    }


@pytest.mark.parametrize("override", ["epochs", "sample_cap"])
def test_report_rejects_cli_overrides_despite_same_yaml_hash(override: str) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    effective = _nominal_effective_configuration(config)
    if override == "epochs":
        effective["training"]["num_epochs"] = 1
    else:
        effective["sampling"]["max_train_samples"] = 8

    with pytest.raises(ProtocolError, match="efectiv|Sampling"):
        _verify_run_against_preflight(
            config=config,
            run_configuration=effective,
            preflight={},
            cohort="main",
            kind="multivariate",
            preset="bursty",
            dataset_id="bursty_0000",
            model="QueryCross",
        )


def _nominal_identifiability_metadata() -> tuple[dict, dict]:
    versions = {
        "python_implementation": "CPython",
        "python": "3.test",
        "numpy": "numpy.test",
        "pandas": "pandas.test",
        "pyarrow": "pyarrow.test",
        "pyyaml": "pyyaml.test",
    }
    executable = str(Path("python-test").resolve())
    preflight = {
        "runtime": {
            "python_executable": executable,
            "versions": versions,
            "repository": {"git_head": "abc123"},
        }
    }
    metadata = {
        "seed": 2026,
        "n_train_anchors_per_dataset": 64,
        "n_eval_anchors_per_dataset": 32,
        "queries_per_anchor": 4,
        "query_slots_randomized": True,
        "physical_horizon_range": [0.25, 8.0],
        "observation_scan_skipped": False,
        "controls": ["Persistence", "Ordinal", "ExplicitHorizon"],
        "control_history_values": "clean_truth_oracle",
        "timestamp_ablations": [
            "real",
            "all_equal",
            "permuted_gaps",
            "regular_grid",
            "ordinal",
            "query_only",
            "history_only",
        ],
        "runtime": {
            "environment": versions | {"python_executable": executable},
            "repository": {"git_commit": "abc123", "worktree_dirty": False},
        },
    }
    return metadata, preflight


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("n_train_anchors_per_dataset", 4),
        ("n_eval_anchors_per_dataset", 2),
        ("observation_scan_skipped", True),
        ("query_slots_randomized", False),
    ],
)
def test_report_rejects_reduced_or_altered_identifiability_contract(
    field: str,
    invalid_value: object,
) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    metadata, preflight = _nominal_identifiability_metadata()
    altered = deepcopy(metadata)
    altered[field] = invalid_value

    with pytest.raises(ProtocolError, match=field):
        _verify_identifiability_metadata_contract(
            config=config,
            metadata=altered,
            preflight=preflight,
        )


def test_consolidated_strata_preserve_channel_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = DatasetUnit(
        kind="multivariate",
        preset="test_preset",
        dataset_id="test_preset_0000",
        generator_seed=2026,
        protocol_index=0,
        shard=0,
    )
    monkeypatch.setattr(thesis_runner, "EXPECTED_SEEDS", (42,))
    monkeypatch.setattr(
        thesis_runner,
        "cohort_units",
        lambda _config, _cohort: (unit,),
    )
    config = {
        "dataset_cohorts": {"main": {"output_subdir": "main"}},
        "models": ["QueryCross"],
    }
    metrics_path = (
        tmp_path
        / "main"
        / "physical_models"
        / unit.kind
        / unit.preset
        / unit.dataset_id
        / "seed_42"
        / "QueryCross"
        / "metrics.csv"
    )
    metrics_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Kind": unit.kind,
                "Preset": unit.preset,
                "Dataset_ID": unit.dataset_id,
                "Seed": 42,
                "Model": "QueryCross",
                "Ablation": "real",
                "Scope": "channel",
                "Level": "0",
                "n": 16,
                "rmse_z": 0.5,
            }
        ]
    ).to_csv(metrics_path, index=False)

    consolidated = consolidate_stratified_metrics(config, "main", tmp_path)

    assert consolidated[["Scope", "Level"]].to_dict("records") == [
        {"Scope": "channel", "Level": 0}
    ]
    assert consolidated.loc[0, "rmse_z"] == pytest.approx(0.5)
    assert consolidated.loc[0, "n_seeds"] == 1


def test_full_commands_keep_cohorts_outputs_and_dataset_ids_separate(
    tmp_path: Path,
) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    commands = build_run_commands(config, root=tmp_path)

    main_physical = commands[:1]
    main_ident = commands[1]
    stress_physical = commands[2:3]
    stress_ident = commands[3]
    assert len(main_physical) == 1
    assert len(stress_physical) == 1
    assert all(item.cohort == "main" and item.protocol == "physical_models" for item in main_physical)
    assert (main_ident.cohort, main_ident.protocol) == ("main", "temporal_identifiability")
    assert all(item.cohort == "stress" and item.protocol == "physical_models" for item in stress_physical)
    assert (stress_ident.cohort, stress_ident.protocol) == ("stress", "temporal_identifiability")
    assert len(commands) == 4
    assert {item.shard for item in main_physical} == {0}
    assert {item.shard for item in stress_physical} == {0}
    assert all(
        all("long_gaps_gseed3031_0000" not in argument for argument in item.command)
        for item in main_physical
    )
    assert all(
        any("long_gaps_gseed3031_0000" in argument for argument in item.command)
        for item in stress_physical
    )
    assert {item.output_dir for item in main_physical}.isdisjoint(
        {item.output_dir for item in stress_physical}
    )
    selected_main_units = []
    for item in main_physical:
        command = item.command
        assert all(str(seed) in command for seed in EXPECTED_SEEDS)
        assert "--non-deterministic" in command
        assert "--deterministic" not in command
        assert "--kinds" not in command
        assert "--presets" not in command
        assert "--dataset-ids" not in command
        assert command[command.index("--batch-size") + 1] == "32"
        assert command[command.index("--device") + 1] == "cuda"
        selected_main_units.extend(
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--dataset-unit"
        )
        assert item.stdout_log != item.stderr_log
        assert item.stdout_log is not None
        assert item.stderr_log is not None
        assert item.stdout_log.name == "physical_models_shard_0.stdout.log"
        assert item.stderr_log.name == "physical_models_shard_0.stderr.log"
    assert len(selected_main_units) == 16
    parsed_indices = sorted(int(value.rsplit(":", 1)[1]) for value in selected_main_units)
    assert parsed_indices == list(range(16))


def test_failed_sequential_physical_run_stops_serial_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial_calls: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs) -> None:
            self.command = tuple(command)
            self.returncode = 7

        def wait(self, timeout=None) -> int:
            del timeout
            return self.returncode

        def poll(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -1

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(thesis_runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        thesis_runner.subprocess,
        "run",
        lambda command, **_kwargs: serial_calls.append(tuple(command)),
    )
    physical = CommandSpec(
        cohort="main",
        protocol="physical_models",
        command=("python", "shard_0"),
        output_dir=tmp_path / "shard_0",
        shard=0,
        stdout_log=tmp_path / "shard_0.out.log",
        stderr_log=tmp_path / "shard_0.err.log",
    )
    identifiability = CommandSpec(
        cohort="main",
        protocol="temporal_identifiability",
        command=("python", "identifiability"),
        output_dir=tmp_path / "identifiability",
    )

    with pytest.raises(ProtocolError, match="no se consolida ni publica"):
        execute_commands([physical, identifiability])

    assert serial_calls == []


def test_sequential_physical_emits_heartbeat_and_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DelayedProcess:
        def __init__(self, _command, **_kwargs) -> None:
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def wait(self, timeout=None) -> int:
            del timeout
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    clock = iter((0.0, 31.0))
    monkeypatch.setattr(thesis_runner.subprocess, "Popen", DelayedProcess)
    monkeypatch.setattr(thesis_runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(thesis_runner.time, "sleep", lambda _seconds: None)
    specs = [
        CommandSpec(
            cohort="main",
            protocol="physical_models",
            command=("python", "shard_0"),
            output_dir=tmp_path / "shard_0",
            shard=0,
            stdout_log=tmp_path / "shard_0.out.log",
            stderr_log=tmp_path / "shard_0.err.log",
        )
    ]

    thesis_runner._execute_sequential_physical(specs)

    output = capsys.readouterr().out
    assert "heartbeat: activos=shard_0; completos=-" in output
    assert "shard_0] terminado (returncode=0)" in output


def test_keyboard_interrupt_terminates_only_active_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children = []

    class InterruptibleProcess:
        def __init__(self, _command, **_kwargs) -> None:
            self.returncode = None
            self.terminate_calls = 0
            self.wait_calls = 0
            self.kill_calls = 0
            children.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None) -> int:
            del timeout
            self.wait_calls += 1
            return int(self.returncode or 0)

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = -15

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

    monkeypatch.setattr(thesis_runner.subprocess, "Popen", InterruptibleProcess)
    monkeypatch.setattr(thesis_runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        thesis_runner.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    specs = [
        CommandSpec(
            cohort="main",
            protocol="physical_models",
            command=("python", "shard_0"),
            output_dir=tmp_path / "shard_0",
            shard=0,
            stdout_log=tmp_path / "shard_0.out.log",
            stderr_log=tmp_path / "shard_0.err.log",
        )
    ]

    with pytest.raises(KeyboardInterrupt):
        thesis_runner._execute_sequential_physical(specs)

    (active,) = children
    assert active.terminate_calls == 1
    assert active.wait_calls == 1
    assert active.kill_calls == 0


def test_incomplete_sequential_run_cannot_publish_consolidated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = (DatasetUnit("multivariate", "p0", "p0_0000", 2026, 0, 0),)
    config = {
        "dataset_cohorts": {"main": {"output_subdir": "main"}},
        "execution": {
            "physical_execution_mode": "sequential",
            "parallel_physical_processes": 1,
        },
        "models": ["QueryCross"],
    }
    monkeypatch.setattr(
        thesis_runner,
        "cohort_units",
        lambda _config, _cohort: units,
    )

    def fake_audit(_config, _cohort, _physical_root, **_kwargs):
        raise ProtocolError("ejecución secuencial incompleta")

    monkeypatch.setattr(thesis_runner, "_audit_physical_root", fake_audit)

    with pytest.raises(ProtocolError, match="ejecución secuencial incompleta"):
        consolidate_physical_shards(
            config,
            "main",
            tmp_path,
            preflight={"manifest_fingerprint": "preflight"},
        )

    assert not (tmp_path / "main" / "physical_models").exists()


def test_cuda_preflight_checks_name_total_and_free_vram(monkeypatch) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    gib = 2**30
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _index: SimpleNamespace(
            name="NVIDIA GeForce RTX 5090", total_memory=32 * gib
        ),
    )
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda _index: (28 * gib, 32 * gib))

    devices = cuda_inventory(config)
    assert devices[0]["total_memory_gib"] == 32.0
    assert devices[0]["free_memory_gib"] == 28.0

    monkeypatch.setattr("torch.cuda.mem_get_info", lambda _index: (4 * gib, 32 * gib))
    with pytest.raises(ProtocolError, match="VRAM total/libre"):
        cuda_inventory(config)


def _raw_rows() -> pd.DataFrame:
    records = []
    for preset, preset_value in (("p1", 2.0), ("p2", 10.0)):
        for kind, kind_offset in (("univariate", -1.0), ("multivariate", 1.0)):
            for seed in EXPECTED_SEEDS:
                records.append(
                    {
                        "Cohort": "main",
                        "Kind": kind,
                        "Preset": preset,
                        "Dataset_ID": f"{preset}_0000",
                        "Seed": seed,
                        "Model": "QueryCross",
                        "test_rmse_z": preset_value + kind_offset,
                        "test_mae_z": preset_value / 2 + kind_offset / 2,
                        "test_rmse": preset_value * 2 + kind_offset,
                        "test_mae": preset_value + kind_offset / 2,
                        "rmse_z_real": preset_value + kind_offset,
                        "rmse_z_all_equal": preset_value + kind_offset + 0.5,
                    }
                )
    return pd.DataFrame(records)


def test_macro_aggregation_is_seed_then_dataset_then_paired_preset() -> None:
    seeded = aggregate_seed_to_dataset(_raw_rows())
    paired = pair_kinds_to_preset(seeded)
    macro = macro_across_presets(paired)

    assert len(seeded) == 4
    assert len(paired) == 2
    assert set(paired["n_kinds"]) == {2}
    assert "test_rmse_mean" not in paired
    assert "test_rmse_macro" not in macro
    assert paired.set_index("Preset").loc["p1", "test_rmse_z_mean"] == 2.0
    assert paired.set_index("Preset").loc["p2", "test_rmse_z_mean"] == 10.0
    assert macro.loc[0, "test_rmse_z_macro"] == 6.0
    assert macro.loc[0, "n_presets"] == 2
    temporal = temporal_ablation_summary(paired, ("real", "all_equal"))
    equal = temporal[temporal["Ablation"] == "all_equal"].iloc[0]
    assert equal["delta_vs_real_macro"] == 0.5
    assert equal["n_presets"] == 2


def _macro_row(model: str, gaussian: bool, ablations: tuple[str, ...]) -> dict:
    row = {
        "Cohort": "main",
        "Model": model,
        "n_presets": 8,
        "test_rmse_z_macro": 0.8,
        "test_rmse_z_preset_sd": 0.1,
        "test_mae_z_macro": 0.6,
        "test_mae_z_preset_sd": 0.08,
        "test_rmse_macro": 1.2,
        "test_rmse_preset_sd": 0.2,
        "test_nll_z_macro": 0.7 if gaussian else float("nan"),
        "test_nll_z_preset_sd": 0.1 if gaussian else float("nan"),
        "test_crps_z_macro": 0.5 if gaussian else float("nan"),
        "test_crps_z_preset_sd": 0.05 if gaussian else float("nan"),
        "test_mean_scale_z_macro": 0.9 if gaussian else float("nan"),
        "test_mean_scale_z_preset_sd": 0.04 if gaussian else float("nan"),
        "test_coverage_90_macro": 0.89 if gaussian else float("nan"),
        "test_coverage_95_macro": 0.94 if gaussian else float("nan"),
    }
    for index, ablation in enumerate(ablations):
        row[f"rmse_z_{ablation}_macro"] = 0.8 + index * 0.01
        row[f"rmse_z_{ablation}_preset_sd"] = 0.1
    return row


def test_latex_report_writes_exactly_the_four_contract_files(tmp_path: Path) -> None:
    config = load_final_config(DEFAULT_CONFIG)
    ablations = tuple(config["evaluation"]["timestamp_ablations"])
    macro = pd.DataFrame(
        [
            _macro_row("QueryCross", False, ablations),
            _macro_row("QueryCross-Gaussian", True, ablations),
        ]
    )
    temporal = pd.DataFrame(
        [
            {
                "Cohort": "main",
                "Model": model,
                "Ablation": ablation,
                "rmse_z_macro": 0.8 + index * 0.01,
                "rmse_z_preset_sd": 0.1,
                "delta_vs_real_macro": index * 0.01,
                "delta_vs_real_preset_sd": 0.02,
                "n_presets": 8,
            }
            for model in ("QueryCross", "QueryCross-Gaussian")
            for index, ablation in enumerate(ablations)
        ]
    )
    horizons = pd.DataFrame(
        [
            {
                "Cohort": "main",
                "Model": model,
                "Level": str(horizon),
                "rmse_z_macro": 0.8 + horizon * 0.01,
                "rmse_z_preset_sd": 0.1,
                "n_presets": 8,
            }
            for model in ("QueryCross", "QueryCross-Gaussian")
            for horizon in (0.25, 1.0, 3.0, 8.0)
        ]
    )
    completion = [
        {
            "cohort": "main",
            "dataset_units": 16,
            "generator_seed": 2026,
            "expected_runs": 432,
            "complete_runs": 432,
        },
        {
            "cohort": "stress",
            "dataset_units": 2,
            "generator_seed": 3031,
            "expected_runs": 54,
            "complete_runs": 54,
        },
    ]

    written = write_latex_tables(
        latex_dir=tmp_path,
        completion=completion,
        main_macro=macro,
        temporal_summary=temporal,
        horizon_summary=horizons,
    )

    assert tuple(written) == LATEX_TABLE_FILENAMES
    assert {path.name for path in tmp_path.iterdir()} == set(LATEX_TABLE_FILENAMES)
    for path in written.values():
        text = path.read_text(encoding="utf-8")
        assert "\\begin{table}" in text
        assert "nan" not in text.lower()
