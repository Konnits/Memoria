from __future__ import annotations

from ts_transformer.hyperparameter_search import (
    SearchConfig,
    SearchParameter,
    apply_trial_overrides,
    generate_trials,
)


def test_generate_grid_trials() -> None:
    cfg = SearchConfig(
        name="demo",
        strategy="grid",
        metric="val_rmse",
        mode="min",
        seed=42,
        max_trials=None,
        parameters=(
            SearchParameter(section="model", path="d_model", values=(64, 128)),
            SearchParameter(section="training", path="optimizer.lr", values=(1e-3, 1e-4)),
        ),
    )

    trials = generate_trials(cfg)

    assert len(trials) == 4
    assert trials[0]["model.d_model"] == 64
    assert trials[0]["training.optimizer.lr"] == 1e-3


def test_generate_random_trials_without_duplicates() -> None:
    cfg = SearchConfig(
        name="demo",
        strategy="random",
        metric="val_rmse",
        mode="min",
        seed=7,
        max_trials=3,
        parameters=(
            SearchParameter(section="model", path="d_model", values=(64, 128)),
            SearchParameter(section="training", path="optimizer.lr", values=(1e-3, 1e-4)),
        ),
    )

    trials = generate_trials(cfg)

    assert len(trials) == 3
    assert len({tuple(sorted(t.items())) for t in trials}) == 3


def test_apply_trial_overrides_updates_nested_values() -> None:
    model_cfg = {"d_model": 128}
    training_cfg = {"optimizer": {"lr": 1e-3}}
    data_cfg = {"history_length": 500}

    model_out, training_out, data_out = apply_trial_overrides(
        model_cfg,
        training_cfg,
        data_cfg,
        {
            "model.d_model": 64,
            "training.optimizer.lr": 5e-4,
            "data.history_length": 256,
        },
    )

    assert model_out["d_model"] == 64
    assert training_out["optimizer"]["lr"] == 5e-4
    assert data_out["history_length"] == 256
    assert model_cfg["d_model"] == 128