from pathlib import Path

import pytest
import torch

from scripts.benchmark_synthetic import (
    DEFAULT_MODEL_NAMES,
    MODEL_NAMES,
    OPTIMIZED_SIZE,
    apply_common_benchmark_task,
    build_models,
    configure_training_for_family,
    load_benchmark_recipes,
)
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper
from ts_transformer.utils import (
    load_data_config,
    load_model_config,
    load_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = ROOT / "configs" / "benchmark" / "synthetic_optuna_best.yaml"


def test_time_bias_is_explicitly_opt_in_for_the_frozen_benchmark():
    assert "Custom-TimeBias" in MODEL_NAMES
    assert "Custom-TimeBias" not in DEFAULT_MODEL_NAMES
    assert len(DEFAULT_MODEL_NAMES) == 9


def test_frozen_optuna_recipes_and_common_task_are_consistent():
    recipes = load_benchmark_recipes(RECIPES_PATH)
    data_cfg = apply_common_benchmark_task(
        load_data_config(str(ROOT / "configs" / "data" / "synthetic_benchmark.yaml")),
        recipes,
    )

    assert data_cfg.history_length == 512
    assert data_cfg.target_offset_choices == [1, 4, 16, 64]
    assert data_cfg.num_targets == 4
    assert recipes["families"]["Custom"]["best_trial_number"] == 117
    assert recipes["families"]["EncDec-AR"]["best_trial_number"] == 249


@pytest.mark.parametrize(
    ("use_events", "input_dim", "model_input_dim"),
    [(False, 1, 1), (True, 6, 1)],
)
def test_every_model_has_an_explicit_compatible_run_spec(
    use_events, input_dim, model_input_dim
):
    recipes = load_benchmark_recipes(RECIPES_PATH)
    model_cfg = load_model_config(
        str(ROOT / "configs" / "model" / "synthetic_transformer.yaml")
    )
    models = build_models(
        model_cfg,
        {
            "model_input_dim": model_input_dim,
            "output_dim": input_dim,
            "input_dim": input_dim,
            "use_events": use_events,
            "time_scale": 1.0,
        },
        (OPTIMIZED_SIZE,),
        recipes,
    )

    assert set(models) == {
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
    }
    assert models["EncDec-AR"].autoregressive is True
    assert models["EncDec-AR"].training_family == "EncDec-AR"
    assert models["Custom"].model.config.d_model == 96
    assert models["Custom"].model.config.num_heads == 4
    assert models["Custom-TimeBias"].training_family == "Custom"
    assert models["Custom-TimeBias"].model.config.use_temporal_attn_bias is True
    assert models["Custom-TimeBias"].model.config.temporal_bias_layers is None
    assert models["Custom-TimeBias"].model.temporal_attn_bias is not None
    assert models["EncDec-AR"].model.config.decoder_num_layers == 1
    assert all(spec.model is not None for spec in models.values())


def test_custom_time_bias_runs_on_irregular_left_padded_sequences():
    recipes = load_benchmark_recipes(RECIPES_PATH)
    model_cfg = load_model_config(
        str(ROOT / "configs" / "model" / "synthetic_transformer.yaml")
    )
    models = build_models(
        model_cfg,
        {
            "model_input_dim": 1,
            "output_dim": 1,
            "input_dim": 1,
            "use_events": False,
            "time_scale": 1.0,
        },
        (OPTIMIZED_SIZE,),
        recipes,
        model_seed=42,
    )
    model = models["Custom-TimeBias"].model.train()

    values = torch.tensor(
        [
            [[0.0], [0.2], [0.5], [0.7], [0.0]],
            [[0.1], [0.3], [0.4], [0.9], [0.0]],
        ],
        dtype=torch.float32,
    )
    timestamps = torch.tensor(
        [[0.0, 1.0, 2.5, 7.0, 11.0], [0.0, 0.4, 3.0, 9.5, 12.0]],
        dtype=torch.float32,
    )
    is_target = torch.zeros(2, 5, dtype=torch.bool)
    is_target[:, -1] = True
    padding_mask = torch.tensor(
        [[True, False, False, False, False], [False, False, False, False, False]],
        dtype=torch.bool,
    )

    predictions = model(
        input_values=values,
        input_timestamps=timestamps,
        is_target_mask=is_target,
        padding_mask=padding_mask,
    )
    predictions.square().mean().backward()

    assert predictions.shape == (2, 1)
    assert torch.isfinite(predictions).all()
    assert model.temporal_attn_bias.log_tau.requires_grad
    assert model.temporal_attn_bias.log_tau.grad is not None
    assert torch.isfinite(model.temporal_attn_bias.log_tau.grad).all()
    assert torch.count_nonzero(model.temporal_attn_bias.log_tau.grad) > 0


def test_optimizer_winners_are_applied_per_family():
    recipes = load_benchmark_recipes(RECIPES_PATH)
    training_cfg, _ = load_training_config(
        str(ROOT / "configs" / "training" / "synthetic_benchmark.yaml")
    )

    custom = configure_training_for_family(training_cfg, recipes, "Custom")
    encdec = configure_training_for_family(training_cfg, recipes, "EncDec-AR")

    assert custom.optimizer_config.lr == pytest.approx(0.0005655377414706213)
    assert custom.optimizer_config.weight_decay == pytest.approx(0.0003748733910339571)
    assert custom.optimizer_config.warmup_epochs == 2
    assert encdec.optimizer_config.lr == pytest.approx(0.0006997557263932117)
    assert encdec.optimizer_config.weight_decay == pytest.approx(0.004344957796532742)
    assert encdec.optimizer_config.warmup_epochs == 3


def test_baseline_time_origin_ignores_left_padding():
    times = torch.tensor([[0.0, 0.0, 10.0, 12.0], [0.0, 4.0, 6.0, 8.0]])
    valid = torch.tensor(
        [[False, False, True, True], [False, True, True, True]], dtype=torch.bool
    )

    t0 = MultiHorizonBaselineWrapper._first_valid_time(times, valid)

    assert torch.equal(t0, torch.tensor([[10.0], [4.0]]))


def test_model_initialization_is_independent_and_repeatable_per_seed():
    recipes = load_benchmark_recipes(RECIPES_PATH)
    model_cfg = load_model_config(
        str(ROOT / "configs" / "model" / "synthetic_transformer.yaml")
    )
    data = {
        "model_input_dim": 1,
        "output_dim": 1,
        "input_dim": 1,
        "use_events": False,
        "time_scale": 1.0,
    }

    first = build_models(
        model_cfg, data, (OPTIMIZED_SIZE,), recipes, model_seed=84
    )
    second = build_models(
        model_cfg, data, (OPTIMIZED_SIZE,), recipes, model_seed=84
    )

    for model_name in (
        "LastValueTimeMLP",
        "Custom",
        "Custom-TimeBias",
        "EncDec-AR",
        "CoFormer",
    ):
        first_parameter = next(first[model_name].model.parameters()).detach()
        second_parameter = next(second[model_name].model.parameters()).detach()
        assert torch.equal(first_parameter, second_parameter), model_name
