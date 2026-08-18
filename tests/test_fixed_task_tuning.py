from pathlib import Path

import optuna

from scripts.tune_synthetic_optuna import sample_trial_config
from ts_transformer.utils import load_data_config, load_model_config, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def test_optuna_samples_architecture_but_keeps_forecasting_task_fixed() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "architecture": "small_4h",
            "encoder_layers": 2,
            "ffn_multiplier": 2,
            "dropout": 0.0,
            "time_encoding_mode": "sinusoidal",
            "time_transform": "linear",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
        }
    )
    training_cfg, _ = load_training_config(
        str(ROOT / "configs" / "training" / "synthetic_benchmark.yaml")
    )
    _, _, data_cfg = sample_trial_config(
        trial,
        "Custom",
        load_model_config(str(ROOT / "configs" / "model" / "synthetic_transformer.yaml")),
        training_cfg,
        load_data_config(str(ROOT / "configs" / "data" / "synthetic_benchmark.yaml")),
        horizon_profile="extended_8",
        history_length=256,
    )

    assert data_cfg.history_length == 256
    assert data_cfg.target_offset_choices == [1, 2, 4, 8, 16, 32, 64, 128]
    assert "horizon_profile" not in trial.params
    assert "history_length" not in trial.params
