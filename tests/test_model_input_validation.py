from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ts_transformer.models.time_series_encoder_decoder import TimeSeriesEncoderDecoder
from ts_transformer.models.time_series_transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)


def _make_config() -> TimeSeriesTransformerConfig:
    return TimeSeriesTransformerConfig(
        input_dim=1,
        output_dim=1,
        d_model=8,
        num_heads=2,
        num_layers=1,
        decoder_num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        validate_inputs=True,
    )


@pytest.mark.parametrize(
    "model_type",
    [TimeSeriesTransformer, TimeSeriesEncoderDecoder],
)
def test_models_reject_nonterminal_target_tokens(model_type) -> None:
    model = model_type(_make_config())
    values = torch.zeros(1, 4, 1)
    timestamps = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    nonterminal_targets = torch.tensor([[False, True, False, True]])

    with pytest.raises(ValueError, match="estrictamente al final"):
        model(values, timestamps, nonterminal_targets)


def test_target_structure_validation_survives_python_optimized_mode() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    script = """
import torch
from ts_transformer.models.time_series_encoder_decoder import TimeSeriesEncoderDecoder
from ts_transformer.models.time_series_transformer import TimeSeriesTransformer, TimeSeriesTransformerConfig

config = TimeSeriesTransformerConfig(
    input_dim=1,
    output_dim=1,
    d_model=8,
    num_heads=2,
    num_layers=1,
    decoder_num_layers=1,
    dim_feedforward=16,
    dropout=0.0,
    validate_inputs=True,
)
values = torch.zeros(1, 4, 1)
timestamps = torch.arange(4, dtype=torch.float32).unsqueeze(0)
invalid_mask = torch.tensor([[False, True, False, True]])

for model_type in (TimeSeriesTransformer, TimeSeriesEncoderDecoder):
    try:
        model_type(config)(values, timestamps, invalid_mask)
    except ValueError as exc:
        if "estrictamente al final" not in str(exc):
            raise
    else:
        raise RuntimeError(f"{model_type.__name__} aceptó una máscara inválida")
"""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )

    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=repository_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
