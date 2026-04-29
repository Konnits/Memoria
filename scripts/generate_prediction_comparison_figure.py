from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.append(ROOT)
if SRC not in sys.path:
    sys.path.append(SRC)

from benchmark_final import _estimate_dataset_time_scale
from state_art.baselines_wrapper import MultiHorizonBaselineWrapper
from state_art.strats.model import STraTSNetwork
from variation_tuning import generate_residual_autoregressive
from ts_transformer.data import (
    SequenceBuilder,
    StandardScaler,
    TimeSeriesDataset,
    build_collate_fn,
    split_dataframe_by_time,
)
from ts_transformer.data.timeseries_dataset import TimeSeriesDatasetConfig
from ts_transformer.models import TimeSeriesEncoderDecoder, TimeSeriesTransformer
from ts_transformer.utils import load_data_config, load_model_config


@dataclass
class PreparedData:
    loader: DataLoader
    value_scaler: StandardScaler
    target_scaler: StandardScaler
    adaptive_time_scale: float
    input_dim: int
    output_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a test-sample prediction comparison figure."
    )
    parser.add_argument("--dataset-id", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-targets", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--history-points", type=int, default=80)
    parser.add_argument("--include-variation", action="store_true")
    parser.add_argument("--include-dilate", action="store_true")
    parser.add_argument(
        "--variation-label",
        type=str,
        default="EncDec-Opt + residual/diff",
    )
    parser.add_argument(
        "--variation-checkpoint",
        type=str,
        default=None,
        help="Optional explicit path to the variation tuning best_model.pt.",
    )
    parser.add_argument(
        "--dilate-label",
        type=str,
        default="EncDec-Opt + DILATE",
    )
    parser.add_argument(
        "--dilate-checkpoint",
        type=str,
        default=None,
        help="Optional explicit path to the DILATE tuning best_model.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="latex/images/predicciones_comparacion_modelos.png",
    )
    return parser.parse_args()


def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("float32").to_numpy()
    return pd.to_datetime(col).apply(lambda x: x.timestamp()).astype("float32").to_numpy()


def _load_checkpoint_state(checkpoint_path: str, device: torch.device | str = "cpu") -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint


def _load_state(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state = _load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state, strict=False)


def _infer_dim_feedforward(checkpoint_path: str, default: int) -> int:
    if not os.path.exists(checkpoint_path):
        return int(default)
    state = _load_checkpoint_state(checkpoint_path, "cpu")
    key = "encoder.layers.0.linear1.weight"
    if key not in state:
        return int(default)
    return int(state[key].shape[0])


def prepare_data(dataset_id: int, num_targets: int, batch_size: int) -> PreparedData:
    data_cfg = load_data_config("configs/data/real_data1.yaml")
    csv_path = os.path.join("data", "processed", f"real_data_{dataset_id}.csv")
    df = pd.read_csv(csv_path)

    time_col = data_cfg.time_column
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col]).apply(lambda x: x.timestamp())

    df = df.sort_values(time_col).reset_index(drop=True)
    df_train, _, df_test = split_dataframe_by_time(
        df,
        time_column=time_col,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
    )

    def process_split(df_part: pd.DataFrame):
        ts = _timestamps_to_float(df_part[time_col])
        x = df_part[data_cfg.feature_columns].to_numpy(dtype="float32")
        y = df_part[data_cfg.target_columns].to_numpy(dtype="float32")
        return ts, x, y

    ts_train, x_train, y_train = process_split(df_train)
    ts_test, x_test, y_test = process_split(df_test)

    value_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train_s = value_scaler.fit_transform(x_train)
    y_train_s = target_scaler.fit_transform(y_train)
    x_test_s = value_scaler.transform(x_test)
    y_test_s = target_scaler.transform(y_test)

    del x_train_s, y_train_s

    values_test = np.concatenate([x_test_s, y_test_s], axis=1)
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    ds_cfg = TimeSeriesDatasetConfig(
        history_length=data_cfg.history_length,
        target_offset_choices=list(range(1, int(num_targets) + 1)),
        stride=data_cfg.stride,
        min_history_length=data_cfg.history_length,
        num_targets=int(num_targets),
    )
    sequence_builder = SequenceBuilder(
        input_dim=input_dim,
        target_token_value="zeros",
        use_sensor_ids=False,
        num_sensors=0,
        num_target_tokens=1,
    )
    dataset = TimeSeriesDataset(
        values_test,
        ts_test,
        ds_cfg,
        input_dim,
        output_dim,
        sequence_builder=sequence_builder,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(pad_to_max_length=True),
        num_workers=0,
    )

    return PreparedData(
        loader=loader,
        value_scaler=value_scaler,
        target_scaler=target_scaler,
        adaptive_time_scale=_estimate_dataset_time_scale(ts_train, fallback=900.0),
        input_dim=input_dim,
        output_dim=output_dim,
    )


def build_models(
    prepared: PreparedData,
    dataset_id: int,
    seed: int,
    device: torch.device,
    include_variation: bool = False,
    variation_label: str = "EncDec-Opt + residual/diff",
    variation_checkpoint: str | None = None,
    include_dilate: bool = False,
    dilate_label: str = "EncDec-Opt + DILATE",
    dilate_checkpoint: str | None = None,
) -> dict[str, torch.nn.Module]:
    exp_root = os.path.join("experiments", "benchmark_final", f"ds_{dataset_id}_seed_{seed}")
    custom_checkpoint = os.path.join(exp_root, "Custom", "best_model.pt")

    base_cfg = load_model_config("configs/model/transformer_base_real_data1.yaml")
    base_cfg.input_dim = prepared.input_dim
    base_cfg.output_dim = prepared.output_dim
    base_cfg.use_sensor_embedding = False
    base_cfg.num_sensors = 0
    base_cfg.time_scale = float(prepared.adaptive_time_scale)
    base_cfg.dim_feedforward = _infer_dim_feedforward(
        custom_checkpoint,
        int(base_cfg.dim_feedforward),
    )

    small_cfg = load_model_config("configs/model/transformer_small.yaml")
    small_cfg.input_dim = prepared.input_dim
    small_cfg.output_dim = prepared.output_dim
    small_cfg.use_sensor_embedding = False
    small_cfg.num_sensors = 0
    small_cfg.time_scale = float(prepared.adaptive_time_scale)
    small_cfg.decoder_num_layers = 1
    small_cfg.use_causal_mask = True

    best = TimeSeriesEncoderDecoder(copy.deepcopy(small_cfg)).to(device)
    variation = TimeSeriesEncoderDecoder(copy.deepcopy(small_cfg)).to(device)
    dilate = TimeSeriesEncoderDecoder(copy.deepcopy(small_cfg)).to(device)
    custom = TimeSeriesTransformer(copy.deepcopy(base_cfg)).to(device)
    custom_state = _load_checkpoint_state(custom_checkpoint, "cpu") if os.path.exists(custom_checkpoint) else {}
    if "value_embedding.ln.weight" not in custom_state:
        custom.value_embedding.ln = torch.nn.Identity()
    if "input_norm.weight" not in custom_state:
        custom.input_norm = torch.nn.Identity()
    strats_base = STraTSNetwork(
        num_features=prepared.input_dim + 1,
        d_model=base_cfg.d_model,
        num_classes=prepared.output_dim,
    )
    strats = MultiHorizonBaselineWrapper(
        strats_base,
        "strats",
        base_cfg.d_model,
        prepared.output_dim,
        use_sensor_embedding=False,
    ).to(device)

    paths = {
        "EncDec-Opt pequeno + AR": os.path.join(
            exp_root,
            "EncDec-Opt-Small-MT8_FT_AR_Contiguous",
            "best_model.pt",
        ),
        "CT-Transformer base": custom_checkpoint,
        "STraTS adaptado": os.path.join(exp_root, "STraTS_Adapter", "best_model.pt"),
    }
    models = {
        "EncDec-Opt pequeno + AR": best,
        "CT-Transformer base": custom,
        "STraTS adaptado": strats,
    }
    if include_variation:
        paths[variation_label] = variation_checkpoint or os.path.join(
            exp_root,
            "EncDec-Opt-Small-MT8_FT_AR_Contiguous_VAR_ResidualDiff",
            "best_model.pt",
        )
        setattr(variation, "_predicts_residual_variation", True)
        models[variation_label] = variation
    if include_dilate:
        paths[dilate_label] = dilate_checkpoint or os.path.join(
            exp_root,
            "EncDec-Opt-Small-MT8_FT_AR_Contiguous_DILATE",
            "best_model.pt",
        )
        models[dilate_label] = dilate

    for name, model in models.items():
        _load_state(model, paths[name], device)
        model.eval()
    return models


def inverse_target(target_scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    inv = target_scaler.inverse_transform(flat)
    return inv.reshape(shape)


def predict_batch(
    models: dict[str, torch.nn.Module],
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, np.ndarray]:
    input_values = batch["input_values"].to(device)
    input_timestamps = batch["input_timestamps"].to(device)
    is_target_mask = batch["is_target_mask"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    num_targets = batch["target_values"].shape[1]

    predictions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, model in models.items():
            if isinstance(model, TimeSeriesEncoderDecoder):
                hist_values = input_values[:, :-num_targets, :]
                hist_timestamps = input_timestamps[:, :-num_targets]
                target_timestamps = input_timestamps[:, -num_targets:]
                hist_padding_mask = padding_mask[:, :-num_targets]
                if getattr(model, "_predicts_residual_variation", False):
                    pred = generate_residual_autoregressive(
                        model,
                        history_values=hist_values,
                        history_timestamps=hist_timestamps,
                        target_timestamps=target_timestamps,
                        history_padding_mask=hist_padding_mask,
                    )
                else:
                    pred = model.generate(
                        history_values=hist_values,
                        history_timestamps=hist_timestamps,
                        target_timestamps=target_timestamps,
                        history_padding_mask=hist_padding_mask,
                    )
            else:
                pred = model(
                    input_values=input_values,
                    input_timestamps=input_timestamps,
                    is_target_mask=is_target_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )
            predictions[name] = pred.detach().cpu().numpy()
    return predictions


def collect_candidates(
    prepared: PreparedData,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    max_batches: int,
):
    batches = []
    preds_by_batch = []
    for batch_idx, batch in enumerate(prepared.loader):
        if batch_idx >= max_batches:
            break
        preds = predict_batch(models, batch, device)
        batches.append(batch)
        preds_by_batch.append(preds)
    return batches, preds_by_batch


def choose_sample(
    prepared: PreparedData,
    batches: list[dict[str, torch.Tensor]],
    preds_by_batch: list[dict[str, np.ndarray]],
) -> tuple[int, int]:
    target_ranges = []
    best_errors = []
    refs = []
    for batch_idx, (batch, preds) in enumerate(zip(batches, preds_by_batch)):
        targets = inverse_target(prepared.target_scaler, batch["target_values"].numpy())
        best_pred = inverse_target(
            prepared.target_scaler,
            preds["EncDec-Opt pequeno + AR"],
        )
        for sample_idx in range(targets.shape[0]):
            target = targets[sample_idx, :, 0]
            pred = best_pred[sample_idx, :, 0]
            target_ranges.append(float(np.ptp(target)))
            best_errors.append(float(np.sqrt(np.mean((pred - target) ** 2))))
            refs.append((batch_idx, sample_idx))

    ranges = np.asarray(target_ranges)
    errors = np.asarray(best_errors)
    if len(ranges) == 0:
        raise RuntimeError("No samples were collected.")

    rich = ranges >= np.quantile(ranges, 0.65)
    stable = errors <= np.quantile(errors, 0.65)
    candidates = np.where(rich & stable)[0]
    if candidates.size == 0:
        candidates = np.arange(len(ranges))

    range_norm = (ranges - ranges.min()) / (np.ptp(ranges) + 1e-8)
    error_norm = (errors - errors.min()) / (np.ptp(errors) + 1e-8)
    score = range_norm - 0.35 * error_norm
    chosen = int(candidates[np.argmax(score[candidates])])
    return refs[chosen]


def plot_sample(
    prepared: PreparedData,
    batch: dict[str, torch.Tensor],
    preds: dict[str, np.ndarray],
    sample_idx: int,
    output_path: str,
    history_points: int,
) -> None:
    target = inverse_target(
        prepared.target_scaler,
        batch["target_values"][sample_idx : sample_idx + 1].numpy(),
    )[0, :, 0]
    pred_orig = {
        name: inverse_target(prepared.target_scaler, arr[sample_idx : sample_idx + 1])[0, :, 0]
        for name, arr in preds.items()
    }

    mask = (
        (~batch["is_target_mask"][sample_idx])
        & (~batch["padding_mask"][sample_idx])
    ).numpy()
    hist_t = batch["input_timestamps"][sample_idx].numpy()[mask]
    hist_v_scaled = batch["input_values"][sample_idx].numpy()[mask, 0:1]
    hist_v = prepared.value_scaler.inverse_transform(hist_v_scaled)[:, 0]
    target_t = batch["target_timestamps"][sample_idx].numpy()

    hist_t = hist_t[-history_points:]
    hist_v = hist_v[-history_points:]

    origin = hist_t[-1]
    hist_x = (hist_t - origin) / 3600.0
    target_x = (target_t - origin) / 3600.0

    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    marker_cycle = ["o", "s", "^", "D", "P", "v"]
    colors = {
        name: color_cycle[idx % len(color_cycle)]
        for idx, name in enumerate(pred_orig.keys())
    }
    markers = {
        name: marker_cycle[idx % len(marker_cycle)]
        for idx, name in enumerate(pred_orig.keys())
    }

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(hist_x, hist_v, color="#777777", linewidth=1.2, alpha=0.75, label="Historia")
    ax.plot(target_x, target, color="black", marker="x", linewidth=2.0, markersize=7, label="Valor real")

    for name in pred_orig:
        ax.plot(
            target_x,
            pred_orig[name],
            color=colors[name],
            marker=markers[name],
            linewidth=1.8,
            markersize=5,
            linestyle="--",
            label=name,
        )

    ax.axvline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Horas relativas al ultimo punto de historia")
    ax.set_ylabel("Valor")
    ax.set_title("Ejemplo de prediccion multi-horizonte en test")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(args.dataset_id, args.num_targets, args.batch_size)
    models = build_models(
        prepared,
        args.dataset_id,
        args.seed,
        device,
        include_variation=args.include_variation,
        variation_label=args.variation_label,
        variation_checkpoint=args.variation_checkpoint,
        include_dilate=args.include_dilate,
        dilate_label=args.dilate_label,
        dilate_checkpoint=args.dilate_checkpoint,
    )
    batches, preds_by_batch = collect_candidates(
        prepared,
        models,
        device,
        max_batches=args.max_batches,
    )
    batch_idx, sample_idx = choose_sample(prepared, batches, preds_by_batch)
    plot_sample(
        prepared,
        batches[batch_idx],
        preds_by_batch[batch_idx],
        sample_idx,
        args.output,
        history_points=args.history_points,
    )
    print(
        "Generated "
        f"{args.output} using dataset={args.dataset_id}, seed={args.seed}, "
        f"batch={batch_idx}, sample={sample_idx}."
    )


if __name__ == "__main__":
    main()
