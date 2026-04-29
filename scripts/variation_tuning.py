from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.append(ROOT)
if SRC not in sys.path:
    sys.path.append(SRC)

from ar_finetuning import prepare_ar_data
from benchmark_final import _estimate_dataset_time_scale
from ts_transformer.data import split_dataframe_by_time
from ts_transformer.models import TimeSeriesEncoderDecoder
from ts_transformer.training.metrics import compute_regression_metrics
from ts_transformer.utils import (
    load_data_config,
    load_model_config,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Variation tuning: residual prediction plus shape/difference loss "
            "initialized from an existing EncDec checkpoint."
        )
    )
    parser.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126])
    parser.add_argument("--exp-dir", type=str, default="experiments/benchmark_final")
    parser.add_argument(
        "--source-model",
        type=str,
        default="EncDec-Opt-Small-MT8_FT_AR_Contiguous",
    )
    parser.add_argument("--output-suffix", type=str, default="VAR_ResidualDiff")
    parser.add_argument("--num-targets", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--diff-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="experiments/benchmark_final/variation_residual_diff.csv",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data/model and compute one training loss batch.",
    )
    return parser.parse_args()


def _timestamps_to_float(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("float32").to_numpy()
    return pd.to_datetime(col).apply(lambda x: x.timestamp()).astype("float32").to_numpy()


def _adaptive_time_scale(base_data_cfg: Any, dataset_id: int) -> float:
    csv_path = os.path.join("data", "processed", f"real_data_{dataset_id}.csv")
    df = pd.read_csv(csv_path)
    time_col = base_data_cfg.time_column
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col]).apply(lambda x: x.timestamp())
    df = df.sort_values(time_col).reset_index(drop=True)
    df_train, _, _ = split_dataframe_by_time(
        df,
        time_column=time_col,
        train_ratio=base_data_cfg.train_ratio,
        val_ratio=base_data_cfg.val_ratio,
    )
    return _estimate_dataset_time_scale(_timestamps_to_float(df_train[time_col]), fallback=900.0)


def _build_model(base_data_cfg: Any, dataset_id: int, data: dict[str, Any]) -> TimeSeriesEncoderDecoder:
    cfg = load_model_config("configs/model/transformer_small.yaml")
    train_ds = data["train_loader"].dataset
    cfg.input_dim = int(train_ds.input_dim)
    cfg.output_dim = int(train_ds.output_dim)
    cfg.use_sensor_embedding = False
    cfg.num_sensors = 0
    cfg.time_scale = float(_adaptive_time_scale(base_data_cfg, dataset_id))
    cfg.decoder_num_layers = 1
    cfg.use_causal_mask = True
    return TimeSeriesEncoderDecoder(cfg)


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatible: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def _last_history_value(batch: dict[str, torch.Tensor], output_dim: int) -> torch.Tensor:
    input_values = batch["input_values"]
    is_target_mask = batch["is_target_mask"].to(torch.bool)
    padding_mask = batch.get("padding_mask", None)
    if padding_mask is None:
        padding_mask = torch.zeros_like(is_target_mask, dtype=torch.bool)
    else:
        padding_mask = padding_mask.to(torch.bool)

    valid_history = (~is_target_mask) & (~padding_mask)
    positions = torch.arange(input_values.shape[1], device=input_values.device).unsqueeze(0)
    last_idx = positions.expand_as(valid_history).masked_fill(~valid_history, -1).max(dim=1).values
    if torch.any(last_idx < 0):
        raise RuntimeError("A batch item has no valid history token.")

    batch_idx = torch.arange(input_values.shape[0], device=input_values.device)
    return input_values[batch_idx, last_idx, :output_dim]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def residual_variation_loss(
    pred_residual: torch.Tensor,
    targets: torch.Tensor,
    origin: torch.Tensor,
    target_loss_mask: torch.Tensor | None,
    residual_weight: float,
    value_weight: float,
    diff_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    if pred_residual.ndim == 2:
        pred_residual = pred_residual.unsqueeze(1)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)

    origin = origin.unsqueeze(1)
    target_residual = targets - origin
    pred_abs = origin + pred_residual

    residual_per = F.smooth_l1_loss(pred_residual, target_residual, reduction="none", beta=0.5)
    value_per = F.smooth_l1_loss(pred_abs, targets, reduction="none", beta=0.5)
    residual_loss = _masked_mean(residual_per, target_loss_mask)
    value_loss = _masked_mean(value_per, target_loss_mask)

    if targets.shape[1] > 1:
        pred_diff = pred_abs[:, 1:, :] - pred_abs[:, :-1, :]
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]
        diff_per = F.smooth_l1_loss(pred_diff, target_diff, reduction="none", beta=0.5)
        diff_mask = None
        if target_loss_mask is not None:
            diff_mask = target_loss_mask[:, 1:, :] * target_loss_mask[:, :-1, :]
        diff_loss = _masked_mean(diff_per, diff_mask)
    else:
        diff_loss = pred_abs.new_tensor(0.0)

    total = (
        float(residual_weight) * residual_loss
        + float(value_weight) * value_loss
        + float(diff_weight) * diff_loss
    )
    parts = {
        "residual_loss": float(residual_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "diff_loss": float(diff_loss.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
    }
    return total, parts, pred_abs


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
    return out


def _forward_loss(
    model: TimeSeriesEncoderDecoder,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    preds_residual = model(
        input_values=batch["input_values"],
        input_timestamps=batch["input_timestamps"],
        is_target_mask=batch["is_target_mask"],
        padding_mask=batch.get("padding_mask", None),
        return_dict=False,
    )
    origin = _last_history_value(batch, model.output_dim)
    return residual_variation_loss(
        preds_residual,
        batch["target_values"],
        origin,
        batch.get("target_loss_mask", None),
        residual_weight=args.residual_weight,
        value_weight=args.value_weight,
        diff_weight=args.diff_weight,
    )


@torch.no_grad()
def evaluate_teacher_forced(
    model: TimeSeriesEncoderDecoder,
    loader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    parts_sum = {"residual_loss": 0.0, "value_loss": 0.0, "diff_loss": 0.0}
    n = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_val_batches and batch_idx >= args.max_val_batches:
            break
        batch = _batch_to_device(batch, device)
        loss, parts, _ = _forward_loss(model, batch, args)
        total += float(loss.item())
        for key in parts_sum:
            parts_sum[key] += float(parts[key])
        n += 1
    if n == 0:
        return {"val_loss": float("nan")}
    out = {"val_loss": total / n}
    out.update({f"val_{key}": value / n for key, value in parts_sum.items()})
    return out


@torch.no_grad()
def generate_residual_autoregressive(
    model: TimeSeriesEncoderDecoder,
    history_values: torch.Tensor,
    history_timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
    history_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    batch_size, history_len, input_dim = history_values.shape
    num_targets = target_timestamps.shape[1]
    device = history_values.device
    dtype = history_values.dtype

    if history_padding_mask is not None:
        history_padding_mask = history_padding_mask.to(device=device, dtype=torch.bool)
    valid_history = torch.ones(batch_size, history_len, dtype=torch.bool, device=device)
    if history_padding_mask is not None:
        valid_history = ~history_padding_mask
    positions = torch.arange(history_len, device=device).unsqueeze(0)
    last_idx = positions.expand_as(valid_history).masked_fill(~valid_history, -1).max(dim=1).values
    batch_idx = torch.arange(batch_size, device=device)
    origin = history_values[batch_idx, last_idx, : model.output_dim]

    history_mask = torch.zeros(batch_size, history_len, dtype=torch.bool, device=device)
    history_emb = model._embed_tokens(
        input_values=history_values,
        input_timestamps=history_timestamps,
        is_target_mask=history_mask,
        padding_mask=history_padding_mask,
    )
    encoder_output = model.encoder(
        history_emb,
        key_padding_mask=history_padding_mask,
        attn_mask=None,
        return_all_layers=False,
    )

    generations = []
    current_target_inputs = torch.zeros(batch_size, 1, input_dim, dtype=dtype, device=device)
    for step in range(num_targets):
        input_values = torch.cat([history_values, current_target_inputs], dim=1)
        current_target_timestamps = target_timestamps[:, : step + 1]
        input_timestamps = torch.cat([history_timestamps, current_target_timestamps], dim=1)
        is_target_mask = torch.zeros(batch_size, history_len + step + 1, dtype=torch.bool, device=device)
        is_target_mask[:, history_len:] = True

        full_padding_mask = None
        if history_padding_mask is not None:
            target_padding = torch.zeros(batch_size, step + 1, dtype=torch.bool, device=device)
            full_padding_mask = torch.cat([history_padding_mask, target_padding], dim=1)

        x_all = model._embed_tokens(
            input_values=input_values,
            input_timestamps=input_timestamps,
            is_target_mask=is_target_mask,
            padding_mask=full_padding_mask,
        )
        x_dec = x_all[:, history_len:, :]
        decoder_output = model.decoder(
            x_dec,
            encoder_out=encoder_output,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=history_padding_mask,
            tgt_attn_mask=None,
            cross_attn_mask=None,
            is_causal=True,
            return_all_layers=False,
        )
        pred_residual = model.head(decoder_output)
        latest_residual = pred_residual[:, -1:, :] if pred_residual.ndim == 3 else pred_residual.unsqueeze(1)
        latest_abs = origin.unsqueeze(1) + latest_residual
        generations.append(latest_abs)

        if step < num_targets - 1:
            next_input = torch.zeros(batch_size, 1, input_dim, dtype=dtype, device=device)
            out_dim = min(input_dim, model.output_dim)
            next_input[:, 0, :out_dim] = latest_abs[:, 0, :out_dim]
            current_target_inputs = torch.cat([current_target_inputs, next_input], dim=1)

    return torch.cat(generations, dim=1)


@torch.no_grad()
def evaluate_autoregressive(
    model: TimeSeriesEncoderDecoder,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    for batch_idx, batch in enumerate(loader):
        if args.max_test_batches and batch_idx >= args.max_test_batches:
            break
        batch = _batch_to_device(batch, device)
        target_values = batch["target_values"]
        num_targets = target_values.shape[1]
        padding_mask = batch.get("padding_mask", None)
        history_padding_mask = padding_mask[:, :-num_targets] if padding_mask is not None else None
        preds = generate_residual_autoregressive(
            model,
            history_values=batch["input_values"][:, :-num_targets, :],
            history_timestamps=batch["input_timestamps"][:, :-num_targets],
            target_timestamps=batch["input_timestamps"][:, -num_targets:],
            history_padding_mask=history_padding_mask,
        )
        all_preds.append(preds.detach().cpu())
        all_targets.append(target_values.detach().cpu())
        if "target_loss_mask" in batch:
            all_masks.append(batch["target_loss_mask"].detach().cpu())

    if not all_preds:
        return {}
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    mask_cat = torch.cat(all_masks, dim=0) if all_masks else None
    if mask_cat is not None and torch.any(mask_cat > 0):
        valid = mask_cat > 0
        preds_for_metrics = preds_cat[valid].view(-1, 1)
        targets_for_metrics = targets_cat[valid].view(-1, 1)
    else:
        preds_for_metrics = preds_cat.view(-1, 1)
        targets_for_metrics = targets_cat.view(-1, 1)

    metrics = compute_regression_metrics(preds_for_metrics, targets_for_metrics, prefix=prefix)
    if preds_cat.shape[1] > 1:
        pred_diff = preds_cat[:, 1:, :] - preds_cat[:, :-1, :]
        target_diff = targets_cat[:, 1:, :] - targets_cat[:, :-1, :]
        if mask_cat is not None:
            diff_mask = (mask_cat[:, 1:, :] * mask_cat[:, :-1, :]) > 0
            pred_diff_metrics = pred_diff[diff_mask].view(-1, 1)
            target_diff_metrics = target_diff[diff_mask].view(-1, 1)
        else:
            pred_diff_metrics = pred_diff.view(-1, 1)
            target_diff_metrics = target_diff.view(-1, 1)
        diff_metrics = compute_regression_metrics(
            pred_diff_metrics,
            target_diff_metrics,
            prefix=f"{prefix}diff_",
        )
        direction = torch.sign(pred_diff_metrics) == torch.sign(target_diff_metrics)
        metrics.update(diff_metrics)
        metrics[f"{prefix}direction_acc"] = float(direction.float().mean().item())
    return metrics


def _save_checkpoint(
    path: str,
    model: TimeSeriesEncoderDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "variation_tuning_config": vars(args),
        },
        path,
    )


def train_one_run(
    dataset_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_global_seed(seed)
    base_data_cfg = load_data_config("configs/data/real_data1.yaml")
    if args.batch_size is not None:
        base_data_cfg.batch_size = int(args.batch_size)

    data = prepare_ar_data(
        "Contiguous",
        base_data_cfg,
        dataset_id,
        logger=None,
        num_targets_override=args.num_targets,
        num_workers_override=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    if data is None:
        raise FileNotFoundError(f"No data for dataset {dataset_id}")

    model = _build_model(base_data_cfg, dataset_id, data).to(device)
    source_dir = os.path.join(args.exp_dir, f"ds_{dataset_id}_seed_{seed}", args.source_model)
    source_ckpt = os.path.join(source_dir, "best_model.pt")
    _load_model_weights(model, source_ckpt, device)
    model.config.use_causal_mask = True

    output_model = f"{args.source_model}_{args.output_suffix}"
    output_dir = os.path.join(args.exp_dir, f"ds_{dataset_id}_seed_{seed}", output_model)
    best_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_path) and not args.force and not args.dry_run:
        return {
            "Dataset_ID": dataset_id,
            "Seed": seed,
            "Modelo": output_model,
            "status": "skipped_existing",
            "checkpoint_dir": output_dir,
        }

    if args.dry_run:
        batch = next(iter(data["train_loader"]))
        batch = _batch_to_device(batch, device)
        loss, parts, pred_abs = _forward_loss(model, batch, args)
        return {
            "Dataset_ID": dataset_id,
            "Seed": seed,
            "Modelo": output_model,
            "status": "dry_run",
            "dry_run_loss": float(loss.item()),
            "dry_run_pred_shape": list(pred_abs.shape),
            **{f"dry_run_{key}": value for key, value in parts.items()},
        }

    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    history: list[dict[str, float]] = []
    best_val = float("inf")
    bad_epochs = 0
    train_start = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        parts_running = {"residual_loss": 0.0, "value_loss": 0.0, "diff_loss": 0.0}
        n_batches = 0
        for batch_idx, batch in enumerate(data["train_loader"]):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _ = _forward_loss(model, batch, args)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            running += float(loss.item())
            for key in parts_running:
                parts_running[key] += float(parts[key])
            n_batches += 1

        val_metrics = evaluate_teacher_forced(model, data["val_loader"], args, device)
        train_loss = running / max(1, n_batches)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            **{f"train_{key}": value / max(1, n_batches) for key, value in parts_running.items()},
            **val_metrics,
        }
        history.append(row)
        print(
            f"[VAR ds={dataset_id} seed={seed}] epoch={epoch:02d} "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['val_loss']:.6f}"
        )

        current_val = float(val_metrics["val_loss"])
        if current_val < best_val - 1e-5:
            best_val = current_val
            bad_epochs = 0
            _save_checkpoint(best_path, model, optimizer, epoch, best_val, args)
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    train_time = time.time() - train_start
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_autoregressive(model, data["test_loader"], args, device, prefix="test_var_")
    val_ar_metrics = evaluate_autoregressive(model, data["val_loader"], args, device, prefix="val_var_")
    history_path = os.path.join(output_dir, "variation_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    record: dict[str, Any] = {
        "Dataset_ID": dataset_id,
        "Seed": seed,
        "Modelo": output_model,
        "status": "completed",
        "source_model": args.source_model,
        "checkpoint_dir": output_dir,
        "train_time_s": round(train_time, 2),
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "train_num_targets": int(data["num_targets"]),
        "eval_num_targets": int(data["num_targets"]),
        "n_train": int(data["n_train"]),
        "n_val": int(data["n_val"]),
        "n_test": int(data["n_test"]),
        "lr": float(args.lr),
        "residual_weight": float(args.residual_weight),
        "value_weight": float(args.value_weight),
        "diff_weight": float(args.diff_weight),
    }
    record.update(test_metrics)
    record.update(val_ar_metrics)

    metrics_path = os.path.join(output_dir, "variation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def append_metrics_csv(path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame(records)
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        subset = ["Dataset_ID", "Seed", "Modelo"]
        if all(col in df.columns for col in subset):
            df = df.drop_duplicates(subset=subset, keep="last")
    else:
        df = new_df
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Variation tuning on device={device}")

    records = []
    for dataset_id in args.datasets:
        for seed in args.seeds:
            record = train_one_run(dataset_id, seed, args, device)
            records.append(record)
            print(json.dumps(record, indent=2))
    if not args.dry_run:
        append_metrics_csv(args.metrics_csv, records)
        print(f"Wrote metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()
