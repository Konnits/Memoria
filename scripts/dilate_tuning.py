from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.append(ROOT)
if SRC not in sys.path:
    sys.path.append(SRC)

from ar_finetuning import prepare_ar_data
from ts_transformer.models import TimeSeriesEncoderDecoder
from ts_transformer.training import DILATELoss
from ts_transformer.training.metrics import compute_regression_metrics
from ts_transformer.utils import load_data_config, set_global_seed
from variation_tuning import (
    _adaptive_time_scale,
    _batch_to_device,
    _load_model_weights,
    append_metrics_csv,
)
from ts_transformer.utils import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the best EncDec architecture with DILATE loss."
    )
    parser.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126])
    parser.add_argument("--exp-dir", type=str, default="experiments/benchmark_final")
    parser.add_argument(
        "--source-model",
        type=str,
        default="EncDec-Opt-Small-MT8_FT_AR_Contiguous",
    )
    parser.add_argument("--output-suffix", type=str, default="DILATE")
    parser.add_argument("--num-targets", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="experiments/benchmark_final/dilate_tuning.csv",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_encdec_model(base_data_cfg: Any, dataset_id: int, data: dict[str, Any]) -> TimeSeriesEncoderDecoder:
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


def make_dilate(args: argparse.Namespace) -> DILATELoss:
    return DILATELoss(
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        normalize_shape=True,
        normalize_temporal=True,
    )


def forward_teacher_forced(
    model: TimeSeriesEncoderDecoder,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        input_values=batch["input_values"],
        input_timestamps=batch["input_timestamps"],
        is_target_mask=batch["is_target_mask"],
        padding_mask=batch.get("padding_mask", None),
        return_dict=False,
    )


def generate_autoregressive(
    model: TimeSeriesEncoderDecoder,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target_values = batch["target_values"]
    num_targets = target_values.shape[1]
    padding_mask = batch.get("padding_mask", None)
    history_padding_mask = padding_mask[:, :-num_targets] if padding_mask is not None else None
    return model.generate(
        history_values=batch["input_values"][:, :-num_targets, :],
        history_timestamps=batch["input_timestamps"][:, :-num_targets],
        target_timestamps=batch["input_timestamps"][:, -num_targets:],
        history_padding_mask=history_padding_mask,
    )


@torch.no_grad()
def _append_regression_batches(
    preds: torch.Tensor,
    targets: torch.Tensor,
    all_preds: list[torch.Tensor],
    all_targets: list[torch.Tensor],
) -> None:
    all_preds.append(preds.detach().cpu())
    all_targets.append(targets.detach().cpu())


def evaluate_teacher_forced(
    model: TimeSeriesEncoderDecoder,
    loader,
    loss_fn: DILATELoss,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    shape = torch.zeros((), device=device, dtype=torch.float64)
    temporal = torch.zeros((), device=device, dtype=torch.float64)
    count = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_val_batches and batch_idx >= args.max_val_batches:
            break
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            preds = forward_teacher_forced(model, batch)
        parts = loss_fn.forward_parts(preds, batch["target_values"])
        bsz = int(batch["target_values"].shape[0])
        total = total + parts.total.detach().to(torch.float64) * bsz
        shape = shape + parts.shape.detach().to(torch.float64) * bsz
        temporal = temporal + parts.temporal.detach().to(torch.float64) * bsz
        count += bsz

    denom = max(1, count)
    return {
        "val_loss": float((total / denom).detach().cpu().item()),
        "val_dilate_shape": float((shape / denom).detach().cpu().item()),
        "val_dilate_temporal": float((temporal / denom).detach().cpu().item()),
    }


def evaluate_autoregressive(
    model: TimeSeriesEncoderDecoder,
    loader,
    loss_fn: DILATELoss,
    args: argparse.Namespace,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total = torch.zeros((), device=device, dtype=torch.float64)
    shape = torch.zeros((), device=device, dtype=torch.float64)
    temporal = torch.zeros((), device=device, dtype=torch.float64)
    count = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_test_batches and batch_idx >= args.max_test_batches:
            break
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            preds = generate_autoregressive(model, batch)
        targets = batch["target_values"]
        parts = loss_fn.forward_parts(preds, targets)
        bsz = int(targets.shape[0])
        total = total + parts.total.detach().to(torch.float64) * bsz
        shape = shape + parts.shape.detach().to(torch.float64) * bsz
        temporal = temporal + parts.temporal.detach().to(torch.float64) * bsz
        count += bsz
        _append_regression_batches(preds, targets, all_preds, all_targets)

    if not all_preds:
        return {}

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(
        preds_cat.reshape(-1, preds_cat.shape[-1]),
        targets_cat.reshape(-1, targets_cat.shape[-1]),
        prefix=prefix,
    )

    if preds_cat.shape[1] > 1:
        pred_diff = preds_cat[:, 1:, :] - preds_cat[:, :-1, :]
        target_diff = targets_cat[:, 1:, :] - targets_cat[:, :-1, :]
        metrics.update(
            compute_regression_metrics(
                pred_diff.reshape(-1, pred_diff.shape[-1]),
                target_diff.reshape(-1, target_diff.shape[-1]),
                prefix=f"{prefix}diff_",
            )
        )
        metrics[f"{prefix}direction_acc"] = float(
            (torch.sign(pred_diff) == torch.sign(target_diff)).float().mean().item()
        )

    denom = max(1, count)
    metrics[f"{prefix}dilate"] = float((total / denom).detach().cpu().item())
    metrics[f"{prefix}dilate_shape"] = float((shape / denom).detach().cpu().item())
    metrics[f"{prefix}dilate_temporal"] = float((temporal / denom).detach().cpu().item())
    return metrics


def save_checkpoint(
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
            "dilate_config": vars(args),
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
        raise FileNotFoundError(f"No data found for dataset {dataset_id}")

    model = build_encdec_model(base_data_cfg, dataset_id, data).to(device)
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

    loss_fn = make_dilate(args)

    if args.dry_run:
        batch = next(iter(data["train_loader"]))
        batch = _batch_to_device(batch, device)
        preds = forward_teacher_forced(model, batch)
        parts = loss_fn.forward_parts(preds, batch["target_values"])
        parts.total.backward()
        return {
            "Dataset_ID": dataset_id,
            "Seed": seed,
            "Modelo": output_model,
            "status": "dry_run",
            "pred_shape": list(preds.shape),
            "dry_run_dilate": float(parts.total.detach().cpu().item()),
            "dry_run_shape": float(parts.shape.detach().cpu().item()),
            "dry_run_temporal": float(parts.temporal.detach().cpu().item()),
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
        total = torch.zeros((), device=device, dtype=torch.float64)
        shape = torch.zeros((), device=device, dtype=torch.float64)
        temporal = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        running_batches = 0
        epoch_start = time.time()
        total_train_batches = (
            min(len(data["train_loader"]), int(args.max_train_batches))
            if args.max_train_batches
            else len(data["train_loader"])
        )

        for batch_idx, batch in enumerate(data["train_loader"]):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            preds = forward_teacher_forced(model, batch)
            parts = loss_fn.forward_parts(preds, batch["target_values"])
            parts.total.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            bsz = int(batch["target_values"].shape[0])
            total = total + parts.total.detach().to(torch.float64) * bsz
            shape = shape + parts.shape.detach().to(torch.float64) * bsz
            temporal = temporal + parts.temporal.detach().to(torch.float64) * bsz
            count += bsz
            running_batches += 1

            log_every = int(args.log_every_n_steps or 0)
            if log_every > 0 and (
                running_batches % log_every == 0
                or running_batches == total_train_batches
            ):
                denom = max(1, count)
                elapsed = time.time() - epoch_start
                train_avg = float((total / denom).detach().cpu().item())
                shape_avg = float((shape / denom).detach().cpu().item())
                temporal_avg = float((temporal / denom).detach().cpu().item())
                print(
                    f"[DILATE ds={dataset_id} seed={seed}] "
                    f"Epoch {epoch:03d} Step {running_batches:05d}/{total_train_batches:05d} "
                    f"- train_dilate={train_avg:.6f}, "
                    f"shape={shape_avg:.6f}, "
                    f"temporal={temporal_avg:.6f}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        denom = max(1, count)
        val_metrics = evaluate_teacher_forced(model, data["val_loader"], loss_fn, args, device)
        train_dilate = float((total / denom).detach().cpu().item())
        train_shape = float((shape / denom).detach().cpu().item())
        train_temporal = float((temporal / denom).detach().cpu().item())
        row = {
            "epoch": float(epoch),
            "train_dilate": train_dilate,
            "train_shape": train_shape,
            "train_temporal": train_temporal,
            **val_metrics,
        }
        history.append(row)
        print(
            f"[DILATE ds={dataset_id} seed={seed}] epoch={epoch:02d} "
            f"train={row['train_dilate']:.6f} val={val_metrics['val_loss']:.6f} "
            f"shape={val_metrics['val_dilate_shape']:.6f} "
            f"time={val_metrics['val_dilate_temporal']:.6f}"
        )

        current_val = float(val_metrics["val_loss"])
        if current_val < best_val - 1e-5:
            best_val = current_val
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val, args)
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    train_time = time.time() - train_start
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_autoregressive(
        model,
        data["test_loader"],
        loss_fn,
        args,
        device,
        prefix="test_dilate_",
    )
    val_ar_metrics = evaluate_autoregressive(
        model,
        data["val_loader"],
        loss_fn,
        args,
        device,
        prefix="val_dilate_ar_",
    )

    with open(os.path.join(output_dir, "dilate_history.json"), "w", encoding="utf-8") as f:
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
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
    }
    record.update(test_metrics)
    record.update(val_ar_metrics)

    with open(os.path.join(output_dir, "dilate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DILATE tuning on device={device}")
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
