"""Generate a heatmap for the best learned Time2Vec encoding in the benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def sinusoidal_encoding(coordinates: np.ndarray, d_model: int) -> np.ndarray:
    pair_index = np.arange(d_model) // 2
    omega = 10000.0 ** ((2.0 * pair_index) / float(d_model))
    arg = coordinates[:, None] / omega[None, :]

    encoding = np.empty((coordinates.shape[0], d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(arg[:, 0::2])
    encoding[:, 1::2] = np.cos(arg[:, 1::2])
    return encoding


def cell_edges(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[0] = centers[0]
    edges[-1] = centers[-1]
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    return edges


def time_axis_ticks(t_max: float) -> np.ndarray:
    step = max(1, int(np.ceil(t_max / 6.0)))
    ticks = np.arange(0.0, t_max + 1.0, step, dtype=np.float64)
    if ticks[-1] != t_max:
        ticks = np.append(ticks, t_max)
    return ticks


def select_best_time2vec_row(benchmark_path: Path) -> pd.Series:
    df = pd.read_csv(benchmark_path)
    mask = df["Modelo"].astype(str).str.contains("Time2Vec|T2V", case=False, regex=True)
    candidates = df.loc[mask].copy()
    if candidates.empty:
        raise ValueError(f"No se encontraron modelos Time2Vec en {benchmark_path}")

    candidates["test_mse"] = pd.to_numeric(candidates["test_mse"], errors="coerce")
    candidates = candidates.dropna(subset=["test_mse"]).sort_values("test_mse")
    if candidates.empty:
        raise ValueError("Las filas Time2Vec no tienen test_mse válido.")
    return candidates.iloc[0]


def state_dict_from_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"No se pudo extraer un state_dict desde {path}")


def find_by_suffix(state_dict: dict[str, Any], suffix: str) -> torch.Tensor:
    matches = [value for key, value in state_dict.items() if key.endswith(suffix)]
    if not matches:
        available = "\n".join(key for key in state_dict if "time" in key.lower())
        raise KeyError(f"No se encontró el parámetro '*{suffix}'. Claves temporales:\n{available}")
    if len(matches) > 1:
        raise KeyError(f"Se encontró más de un parámetro con sufijo '*{suffix}'.")
    return matches[0].detach().cpu().to(torch.float32)


def learned_time2vec_encoding(state_dict: dict[str, Any], tau: np.ndarray) -> np.ndarray:
    linear_weight = find_by_suffix(state_dict, "time2vec.linear_weight").reshape(1)
    linear_bias = find_by_suffix(state_dict, "time2vec.linear_bias").reshape(1)
    periodic_weights = find_by_suffix(state_dict, "time2vec.periodic_weights").reshape(-1)
    periodic_biases = find_by_suffix(state_dict, "time2vec.periodic_biases").reshape(-1)

    tau_t = torch.as_tensor(tau, dtype=torch.float32).unsqueeze(-1)
    linear = tau_t * linear_weight + linear_bias
    periodic = torch.sin(tau_t * periodic_weights + periodic_biases)
    encoding = torch.cat([linear, periodic], dim=-1)

    try:
        norm_weight = find_by_suffix(state_dict, "time2vec.output_norm.weight")
        norm_bias = find_by_suffix(state_dict, "time2vec.output_norm.bias")
    except KeyError:
        return encoding.numpy()

    mean = encoding.mean(dim=-1, keepdim=True)
    var = encoding.var(dim=-1, unbiased=False, keepdim=True)
    encoding = (encoding - mean) / torch.sqrt(var + 1e-5)
    encoding = encoding * norm_weight + norm_bias
    return encoding.numpy()


def make_figure(
    benchmark_path: Path,
    experiments_dir: Path,
    output_path: Path,
    n_reference_tokens: int,
    n_continuous_tokens: int,
) -> None:
    best = select_best_time2vec_row(benchmark_path)
    dataset_id = int(best["Dataset_ID"])
    seed = int(best["Seed"])
    model_name = str(best["Modelo"])
    checkpoint_path = (
        experiments_dir / f"ds_{dataset_id}_seed_{seed}" / model_name / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint esperado: {checkpoint_path}")

    state_dict = state_dict_from_checkpoint(checkpoint_path)
    d_model = int(find_by_suffix(state_dict, "time2vec.linear_bias").numel()) + int(
        find_by_suffix(state_dict, "time2vec.periodic_weights").numel()
    )

    t_max = float(n_reference_tokens - 1)
    tau = np.linspace(0.0, t_max, n_continuous_tokens)
    fixed = sinusoidal_encoding(tau, d_model)
    learned = learned_time2vec_encoding(state_dict, tau)

    x_edges = np.arange(d_model + 1, dtype=np.float64)
    y_edges = cell_edges(tau)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.2),
        sharey=True,
        constrained_layout=True,
    )

    panels = [
        (fixed, "Sinusoidal fijo", "Tiempo relativo con muestras intermedias"),
        (learned, "Time2Vec aprendido", "Tiempo relativo con muestras intermedias"),
    ]
    vmax = max(1.0, float(np.nanpercentile(np.abs(np.concatenate([fixed, learned])), 98)))
    image = None
    for ax, (values, title, ylabel) in zip(axes, panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Dimension del embedding")
        ax.set_ylim(0.0, t_max)
        ax.set_xticks(np.arange(0, d_model, max(1, d_model // 8)))
        ax.set_yticks(time_axis_ticks(t_max))

    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.015)
    cbar.set_label("Valor del encoding")

    subtitle = (
        f"Mejor Time2Vec disponible: {model_name}, dataset {dataset_id}, "
        f"seed {seed}, MSE test={float(best['test_mse']):.4f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    print(subtitle)
    print(f"Checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a heatmap comparison using the best Time2Vec checkpoint."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("experiments/benchmark_final/benchmark_final.csv"),
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments/benchmark_final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/images/encoding_time2vec_aprendido.png"),
    )
    parser.add_argument("--n-reference-tokens", type=int, default=32)
    parser.add_argument("--n-continuous-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(
        benchmark_path=args.benchmark,
        experiments_dir=args.experiments_dir,
        output_path=args.output,
        n_reference_tokens=args.n_reference_tokens,
        n_continuous_tokens=args.n_continuous_tokens,
    )


if __name__ == "__main__":
    main()
