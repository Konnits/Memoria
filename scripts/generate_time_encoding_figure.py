"""Generate a heatmap comparison for sinusoidal time encodings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sinusoidal_encoding(coordinates: np.ndarray, d_model: int) -> np.ndarray:
    """Return the fixed sinusoidal encoding used by the Transformer."""
    pair_index = np.arange(d_model) // 2
    omega = 10000.0 ** ((2.0 * pair_index) / float(d_model))
    arg = coordinates[:, None] / omega[None, :]

    encoding = np.empty((coordinates.shape[0], d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(arg[:, 0::2])
    encoding[:, 1::2] = np.cos(arg[:, 1::2])
    return encoding


def cell_edges(centers: np.ndarray) -> np.ndarray:
    """Return pcolormesh cell edges whose first and last values match centers."""
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


def make_figure(
    output_path: Path,
    n_tokens: int,
    n_continuous_tokens: int,
    d_model: int,
) -> None:
    t_max = float(n_tokens - 1)
    ordinal_positions = np.linspace(0.0, t_max, n_tokens)
    continuous_times = np.linspace(0.0, t_max, n_continuous_tokens)

    ordinal_encoding = sinusoidal_encoding(ordinal_positions, d_model)
    continuous_encoding = sinusoidal_encoding(continuous_times, d_model)
    x_edges = np.arange(d_model + 1, dtype=np.float64)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (
            ordinal_encoding,
            ordinal_positions,
            "Encoding sinusoidal tradicional",
            "Tiempo relativo en grilla regular",
        ),
        (
            continuous_encoding,
            continuous_times,
            "Encoding temporal continuo",
            "Tiempo relativo con muestras intermedias",
        ),
    ]

    image = None
    for ax, (values, times, title, ylabel) in zip(axes, panels):
        image = ax.pcolormesh(
            x_edges,
            cell_edges(times),
            values,
            shading="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a positional/time encoding heatmap comparison."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/images/encoding_posicional_comparacion.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--n-continuous-tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.output, args.n_tokens, args.n_continuous_tokens, args.d_model)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
