"""
fill_thesis_tables.py — Llena las tablas del capítulo 7 con los resultados finales.

Lee los CSVs de experiments/benchmark_final/ y genera el LaTeX listo
para copiar y pegar en cap7_resultados_discusion.tex.

Uso:
  python scripts/fill_thesis_tables.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from statistical_analysis import (
    compute_pairwise_comparison,
    generate_summary_table,
    bootstrap_ci,
)

EXP_DIR = "experiments/benchmark_final"


def main():
    csv_path = os.path.join(EXP_DIR, "benchmark_final.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró {csv_path}. Ejecuta benchmark_final.py primero.")
        return

    df = pd.read_csv(csv_path)
    test_metrics = ["test_mse", "test_rmse", "test_mae"]
    available = [m for m in test_metrics if m in df.columns]

    if not available:
        print("[ERROR] No se encontraron métricas de test en el CSV.")
        return

    n_datasets = df["Dataset_ID"].nunique()
    n_seeds = df["Seed"].nunique()
    models = sorted(df["Modelo"].unique())

    print(f"\n{'='*70}")
    print(f"RESULTADO DE {n_datasets} datasets, {n_seeds} semillas, {len(models)} modelos")
    print(f"{'='*70}\n")

    # =====================================================================
    # Tabla 1: Resultados generales
    # =====================================================================
    summary = generate_summary_table(df, available)
    print("TABLA 1 — Resultados generales (LaTeX):")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Modelo & MSE & RMSE & MAE \\\\")
    print("\\midrule")

    for _, row in summary.iterrows():
        name = row["Modelo"]
        parts = []
        for m in available:
            parts.append(row[m])
        line = f"{name} & " + " & ".join(parts) + " \\\\"
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}\n")

    # =====================================================================
    # Tabla 2: Victorias por modelo
    # =====================================================================
    # Promediar semillas por (dataset, modelo) y luego encontrar ganador
    agg = df.groupby(["Dataset_ID", "Modelo"])["test_mse"].mean().reset_index()
    pivot = agg.pivot(index="Dataset_ID", columns="Modelo", values="test_mse")
    winners = pivot.idxmin(axis=1)
    win_counts = winners.value_counts()

    print("TABLA 2 — Victorias por modelo:")
    for model in models:
        count = win_counts.get(model, 0)
        print(f"  {model}: {count}/{n_datasets}")
    print()

    # =====================================================================
    # Tabla 3: Comparaciones emparejadas
    # =====================================================================
    print("TABLA 3 — Comparaciones emparejadas (Custom vs cada baseline):")
    reference = "Custom"
    others = [m for m in models if m != reference]

    for metric in available:
        print(f"\n  --- {metric} ---")
        for other in others:
            comp = compute_pairwise_comparison(df, reference, other, metric)
            if "error" in comp:
                print(f"  vs {other}: {comp['error']}")
                continue
            print(
                f"  vs {other}: "
                f"Victorias {comp['wins_a']}/{comp['n_datasets']}, "
                f"Mejora: {comp['mean_improvement_pct']:.1f}%, "
                f"Wilcoxon W={comp['wilcoxon_stat']}, p={comp['wilcoxon_p']:.6f}"
            )

    # =====================================================================
    # Tabla 4: Ablación
    # =====================================================================
    ablation_models = [m for m in models if m.startswith("No")]
    if ablation_models:
        print("\n\nTABLA 4 — Ablación (LaTeX):")
        agg2 = df.groupby(["Dataset_ID", "Modelo"])["test_mse"].mean().reset_index()
        model_means = agg2.groupby("Modelo")["test_mse"].agg(["mean", "std"])

        if reference in model_means.index:
            ref_mse = model_means.loc[reference, "mean"]
            print(f"  Custom: {ref_mse:.4f} ± {model_means.loc[reference, 'std']:.4f} (referencia)")
            for ab in ablation_models:
                if ab in model_means.index:
                    ab_mse = model_means.loc[ab, "mean"]
                    ab_std = model_means.loc[ab, "std"]
                    delta = ((ab_mse - ref_mse) / ref_mse) * 100
                    print(f"  {ab}: {ab_mse:.4f} ± {ab_std:.4f} (+{delta:.1f}%)")

    # =====================================================================
    # Tabla 5: Costo computacional
    # =====================================================================
    cost_cols = ["n_params_trainable", "n_params_total", "train_time_s", "epochs_run"]
    avail_cost = [c for c in cost_cols if c in df.columns]
    if avail_cost:
        print("\n\nTABLA 5 — Costo computacional:")
        cost = df.groupby("Modelo")[avail_cost].mean().round(1)
        for model_name in cost.index:
            row = cost.loc[model_name]
            params = int(row.get("n_params_trainable", 0))
            time_s = row.get("train_time_s", 0)
            epochs = int(row.get("epochs_run", 0))
            print(f"  {model_name}: {params:,} params, {time_s:.0f}s, {epochs} epochs")

    # =====================================================================
    # Bootstrap CIs
    # =====================================================================
    print("\n\nINTERVALOS BOOTSTRAP 95%:")
    agg3 = df.groupby(["Dataset_ID", "Modelo"])["test_mse"].mean().reset_index()
    for model_name in models:
        vals = agg3[agg3["Modelo"] == model_name]["test_mse"].values
        if len(vals) > 0:
            mean, lo, hi = bootstrap_ci(vals)
            print(f"  {model_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n{'='*70}")
    print("Copia estos valores a cap7_resultados_discusion.tex")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
