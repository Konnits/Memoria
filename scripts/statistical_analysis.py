"""
Módulo de análisis estadístico para comparación de modelos.

Proporciona:
  - Test de Wilcoxon signed-rank (comparación emparejada por dataset).
  - Intervalos de confianza bootstrap.
  - Tabla resumen con media ± std, victorias, y p-values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def trimmed_mean(values: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    """
    Calcula media recortada eliminando una fracción en ambos extremos.
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")

    vals = np.sort(vals)
    k = int(np.floor(vals.size * proportion_to_cut))
    if vals.size - 2 * k <= 0:
        return float(vals.mean())
    return float(vals[k : vals.size - k].mean())


def wilcoxon_signed_rank(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    alternative: str = "less",
) -> Tuple[float, float]:
    """
    Test de Wilcoxon signed-rank para comparar dos modelos emparejados por dataset.

    Parameters
    ----------
    metric_a : array de métricas del modelo A (uno por dataset)
    metric_b : array de métricas del modelo B (uno por dataset)
    alternative : "less" si H1 es que A < B (A es mejor cuando menor es mejor)

    Returns
    -------
    statistic, p_value
    """
    diffs = metric_a - metric_b
    # Si todas las diferencias son cero, no hay señal
    if np.all(diffs == 0):
        return 0.0, 1.0
    # Necesitamos al menos 6 pares para que el test tenga sentido
    if len(diffs) < 6:
        # Con pocos pares, usamos un test de signos simple
        n_neg = np.sum(diffs < 0)
        n_pos = np.sum(diffs > 0)
        n_total = n_neg + n_pos
        if n_total == 0:
            return 0.0, 1.0
        # Test binomial: probabilidad de ver n_neg o más bajo H0 (p=0.5)
        result = stats.binomtest(n_neg, n_total, 0.5, alternative="greater")
        p_val = result.pvalue
        return float(n_neg), float(p_val)

    stat, p_val = stats.wilcoxon(
        metric_a, metric_b, alternative=alternative, zero_method="wilcox"
    )
    return float(stat), float(p_val)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calcula intervalo de confianza bootstrap para la media.

    Returns
    -------
    mean, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = 1 - confidence
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(ci_lo), float(ci_hi)


def compute_pairwise_comparison(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_col: str = "test_mse",
    dataset_col: str = "Dataset_ID",
    model_col: str = "Modelo",
    seed_col: str = "Seed",
) -> Dict:
    """
    Compara dos modelos usando métricas emparejadas por dataset.

    Si hay múltiples semillas, promedia primero por (dataset, modelo).

    Returns
    -------
    dict con: wins_a, wins_b, ties, mean_improvement_pct,
              wilcoxon_stat, wilcoxon_p, bootstrap_ci_a, bootstrap_ci_b
    """
    # Promediar semillas por dataset
    grouped = (
        df[df[model_col].isin([model_a, model_b])]
        .groupby([dataset_col, model_col])[metric_col]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index=dataset_col, columns=model_col, values=metric_col)

    # Solo datasets donde ambos modelos tienen datos
    pivot = pivot.dropna()
    if pivot.empty:
        return {"error": f"No hay datasets en común entre {model_a} y {model_b}"}

    vals_a = pivot[model_a].values
    vals_b = pivot[model_b].values

    wins_a = int(np.sum(vals_a < vals_b))
    wins_b = int(np.sum(vals_b < vals_a))
    ties = int(np.sum(vals_a == vals_b))

    # Mejora porcentual media: (B - A) / B * 100
    improvement_pct = ((vals_b - vals_a) / np.abs(vals_b).clip(1e-8)) * 100
    mean_improvement = float(improvement_pct.mean())

    w_stat, w_p = wilcoxon_signed_rank(vals_a, vals_b, alternative="less")

    mean_a, ci_lo_a, ci_hi_a = bootstrap_ci(vals_a)
    mean_b, ci_lo_b, ci_hi_b = bootstrap_ci(vals_b)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric_col,
        "n_datasets": len(vals_a),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "mean_improvement_pct": round(mean_improvement, 2),
        "wilcoxon_stat": round(w_stat, 4),
        "wilcoxon_p": round(w_p, 6),
        "mean_a": round(mean_a, 6),
        "ci_95_a": f"[{ci_lo_a:.6f}, {ci_hi_a:.6f}]",
        "mean_b": round(mean_b, 6),
        "ci_95_b": f"[{ci_lo_b:.6f}, {ci_hi_b:.6f}]",
    }


def generate_summary_table(
    df: pd.DataFrame,
    metrics: List[str],
    model_col: str = "Modelo",
    dataset_col: str = "Dataset_ID",
    seed_col: str = "Seed",
) -> pd.DataFrame:
    """
    Genera tabla resumen por modelo sobre promedios por dataset.

    Incluye métricas clásicas (media ± std) y robustas:
    - mediana
    - IQR
    - media recortada (trimmed mean, 10%)

    Returns
    -------
    DataFrame con columnas: Modelo, metric_mean, metric_std para cada métrica
    """
    # Promediar semillas por (dataset, modelo)
    group_cols = [dataset_col, model_col]
    agg = df.groupby(group_cols)[metrics].mean().reset_index()

    # Ahora calcular media ± std sobre datasets
    rows = []
    for model_name, grp in agg.groupby(model_col):
        row = {"Modelo": model_name}
        for m in metrics:
            vals = grp[m].dropna().values
            if vals.size == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_median"] = float("nan")
                row[f"{m}_q1"] = float("nan")
                row[f"{m}_q3"] = float("nan")
                row[f"{m}_iqr"] = float("nan")
                row[f"{m}_trimmed_mean"] = float("nan")
                row[f"{m}"] = "nan ± nan"
                row[f"{m}_robust"] = "nan [IQR=nan]"
                continue

            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            med = float(np.median(vals))
            iqr = float(q3 - q1)
            tmean = trimmed_mean(vals, proportion_to_cut=0.1)

            row[f"{m}_mean"] = round(float(vals.mean()), 6)
            row[f"{m}_std"] = round(float(vals.std()), 6)
            row[f"{m}_median"] = round(med, 6)
            row[f"{m}_q1"] = round(q1, 6)
            row[f"{m}_q3"] = round(q3, 6)
            row[f"{m}_iqr"] = round(iqr, 6)
            row[f"{m}_trimmed_mean"] = round(float(tmean), 6)
            row[f"{m}"] = f"{vals.mean():.4f} ± {vals.std():.4f}"
            row[f"{m}_robust"] = f"{med:.4f} [IQR={iqr:.4f}]"
        rows.append(row)

    return pd.DataFrame(rows)


def generate_full_report(
    df: pd.DataFrame,
    reference_model: str,
    metrics: List[str] = None,
    model_col: str = "Modelo",
    dataset_col: str = "Dataset_ID",
    seed_col: str = "Seed",
) -> str:
    """
    Genera un reporte textual completo de la comparación entre modelos.
    """
    if metrics is None:
        metrics = ["test_mse", "test_rmse", "test_mae"]

    models = sorted(df[model_col].unique())
    lines = ["=" * 70, "REPORTE ESTADÍSTICO DE COMPARACIÓN DE MODELOS", "=" * 70, ""]

    # Tabla resumen
    summary = generate_summary_table(df, metrics, model_col, dataset_col, seed_col)
    lines.append("RESUMEN POR MODELO (media ± std y robustez sobre datasets):")
    lines.append(summary.to_string(index=False))
    lines.append("")

    # Comparaciones emparejadas vs modelo de referencia
    other_models = [m for m in models if m != reference_model]
    for m in metrics:
        lines.append(f"\n--- Comparación emparejada: {m} ---")
        for other in other_models:
            comp = compute_pairwise_comparison(
                df, reference_model, other, m, dataset_col, model_col, seed_col
            )
            if "error" in comp:
                lines.append(f"  {reference_model} vs {other}: {comp['error']}")
                continue
            lines.append(
                f"  {reference_model} vs {other}: "
                f"Victorias {comp['wins_a']}/{comp['n_datasets']}, "
                f"Mejora media: {comp['mean_improvement_pct']:.1f}%, "
                f"Wilcoxon p={comp['wilcoxon_p']:.6f}"
            )
        lines.append("")

    return "\n".join(lines)
