import os
import pandas as pd
import numpy as np


def trimmed_mean(values: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")
    vals = np.sort(vals)
    k = int(np.floor(vals.size * proportion_to_cut))
    if vals.size - 2 * k <= 0:
        return float(vals.mean())
    return float(vals[k : vals.size - k].mean())


def summarize_metric(values: pd.Series) -> dict:
    arr = values.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "trimmed_mean": np.nan,
        }

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "trimmed_mean": float(trimmed_mean(arr, 0.1)),
    }

def main():
    csv_path = "experiments/benchmark_final/benchmark_final.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required_cols = ["Dataset_ID", "Modelo", "Seed", "test_mse", "test_rmse", "test_mae"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: La columna {col} no se encuentra en el CSV.")
            return
            
    # La tesis declara: "Media ± desviación estándar: se promedian las semillas por dataset..."
    # 1. Promediar las semillas para cada par (Dataset_ID, Modelo)
    metrics = ["test_mse", "test_rmse", "test_mae"]
    if "test_mape" in df.columns:
        metrics.append("test_mape")
        
    seed_avg = df.groupby(["Dataset_ID", "Modelo"])[metrics].mean().reset_index()

    # 2. Calcular estadísticas por modelo sobre datasets (robustas + clásicas)
    rows = []
    for model_name, grp in seed_avg.groupby("Modelo"):
        row = {"Modelo": model_name}
        for metric in metrics:
            s = summarize_metric(grp[metric])
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
            row[f"{metric}_median"] = s["median"]
            row[f"{metric}_q1"] = s["q1"]
            row[f"{metric}_q3"] = s["q3"]
            row[f"{metric}_iqr"] = s["iqr"]
            row[f"{metric}_trimmed_mean"] = s["trimmed_mean"]
        rows.append(row)
    summary_stats = pd.DataFrame(rows)

    # 3. Calcular "Victorias por dataset"
    # "se cuenta en cuántos de los 17 datasets cada modelo obtiene el menor error"
    # Lo haremos priorizando el MSE (Mean Squared Error), que es la pérdida principal.
    idx_min_mse = seed_avg.groupby("Dataset_ID")["test_mse"].idxmin()
    best_models = seed_avg.loc[idx_min_mse]
    wins_counts = best_models["Modelo"].value_counts().reset_index()
    wins_counts.columns = ["Modelo", "Victorias (MSE)"]

    # Hacer merge de las victorias con el summary
    final_df = pd.merge(summary_stats, wins_counts, on="Modelo", how="left")
    final_df["Victorias (MSE)"] = final_df["Victorias (MSE)"].fillna(0).astype(int)

    # 4. Ordenar desde el MEJOR al PEOR por criterio robusto (trimmed mean),
    # con desempate por mediana de MSE.
    final_df = final_df.sort_values(
        by=["test_mse_trimmed_mean", "test_mse_median"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # 5. Formatear output para mostrar en consola de manera limpia y clara para la tesis
    print("=" * 110)
    print(" " * 35 + "RANKING DE MODELOS PARA LA TESIS")
    print("=" * 110)
    print("Metodología Aplicada (alineada con cap6_experimentos.tex):")
    print(" 1. Los resultados de las semillas se promedian a nivel de dataset para estabilizar la varianza interna.")
    print(" 2. Se reportan estadísticas clásicas (mean±std) y robustas (mediana + IQR, trimmed mean 10%).")
    print(" 3. 'Victorias': Número de datasets en los que el modelo logró el MSE promedio MÁS BAJO frente al resto.")
    print(" 4. Ranking robusto: ordenado por menor MSE trimmed mean (10%), con desempate por mediana.")
    print("-" * 110)
    
    # Crear un DataFrame formateado para impresión bonita
    print_df = pd.DataFrame()
    print_df["Rank"] = range(1, len(final_df) + 1)
    print_df["Modelo"] = final_df["Modelo"]
    print_df["MSE TrimmedMean(10%)"] = final_df["test_mse_trimmed_mean"].map(lambda x: f"{x:.4f}")
    
    for metric in metrics:
        clean_name = metric.replace("test_", "").upper()
        # Formato clásico y robusto
        print_df[f"{clean_name} (Mean ± std)"] = final_df.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}", axis=1
        )
        print_df[f"{clean_name} (Median [IQR])"] = final_df.apply(
            lambda row: f"{row[f'{metric}_median']:.4f} [{row[f'{metric}_q1']:.4f}, {row[f'{metric}_q3']:.4f}]",
            axis=1,
        )
        
    print_df["Victorias"] = final_df["Victorias (MSE)"]

    # Configurar opciones de impresión de pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    
    # Imprimir usando string estándar
    print(print_df.to_string(index=False))
    
    print("=" * 110)
    
    # Exportar también en Markdown y CSV para fácil manipulación
    out_md = "experiments/benchmark_final/ranking_tesis.md"
    out_csv = "experiments/benchmark_final/ranking_tesis_export.csv"
    
    try:
        print_df.to_markdown(out_md, index=False)
        final_df.to_csv(out_csv, index=False)
        print(f"[*] La tabla formateada fue exportada a Markdown en: {out_md}")
        print(f"[*] Los datos listos para tablas en LaTeX se guardaron en: {out_csv}")
    except ImportError:
        # En caso de que el módulo tabulate no esté instalado para to_markdown
        print_df.to_csv(out_csv, index=False)
        print(f"[*] Los datos listos para tablas se guardaron en CSV en: {out_csv}")

if __name__ == "__main__":
    main()
