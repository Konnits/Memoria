import os
import pandas as pd
import numpy as np

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

    # 2. Calcular la media y desviación estándar para cada modelo a través de todos datasets
    summary_stats = seed_avg.groupby("Modelo")[metrics].agg(['mean', 'std'])
    
    # Aplanar el MultiIndex de columnas
    summary_stats.columns = [f"{metric}_{stat}" for metric, stat in summary_stats.columns]
    summary_stats = summary_stats.reset_index()

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

    # 4. Ordenar desde el MEJOR (menor error medio MSE) al PEOR
    final_df = final_df.sort_values(by="test_mse_mean", ascending=True).reset_index(drop=True)

    # 5. Formatear output para mostrar en consola de manera limpia y clara para la tesis
    print("=" * 110)
    print(" " * 35 + "RANKING DE MODELOS PARA LA TESIS")
    print("=" * 110)
    print("Metodología Aplicada (alineada con cap6_experimentos.tex):")
    print(" 1. Los resultados de las semillas se promedian a nivel de dataset para estabilizar la varianza interna.")
    print(" 2. Se reporta la Media ± Desviación Estándar inter-datasets para cuantificar desempeño y dispersión global.")
    print(" 3. 'Victorias': Número de datasets en los que el modelo logró el MSE promedio MÁS BAJO frente al resto.")
    print(" 4. Ranking general descendente: Ordenados jerárquicamente por el menor MSE Medio.")
    print("-" * 110)
    
    # Crear un DataFrame formateado para impresión bonita
    print_df = pd.DataFrame()
    print_df["Rank"] = range(1, len(final_df) + 1)
    print_df["Modelo"] = final_df["Modelo"]
    
    for metric in metrics:
        clean_name = metric.replace("test_", "").upper()
        # Formato: Media ± STD
        print_df[f"{clean_name} (Mean ± std)"] = final_df.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}", axis=1
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
