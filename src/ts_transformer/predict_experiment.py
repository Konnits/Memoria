from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from ts_transformer.inference import ExperimentPredictor


def _parse_csv_arg(raw: str, cast):
    values = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if item == "":
            continue
        values.append(cast(item))
    if not values:
        raise ValueError("La lista no puede estar vacía.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga un experimento entrenado y genera predicciones sobre un CSV."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Carpeta del experimento, por ejemplo experiments/test_001.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="CSV con los datos históricos a usar en inferencia.",
    )
    parser.add_argument(
        "--history-start-index",
        type=int,
        required=True,
        help="Índice inicial inclusivo de la ventana histórica dentro del CSV.",
    )
    parser.add_argument(
        "--history-end-index",
        type=int,
        required=True,
        help="Índice final exclusivo de la ventana histórica dentro del CSV.",
    )
    parser.add_argument(
        "--future-indexes",
        type=str,
        default=None,
        help="Índices futuros del CSV separados por comas. Ejemplo: 50,60,70",
    )
    parser.add_argument(
        "--future-timestamps",
        type=str,
        default=None,
        help="Timestamps futuros separados por comas. Ejemplo: 15818100,15818400",
    )
    parser.add_argument(
        "--future-offsets",
        type=str,
        default=None,
        help="Offsets futuros separados por comas, en la misma unidad del timestamp.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Dispositivo de inferencia, por ejemplo "cpu" o "cuda".',
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Ruta opcional para guardar las predicciones como CSV.",
    )
    return parser.parse_args()


def _resolve_future_timestamps(
    df: pd.DataFrame,
    predictor: ExperimentPredictor,
    args: argparse.Namespace,
) -> Sequence[float]:
    specified = [
        args.future_indexes is not None,
        args.future_timestamps is not None,
        args.future_offsets is not None,
    ]
    if sum(specified) != 1:
        raise ValueError(
            "Debes indicar exactamente uno de --future-indexes, --future-timestamps o --future-offsets."
        )

    if args.future_indexes is not None:
        future_indexes = _parse_csv_arg(args.future_indexes, int)
        max_index = len(df) - 1
        for idx in future_indexes:
            if idx < 0 or idx > max_index:
                raise IndexError(
                    f"future-index {idx} fuera de rango. Máximo permitido: {max_index}."
                )
        return df.iloc[future_indexes][predictor.time_column].tolist()

    if args.future_timestamps is not None:
        return _parse_csv_arg(args.future_timestamps, float)

    offsets = _parse_csv_arg(args.future_offsets, float)
    history_timestamps = df.iloc[
        args.history_start_index:args.history_end_index
    ][predictor.time_column]
    if history_timestamps.empty:
        raise ValueError("La ventana histórica no puede estar vacía.")
    last_timestamp = float(history_timestamps.iloc[-1])
    return [last_timestamp + offset for offset in offsets]


def main() -> None:
    args = parse_args()

    predictor = ExperimentPredictor.from_experiment_dir(
        args.experiment_dir,
        device=args.device,
    )
    df = pd.read_csv(args.csv_path)

    if predictor.time_column not in df.columns:
        raise ValueError(
            f"No existe la columna temporal {predictor.time_column!r} en {args.csv_path}."
        )

    if args.history_start_index < 0 or args.history_end_index > len(df):
        raise IndexError("La ventana histórica cae fuera del rango del CSV.")
    if args.history_start_index >= args.history_end_index:
        raise ValueError("history-start-index debe ser menor que history-end-index.")

    history = df.iloc[args.history_start_index:args.history_end_index].copy()
    future_timestamps = _resolve_future_timestamps(df, predictor, args)

    if predictor.use_event_tokens:
        if len(predictor.feature_columns) != 1:
            raise ValueError(
                "El CLI actual asume un único feature para modelos con event tokens."
            )
        past_values = history[predictor.feature_columns[0]]
    else:
        missing = [col for col in predictor.feature_columns if col not in history.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")
        past_values = history[list(predictor.feature_columns)]

    predictions = predictor.predict(
        past_values=past_values,
        past_timestamps=history[predictor.time_column],
        future_timestamps=future_timestamps,
        return_dataframe=True,
    )

    print(predictions.to_string(index=False))

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"\nPredicciones guardadas en {output_path}")


if __name__ == "__main__":
    main()