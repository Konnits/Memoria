# Memoria — Benchmark de Transformers para Series de Tiempo Irregulares

Un sistema completo de investigación para el benchmark reproducible de modelos Transformer sobre series de tiempo con observaciones asíncronas (no equiespaciadas). El proyecto genera datos sintéticos controlados, compara 9 arquitecturas, optimiza hiperparámetros con Optuna y produce resultados estadísticamente rigurosos para una tesis doctoral.

## Tabla de Contenidos

- [Propósito](#propósito)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Modelos Comparados](#modelos-comparados)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Protocolo de Benchmark](#protocolo-de-benchmark)
- [Generador de Datos Sintéticos](#generador-de-datos-sintéticos)
- [Búsqueda de Hiperparámetros](#búsqueda-de-hiperparámetros)
- [Pipeline de Datos](#pipeline-de-datos)
- [Training Loop](#training-loop)
- [Inferencia](#inferencia)
- [Tests](#tests)
- [Tesis](#tesis)
- [Stack Tecnológico](#stack-tecnológico)
- [Contribuciones Técnicas](#contribuciones-técnicas)

---

## Propósito

Este proyecto implementa un **benchmark controlado** para evaluar la capacidad de modelos Transformer en tareas de forecasting sobre series de tiempo donde las observaciones llegan en **tiempos irregulares**, simulando escenarios del mundo real como datos de sensores IoT, transacciones financieras o eventos médicos.

El enfoque científico es comparar arquitecturas sobre **datos sintéticos generados controladamente**, donde la irregularidad temporal es una propiedad medible y reproducible. Las conclusiones son válidas para los procesos sintéticos cubiertos por el generador, no para validez externa sobre datos reales.

---

## Arquitectura del Proyecto

```
Memoria/
├── src/
│   ├── ts_transformer/                # Núcleo del proyecto
│   │   ├── models/                    # Modelos de red neuronal
│   │   │   ├── time_series_transformer.py    ← Modelo "Custom" principal
│   │   │   ├── time_series_encoder_decoder.py ← Modelo "EncDec-AR"
│   │   │   ├── attention.py           ← Self-attention + Cross-attention
│   │   │   ├── transformer_blocks.py  ← Encoder/Decoder blocks (Pre-LN)
│   │   │   ├── heads.py               ← RegressionHead + AttentionPooling
│   │   │   └── masking.py             ← create_causal_mask
│   │   ├── features/                  ← Componentes de embedding
│   │   │   ├── value_embedding.py     ← FeatureEmbedding (proyección de features)
│   │   │   ├── time_encoding.py       ← TimePositionalEncoding + Time2Vec
│   │   │   ├── target_flag_embedding.py ← Flag historia vs target
│   │   │   ├── sensor_embedding.py    ← Embedding por sensor
│   │   │   └── temporal_attention_bias.py ← Bias temporal tipo ALiBi
│   │   ├── data/                      ← Pipeline de datos
│   │   │   ├── timeseries_dataset.py  ← TimeSeriesDataset + EventTimeSeriesDataset
│   │   │   ├── sequence_builder.py    ← SequenceBuilder + AutoregressiveSequenceBuilder
│   │   │   ├── collate.py             ← build_collate_fn (left-padding)
│   │   │   ├── scalers.py             ← StandardScaler + MinMaxScaler
│   │   │   ├── splits.py              ← time_series_train_val_test_split
│   │   │   └── samplers.py            ← BucketBatchSampler
│   │   ├── training/                  ← Loop de entrenamiento
│   │   │   ├── train_loop.py          ← Trainer + TrainingConfig
│   │   │   ├── losses.py              ← get_loss_fn (MSE, MAE, Huber, DILATE)
│   │   │   ├── dilate_loss.py         ← DILATELoss (soft-DTW + distortion temporal)
│   │   │   ├── metrics.py             ← compute_regression_metrics
│   │   │   └── optimizers.py          ← OptimizerConfig + build_optimizer + WarmupCosineScheduler
│   │   ├── inference/                 ← Inferencia
│   │   │   ├── predictor.py           ← Predictor + build_predictor_from_experiment
│   │   │   ├── experiment_predictor.py ← ExperimentPredictor + RollingForecaster
│   │   │   └── rolling_forecast.py    ← RollingForecaster
│   │   ├── train.py                   ← CLI de entrenamiento standalone
│   │   ├── predict_experiment.py      ← CLI de predicción
│   │   └── hyperparameter_search.py   ← Grid/random search de hiperparámetros
│   └── data/                          ← (git-ignored) IrregularTimeSeriesGenerator
├── state_art/                         ← State-of-the-art baselines integrados
│   ├── baselines_wrapper.py           ← MultiHorizonBaselineWrapper (STraTS/CoFormer)
│   ├── simple_baselines.py            ← Persistence, Linear, LastValueTimeMLP
│   ├── strats/
│   │   ├── model.py                   ← STraTSNetwork (Särkkä et al.)
│   │   └── embeddings.py              ← ContinuousValueEmbedding
│   └── coformer/
│       ├── model.py                   ← CompatibleTransformer
│       ├── attention.py               ← CoFormerAttentionLayer
│       └── encodings.py               ← MeasurementEmbedding, VariateTimeEncoding
├── scripts/                           ← 25 scripts de experimentación
│   ├── benchmark_final.py             ← Alias de benchmark_synthetic.py
│   ├── benchmark_synthetic.py         ← Motor principal del benchmark (1056 líneas)
│   ├── generate_synthetic_benchmarks.py ← Generador de datos sintéticos
│   ├── tune_synthetic_optuna.py       ← Búsqueda Optuna de hiperparámetros
│   ├── analyze_synthetic_benchmark.py ← Análisis estadístico
│   └── ... (20 scripts más: plotting, ablation, comparison)
├── configs/                           ← Configuraciones YAML
│   ├── benchmark/
│   │   ├── synthetic_optuna_best.yaml ← Recetas ganadoras de Optuna
│   │   └── synthetic_confirmatory_protocol.yaml ← Protocolo de evaluación
│   ├── model/
│   │   └── synthetic_transformer.yaml ← Arquitectura base del Transformer
│   ├── training/
│   │   └── synthetic_benchmark.yaml   ← Hiperparámetros de entrenamiento
│   └── data/                          ← (git-ignored) Configuraciones de datos
├── experiments/                       ← Resultados
│   ├── synthetic_benchmark/           ← Resultados del benchmark
│   ├── optuna_synthetic/              ← Estudios Optuna (db + CSV)
│   └── synthetic_preflight*/          ← Preflight runs
├── latex/                             ← Tesis en LaTeX
│   ├── main.tex                       ← Documento principal
│   ├── documents/                     ← Capítulos (8 + anexos)
│   ├── images/                        ← Figuras
│   ├── references/                    ← Bibliografía
│   └── build_thesis.ps1              ← Script de compilación
├── papers/                            ← Papers referenciados (2 .tex)
├── tests/                             ← 6 suites pytest
├── notebooks/                         ← Jupyter notebooks (4 subcarpetas)
├── requirements.txt                   ← Dependencias
├── pytest.ini                         ← Configuración de tests
└── README.md                          ← Este archivo
```

---

## Modelos Comparados

### Modelos Propios (Custom Family)

| Modelo | Tipo | Descripción |
|--------|------|-------------|
| **Custom** | Transformer | Atención causal opcional y encoding temporal continuo (sinusoidal/Time2Vec), con soporte para bias temporal. **Ganador Optuna**: d_model=96, 2 capas, LR=5.66e-4, dropout=0.05. |
| **Custom-TimeBias** | Variante temporal | Receta optimizada de Custom con `TemporalAttentionBias` activo en la primera capa; aprende una escala de distancia temporal distinta por cabeza. |
| **Custom-Time2Vec** | Ablación | Idéntico a Custom pero con encoding Time2Vec (frecuencias periódicas aprendidas, Kazemi et al. ICLR 2019). |
| **EncDec-AR** | Encoder-Decoder | Autoregresivo: decoder genera tokens uno a uno con cross-attention. **Ganador Optuna**: d_model=96, encoder=3 capas + decoder=1, LR=7.00e-4, time2vec. |

### Ablaciones

| Modelo | Diferencia con Custom |
|--------|-----------------------|
| `Custom-OrdinalTime` | `time_encoding_mode="ordinal"` — ignora timestamps reales, usa posición ordinal |
| `Custom-NoRole` | `use_target_flag_embedding=False` — sin embedding de flag historia/target |

### Variantes experimentales (opt-in)

| Modelo | Cambio aislado |
|--------|----------------|
| `Custom-Gaussian` | Cabeza heteroscedástica (media y desviación) entrenada con NLL gaussiana enmascarada |
| `Custom-LearnableScale` | Escala temporal positiva y aprendible, excluida de weight decay |
| `Custom-RoPE` | RoPE continuo calculado sobre timestamps reales normalizados |
| `Custom-TimeWindow` | Atención causal dispersa sobre vecinos dentro de una ventana temporal |

Estas variantes no forman parte del benchmark congelado y se escriben por defecto en
`experiments/synthetic_architecture_ablations/`:

```powershell
python scripts/benchmark_synthetic.py --models Custom Custom-Gaussian Custom-LearnableScale Custom-RoPE Custom-TimeWindow --temporal-window 8
```

### State-of-the-Art Wrappers

| Modelo | Paper | Arquitectura |
|--------|-------|-------------|
| **STraTS_Adapter** | Särkkä et al. | Tripleta `(feature_id, value, timestamp)` → embedding → Transformer encoder → FusionAttention → Classifier |
| **CoFormer** | Compatible Transformer | MeasurementEmbedding + VariateTimeEncoding → capas CoFormer → observation-wise MHA aggregation |

### Baselines

| Modelo | Entrenable | Descripción |
|--------|-----------|-------------|
| **Persistence** | No | Predice el último valor observado (parámetro dummy para compatibilidad) |
| **PerTargetPersistence** | No | Persistencia por canal objetivo |
| **LastValueTimeMLP** | Sí | MLP sobre última observación + horizonte temporal + sensor |
| **LinearBaselineModel** | Sí | MLP sobre últimos N valores + timestamps relativos |

---

## Flujo de Trabajo

### 1. Entorno y Datos

```powershell
conda activate memoria
pip install -r requirements.txt

# Generar datos sintéticos (7 escenarios × 2 modalidades)
python scripts/generate_synthetic_benchmarks.py --seed 2026 \
    --univariate-observations 1000000 \
    --multivariate-observations 1000000 \
    --n-channels 6
```

El generador produce **7 escenarios** en cada modalidad:

| Escenario | Característica |
|-----------|---------------|
| `regular_control` | Control regular (equiespaciado) |
| `renewal` | Procesos de renovación |
| `bursty` | Eventos en ráfagas (con Numba opcional) |
| `long_gaps` | Huecos temporales largos |
| `informative` | Información variable por canal |
| `nonstationary` | No estacionario |
| `noisy` | Ruidoso |
| `hard_mixed` | Mezcla difícil de todos los factores |

Cada dataset tiene ~1M de observaciones, con splits 70/15/15 (train/val/test). Formato Parquet con compresión snappy.

### 2. Preflight (Validación)

**Antes** del benchmark completo, ejecutar el preflight para validar contratos de datos, entrenamiento y scoring:

```powershell
python scripts/benchmark_final.py --validate-only --exp-dir experiments/synthetic_preflight
```

Recorre los 14 datasets y los 7 modelos, realiza un paso de entrenamiento cuando corresponde y valida el cálculo del score sin consultar el split test. El reporte queda en `model_contract_validation.csv`.

### 3. Benchmark

```powershell
python scripts/benchmark_final.py
```

El resultado reanudable se escribe en `experiments/synthetic_benchmark/`. Por defecto, `Custom` y `EncDec-AR` usan las recetas `Optimized` congeladas en `configs/benchmark/synthetic_optuna_best.yaml`.

```powershell
# Comparar solo familias con perfiles históricos
python scripts/benchmark_final.py --models Custom EncDec-AR --model-sizes Small Medium Large
```

### 4. Búsqueda de Hiperparámetros

```powershell
python scripts/tune_synthetic_optuna.py
```

250 trials por familia (500 total) sobre 6 escenarios representativos. Se reanuda desde `experiments/optuna_synthetic/optuna_studies.db`.

---

## Protocolo de Benchmark

| Parámetro | Valor |
|-----------|-------|
| **Historia** | 512 observaciones fijas para todos los modelos |
| **Horizontes** | `[1, 4, 16, 64]` (4 targets) |
| **Score** | RMSE en espacio estandarizado con estadísticas de entrenamiento |
| **Semillas** | `[42, 84, 126]` (3 semillas por dataset) |
| **Total esperado** | 1296 corridas (48 datasets × 9 modelos × 3 semillas) |

### Corrección Estadística

- **Multiplicidad**: Corrección Holm
- **Test no paramétrico**: Wilcoxon signed-rank
- **IC 95%**: Bootstrap pareado por dataset
- **Superioridad**: delta < 0 + IC95 superior < 0 + p-Holm < 0.05

### Recetas Optuna Congeladas

| Familia | Trial | Val RMSE | d_model | Capas | LR | Dropout | Encoding |
|---------|-------|----------|---------|-------|-----|---------|----------|
| Custom | 117 | 0.1411 | 96 | 2 | 5.66e-4 | 0.05 | sinusoidal |
| EncDec-AR | 249 | 0.1477 | 96 | 3+1 | 7.00e-4 | 0.0 | time2vec |

---

## Generador de Datos Sintéticos

### Arquitectura

```
FastIrregularTimeSeriesGenerator
    ├── generate_univariate_collection()
    └── generate_multivariate_collection()
            └── layout: "asynchronous" (canales con frecuencias heterogéneas)
```

### FastGenerationOptions

| Opción | Default | Descripción |
|--------|---------|-------------|
| `compact_dtypes` | True | dtypes compactos para reducir memoria |
| `include_clean_value` | True | Incluir referencia densa sin ruido |
| `global_sort` | True | Ordenar eventos multivariados por tiempo |
| `compute_metrics` | True | Calcular métricas de irregularidad |
| `categorical_labels` | True | Etiquetas categóricas por canal |
| `use_numba_for_bursty` | True | Numba para sampling de ráfagas |

### Métricas de Irregularidad

Cada dataset reporta: `cv_dt` (coeficiente de variación de deltas), `max_gap_ratio` (ratio del mayor hueco), `burstiness` (medida de ráfagas), por canal y global.

---

## Búsqueda de Hiperparámetros

### Espacio de Búsqueda

| Hiperparámetro | Valores |
|---------------|---------|
| `architecture` | small_2h, small_4h, medium_4h, medium_8h, wide_4h, wide_8h, large_8h |
| `encoder_layers` | [2, 4] |
| `ffn_multiplier` | [2, 4, 6] |
| `dropout` | [0.0, 0.05, 0.1, 0.15, 0.2] |
| `time_encoding_mode` | sinusoidal, time2vec |
| `time_transform` | linear, log1p |
| `horizon_profile` | short_2, standard_4, extended_8 |
| `history_length` | [128, 256, 512] |
| `learning_rate` | [5e-5, 7e-4] log-uniform |
| `weight_decay` | [1e-5, 1e-2] log-uniform |
| `warmup_epochs` | [1, 5] |

### Configuración Optuna

- **6 datasets representativos**: 3 univariados + 3 multivariados
- **250 trials/familia** con pruning (MedianPruner, 25 startup trials)
- **Storage**: SQLite (`experiments/optuna_synthetic/optuna_studies.db`)
- **Objetivo**: minimizar `mean_val_rmse` sobre los 6 escenarios
- **Recetas ganadoras**: escritas en `best_custom.json` y `best_encdec-ar.json`

---

## Pipeline de Datos

### Datasets

#### `TimeSeriesDataset` (Dense)
- **Entrada**: `values [T, D_total]`, `timestamps [T]`
- **Split**: primeras `input_dim` columnas = features, siguientes `output_dim` = targets
- **Config**: `history_length`, `target_offset_choices`, `stride`, `min_history_length`, `num_targets`
- **Multi-target**: Muestra K offsets combinados por ejemplo

#### `EventTimeSeriesDataset` (Sparse/Event)
- **Entrada**: Observaciones como tokens `(sensor_id, timestamp, value)`
- **Validación**: `~torch.isnan(values)` para contar eventos válidos
- **BucketBatchSampler**: Agrupa por longitud aproximada para minimizar padding

### Sequence Builder

```python
# Dense mode
SequenceBuilder(input_dim=D, target_token_value="zeros", num_target_tokens=1)

# Event mode
SequenceBuilder(input_dim=1, use_sensor_ids=True, num_sensors=M, num_target_tokens=M)
```

Concatena historia + targets al final de la secuencia. `target_token_value` puede ser `"zeros"` o `"last"` (copia del último valor).

### AutoregressiveSequenceBuilder

Subclase para entrenamiento autoregresivo con **teacher forcing**: inserta valores reales desplazados 1 posición a la derecha. El primer token a predecir recibe ceros.

### Collate Function

**Left-padding**: Mantiene los tokens target al final global del batch.

```
[padding, padding, history, target_1, target_2, ...]
```

### BucketBatchSampler

Agrupa muestras de longitud similar con ruido proporcional (~10%) para batches más eficientes en datos irregulares.

---

## Training Loop

### `TrainingConfig` (dataclass)

| Campo | Default | Descripción |
|-------|---------|-------------|
| `num_epochs` | 30 | Épocas |
| `device` | "cuda" | Dispositivo |
| `loss_name` | "huber" | Pérdida (MSE, MAE, Huber, DILATE) |
| `grad_clip_norm` | 1.0 | Clipping L2 |
| `early_stopping_patience` | 6 | Paciencia |
| `restore_best_weights` | True | Restaurar mejores pesos |
| `use_amp` | True | Mixed precision |
| `enable_cuda_runtime_optimizations` | True | TF32, cuDNN benchmark, SDPA |
| `use_torch_compile` | False | torch.compile |
| `freeze_encoder_epochs` | 0 | Congelar encoder N épocas |
| `input_noise_std` | 0.0 | Ruido gaussiano en entrada |

### `Trainer`

**Ruta de entrenamiento**:
1. `apply_finetune_schedule()` → congela/descongela encoder
2. `_train_one_epoch()` → batch loop con AMP
3. `_evaluate()` → validación con métricas estructuradas
4. `_maybe_save_best()` → checkpoint + early stopping
5. Scheduler step

**Optimizador**: AdamW con parameter groups diferenciados (weight_decay=0 para parámetros de `time_encoding.time2vec`).

**Scheduler**: Cosine annealing con warmup lineal (`WarmupCosineScheduler`).

### `AutoregressiveTrainer`

Subclase de `Trainer` que reemplaza `_evaluate()` con `_evaluate_generated()`, usando `model.generate()` para predicción recursiva en validación.

---

## Inferencia

### `Predictor`

Envoltor de alto nivel para `TimeSeriesTransformer`:
- `predict_single()` → Un timestamp objetivo
- `predict_multi_targets()` → Múltiples timestamps (loop sobre predict_single)
- `from_checkpoint()` → Carga desde checkpoint torch.save

### `ExperimentPredictor`

Carga desde carpeta de experimento (`best_model.pt`, `model_config.yaml`, `scalers.pt`):
- `predict()` → Acepta valores numéricos o datetimes, devuelve DataFrame opcionalmente
- `predict_from_offsets()` → Construye timestamps a partir de offsets
- `RollingForecaster` → Multi-step forecasting con modo fixed_history

### `build_predictor_from_experiment()`

Helper para construir un Predictor desde una carpeta de experimento generada por `train.py`.

---

## Tests

| Archivo | Cobertura |
|---------|-----------|
| `test_synthetic_benchmark.py` | Recetas Optuna, construcción de modelos, optimizadores, padding |
| `test_experiment_predictor.py` | Predictor desde experimentos, sensores, checkpoints legacy |
| `test_hyperparameter_search.py` | Grid/random search, trial generation |
| `test_structured_metrics.py` | Métricas por horizonte/canal |
| `test_synthetic_analysis.py` | Análisis estadístico |
| `test_experimental_architectures.py` | Ablaciones experimentales de arquitectura |
| `test_time_encoding_ablations.py` | Ablaciones de encoding temporal |

```powershell
pytest
```

---

## Tesis

### Estructura (`latex/main.tex`)

| Capítulo | Contenido |
|----------|-----------|
| 1 | Introducción |
| 2 | Estado del arte |
| 3 | Series de tiempo y problema |
| 4 | Transformers para series |
| 5 | Modelo no equiespaciado (núcleo teórico) |
| 6 | Experimentos |
| 7 | Resultados y discusión |
| 8 | Conclusiones y trabajo futuro |

### Anexos

| Anexo | Contenido |
|-------|-----------|
| A | Fundamentos de redes neuronales |
| B | Optimizadores y Adam |

### Compilación

```powershell
.\latex\build_thesis.ps1
```

Estilo APA, español, Biber backend.

---

## Stack Tecnológico

| Componente | Versión |
|------------|---------|
| PyTorch | nightly (CUDA 13.0) |
| Optuna | 4.x |
| Pandas | 2.3.3 |
| PyArrow | parquet |
| pytest | 9.0.1 |
| LaTeX | Biber + APA style |

---

## Contribuciones Técnicas

1. **Encoding temporal continuo**: Soporte para 4 modos (sinusoidal, ordinal, MLP, Time2Vec) con transformación log1p/linear
2. **TemporalAttentionBias**: Bias tipo ALiBi adaptado para timestamps continuos, con escalas aprendidas por cabeza
3. **SDPA inteligente**: Manejo de máscaras combinadas (causal + padding + bias) sin materialización densa
4. **BucketBatchSampler**: Agrupación por longitud con ruido para datos irregulares
5. **Left-padding con target al final**: Diseño consistente en collate para mantener targets fijos
6. **AutoregressiveSequenceBuilder**: Teacher forcing para entrenamiento de modelos autoregresivos
7. **DILATELoss**: Pérdida shape+time con soft-DTW y distorsión temporal
8. **Parameter groups diferenciados**: Weight decay=0 para parámetros de time2vec
9. **Preflight validation**: Validación de contratos antes del benchmark completo
10. **Reanudabilidad**: CSV incremental + manifest de corrida

---

## Enlaces Rápidos

| Recurso | Ruta |
|---------|------|
| Motor del benchmark | `scripts/benchmark_synthetic.py` |
| Generador de datos | `scripts/generate_synthetic_benchmarks.py` |
| Búsqueda Optuna | `scripts/tune_synthetic_optuna.py` |
| Recetas ganadoras | `configs/benchmark/synthetic_optuna_best.yaml` |
| Protocolo de evaluación | `configs/benchmark/synthetic_confirmatory_protocol.yaml` |
| Arquitectura base | `configs/model/synthetic_transformer.yaml` |
| Resultados del benchmark | `experiments/synthetic_benchmark/` |
| Estudios Optuna | `experiments/optuna_synthetic/` |
| Tesis LaTeX | `latex/main.tex` |
| Papers referenciados | `papers/` |
