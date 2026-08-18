# Memoria — Benchmark de Transformers para Series de Tiempo Irregulares

Un sistema de investigación para el benchmark reproducible de modelos Transformer sobre series de tiempo con observaciones asíncronas (no equiespaciadas). El proyecto genera datos sintéticos controlados, compara arquitecturas y ablaciones temporales, optimiza hiperparámetros con Optuna y produce resultados estratificados para una tesis doctoral.

## Tabla de Contenidos

- [Propósito](#propósito)
- [Contrato de Tiempo Físico](#contrato-de-tiempo-físico)
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

## Contrato de Tiempo Físico

El benchmark original usa una historia de 512 **eventos** y offsets `[1, 4, 16,
64]` medidos también en eventos. Ese contrato sigue disponible para reproducir
experimentos anteriores, pero no permite atribuir una mejora al uso correcto del
tiempo físico:

- el mismo offset ordinal representa duraciones distintas según la densidad del
  dataset y del canal;
- una historia con igual cantidad de eventos cubre intervalos físicos distintos;
- si las consultas permanecen ordenadas, el slot del target puede revelar el
  horizonte aunque se corrompan los timestamps;
- convertir timestamps absolutos grandes a `float32` antes de recentrarlos puede
  hacer indistinguibles instantes cercanos.

Por eso, una ablación de encoding temporal que cambie poco el resultado del
benchmark legacy no demuestra que el tiempo sea irrelevante. Puede indicar que la
tarea se resuelve mediante orden, densidad, persistencia o fuga del slot de query.
El protocolo congelado en
`configs/benchmark/synthetic_confirmatory_protocol.yaml` se conserva como
**legacy/event-offset**; no se deben comparar sus scores directamente con el
nuevo protocolo.

El contrato físico definido en
`configs/benchmark/synthetic_physical_protocol.yaml` corrige esos problemas:

1. conserva timestamps absolutos en `float64` durante carga y selección;
2. selecciona historia por duración física y limita costo con subsampling que
   preserva la cobertura del intervalo;
3. toma los orígenes de forecast desde una grilla física de `truth.parquet`, no
   desde la llegada de un evento; así incluye consultas dentro de bloques sin
   observaciones y mide la edad de la última observación;
4. durante training muestrea cuatro horizontes físicos `log_uniform` en
   `[0.25, 8.0]`; evalúa en `[0.25, 1.0, 3.0, 8.0]` y aleatoriza su orden;
5. obtiene targets desde la misma trayectoria latente limpia en `truth.parquet`,
   almacenada por separado e independiente del patrón de observación y del
   ruido, mediante interpolación lineal;
6. recentra los tiempos dentro de cada ventana y sólo entonces los entrega al
   modelo en `float32`.

La comparación principal incluye `Custom-QueryCross`, `Continuous-Basis`, sus
variantes CTSSM, las ablaciones `NoTime`/`QueryOnly` y `Persistence`. Las
cabezas `Gaussian` agregan evaluación probabilística. Los resultados se reportan también
por bins de horizonte, mayor gap histórico, densidad y edad desde la última
observación; un promedio global no es suficiente para afirmar robustez temporal.
En multivariado también se publican `channel_density_bin`,
`channel_max_gap_bin` y `channel_last_age_bin`: sus cuantiles se calculan dentro
de cada canal para no confundir identidad del sensor con irregularidad.
Las variantes `real_no_count_feature` y `all_equal_no_count_feature` anulan la
feature explícita de conteo, sin afirmar que eliminan los gaps/edades derivados.

Cada origen candidato se audita antes de construir el dataset. Los descartes se
asignan de forma mutuamente excluyente a la primera causa aplicable: cobertura
histórica insuficiente, origen posterior a la última observación, historia sin
observaciones reales o consulta fuera del rango de `truth`. Una historia vacía
se descarta; nunca se fabrica una observación cero. `data_metadata.json` registra
por split candidatos, aceptados, descartados por causa y ejemplos retenidos tras
el cap de ejecución.

### Cierre de tesis: flujo físico congelado

El punto de entrada recomendado para concluir la tesis es
`scripts/run_thesis_physical_benchmark.py`, gobernado exclusivamente por
`configs/benchmark/thesis_physical_final.yaml`. Todos los comandos deben
ejecutarse desde la raíz del repositorio con el entorno `memoria`:

```powershell
# 1. Inspecciona selección, archivos y comandos sin entrenar
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py dry-run `
    --config configs/benchmark/thesis_physical_final.yaml

# 2. Valida en pequeño los contratos; sus resultados no sustentan conclusiones
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py smoke `
    --config configs/benchmark/thesis_physical_final.yaml

# 3. Flujo final reanudable: auditar, ejecutar y consolidar
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py preflight `
    --config configs/benchmark/thesis_physical_final.yaml
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py run `
    --config configs/benchmark/thesis_physical_final.yaml
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py report `
    --config configs/benchmark/thesis_physical_final.yaml

# Alternativa equivalente a preflight + run + report
conda.exe run -n memoria python scripts/run_thesis_physical_benchmark.py all `
    --config configs/benchmark/thesis_physical_final.yaml
```

El protocolo selecciona exactamente **16 unidades main** (8 presets por
univariado/multivariado, generadas con seed 2026) y **2 unidades stress**
(`long_gaps` por modalidad, seed de generación 3031). Cada unidad se entrena con
las tres semillas `[42, 84, 126]`; estas semillas de entrenamiento no deben
confundirse con las del generador. Los nueve modelos definidos en el YAML
producen 432 corridas main y 54 stress, además de los controles de
identificabilidad exigidos por el orquestador.

La salida queda en `experiments/thesis_physical_benchmark/`: `main/` y
`stress_gseed3031/` contienen las corridas separadas; `reports/` contiene las
agregaciones por dataset, preset emparejado, macro y estratos; y
`report_manifest.json` registra completitud y hashes. `report` también escribe
en `experiments/thesis_physical_benchmark/latex/` las tablas
`protocol_summary.tex`, `model_results.tex`, `temporal_ablations.tex` y
`gaussian_calibration.tex`. Sólo se generan cuando están completas todas las
corridas esperadas.

Este cierre es una **evaluación retrospectiva de datos sintéticos generados
previamente**. No equivale a validación externa ni a desempeño prospectivo en
datos reales. El orquestador tampoco ejecuta Optuna ni carga automáticamente
`experiments/optuna_physical/best_physical.yaml`: usa los hiperparámetros fijos
del YAML. Incorporar otra receta requiere congelarla explícitamente antes del
preflight y constituye un protocolo nuevo.

### Runners de diagnóstico

Los siguientes comandos de bajo nivel también leen `data/univariate` y
`data/multivariate`, pero no reemplazan el flujo congelado de cierre:

```powershell
# Auditoría rápida de identificabilidad, precisión y controles no neuronales
conda.exe run -n memoria python scripts/temporal_identifiability_benchmark.py --kinds univariate multivariate

# Smoke test neuronal acotado sobre una colección de cada modalidad
conda.exe run -n memoria python scripts/benchmark_physical_models.py `
    --kinds univariate multivariate `
    --limit-datasets-per-kind 1 `
    --models QueryCross QueryCross-Gaussian BasisDecoder BasisDecoder-Gaussian BasisDecoder-CTSSM NoTime QueryOnly CTSSM Persistence `
    --epochs 1 `
    --max-train-samples 32 `
    --max-val-samples 16 `
    --max-test-samples 16 `
    --max-observation-rows-per-split 20000

# Runner físico directo (desarrollo; no usar como cierre final de tesis)
conda.exe run -n memoria python scripts/benchmark_physical_models.py `
    --config configs/benchmark/physical_models.yaml `
    --kinds univariate multivariate
```

El smoke test limita filas sólo para verificar contratos y no sirve para extraer
conclusiones. El runner físico guarda el resumen en
`experiments/physical_models/benchmark_physical_models.csv` y artefactos por
dataset/semilla/modelo (configuración, métricas, predicciones y checkpoint).
Cada corrida completa publica atómicamente `result.json` con un fingerprint del
protocolo, hashes del código ejecutado y versiones numéricas del entorno: una
ejecución idéntica se reutiliza y una incompatible exige `--force-rerun` o un
directorio de salida diferente. Al forzar una corrida, el sentinel previo se
archiva antes de entrenar para que un fallo intermedio no parezca completo.

El benchmark de identificabilidad compara `Persistence`, un control `Ordinal`
sin tiempo físico y un regresor `ExplicitHorizon`. Además materializa un
manifiesto pareado de las corruptelas `real`, `all_equal`, `permuted_gaps`,
`regular_grid`, `ordinal`, `query_only` y `history_only`; su API
`predict_under_timestamp_ablations` permite aplicarlas a cualquier modelo.
Escribe predicciones de los controles, métricas agregadas, métricas por dataset,
estratos temporales y una auditoría de precisión en
`experiments/temporal_identifiability/`.

---

## Arquitectura del Proyecto

```
Memoria/
├── src/
│   ├── ts_transformer/                # Núcleo del proyecto
│   │   ├── models/                    # Modelos de red neuronal
│   │   │   ├── time_series_transformer.py    ← Modelo "Custom" principal
│   │   │   ├── time_series_encoder_decoder.py ← Modelo "EncDec-AR"
│   │   │   ├── query_cross_attention.py ← Queries físicas + cross-attention temporal
│   │   │   ├── continuous_basis_decoder.py ← Función continua de tendencia/RBF/Fourier
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
│   └── data/                          ← Generadores de series sintéticas irregulares
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
│   ├── benchmark_physical_models.py   ← Benchmark neuronal con horizontes físicos
│   ├── run_thesis_physical_benchmark.py ← Orquestador congelado para cierre de tesis
│   ├── temporal_identifiability_benchmark.py ← Controles y corruptelas temporales
│   ├── tune_physical_optuna.py        ← Optuna sobre la tarea física (sin usar test)
│   ├── generate_synthetic_benchmarks.py ← Generador de datos sintéticos
│   ├── tune_synthetic_optuna.py       ← Búsqueda Optuna de hiperparámetros
│   ├── analyze_synthetic_benchmark.py ← Análisis estadístico
│   └── ... (20 scripts más: plotting, ablation, comparison)
├── configs/                           ← Configuraciones YAML
│   ├── benchmark/
│   │   ├── synthetic_optuna_best.yaml ← Recetas ganadoras de Optuna
│   │   ├── synthetic_confirmatory_protocol.yaml ← Protocolo legacy/event-offset
│   │   ├── synthetic_physical_protocol.yaml ← Protocolo con tiempo físico
│   │   ├── physical_models.yaml       ← Parámetros ejecutables del runner físico
│   │   ├── temporal_identifiability.yaml ← Auditoría de identificabilidad
│   │   └── thesis_physical_final.yaml ← Protocolo final congelado y cohortes exactas
│   ├── model/
│   │   └── synthetic_transformer.yaml ← Arquitectura base del Transformer
│   ├── training/
│   │   └── synthetic_benchmark.yaml   ← Hiperparámetros de entrenamiento
│   └── data/                          ← Configuraciones versionadas de generación
├── experiments/                       ← Resultados
│   ├── synthetic_benchmark/           ← Resultados del benchmark
│   ├── physical_models/               ← Benchmark con horizontes físicos
│   ├── thesis_physical_benchmark/      ← Corridas y tablas del cierre final
│   ├── temporal_identifiability/      ← Controles y auditorías temporales
│   ├── optuna_synthetic_fixed_task/   ← Estudios Optuna comparables (db + CSV)
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
| `Custom-TimeWindow` | Atención dispersa por ventana temporal con mínimo de vecinos previos (causal) o bilaterales (no causal) |
| `Custom-QueryCross` | Codifica sólo la historia y predice con queries independientes en tiempo físico. Usa cross-attention con bias de lag relativo y modulación por gap, edad por sensor y densidad. |
| `Custom-QueryCross-NoTime` | Control negativo: desactiva horizonte, encoding histórico, bias de lag, FiLM temporal y CTSSM. |
| `Custom-QueryCross-QueryOnly` | Conserva sólo el horizonte físico explícito de la query; no modela los gaps internos de la historia. |
| `Custom-QueryCross-CTSSM` | Añade una transición de estado continuo diagonal y estable entre eventos, sin solver ODE; la recurrencia afín se evalúa con un scan paralelo asociativo. |
| `Custom-QueryCross-Gaussian` | QueryCross con cabeza heteroscedástica; reporta NLL, CRPS, escala predictiva y coberturas 90/95 %. |
| `Continuous-Basis` | Codifica la historia una vez y evalúa una función continua con bases de tendencia, RBF y Fourier en cada horizonte físico. |
| `Continuous-Basis-CTSSM` | Combina el decoder de bases con transición continua estable entre eventos. |
| `Continuous-Basis-Gaussian` | Decoder continuo con media y escala predictiva por consulta. |

Estas variantes no forman parte del benchmark congelado y se escriben por defecto en
`experiments/synthetic_architecture_ablations/`:

```powershell
conda.exe run -n memoria python scripts/benchmark_synthetic.py --models Custom Custom-Gaussian Custom-LearnableScale Custom-RoPE Custom-TimeWindow Custom-QueryCross Custom-QueryCross-NoTime Custom-QueryCross-QueryOnly Custom-QueryCross-CTSSM Custom-QueryCross-Gaussian --temporal-window 8
```

`QueryCross` no concatena targets ficticios al encoder. Cada target es una query
independiente, identificada por su tiempo y sensor, que atiende sólo a la
historia compatible. El bias relativo combina decaimiento monótono, bases
Fourier y RBF multiescala; el residual de último valor entrega un punto de
partida fuerte para forecasting. Esta separación elimina el índice ordinal del
target como canal temporal oculto.

El runner físico usa alias cortos para QueryCross y añade `BasisDecoder`,
`BasisDecoder-Gaussian` y `BasisDecoder-CTSSM` para el decoder explícito de
función continua.

### Decisiones arquitectónicas deliberadas

Los siguientes comportamientos se conservan intencionalmente y tienen pruebas
de regresión; no constituyen bugs pendientes:

- El contenido de los tokens de left-padding puede tener embeddings arbitrarios:
  `key_padding_mask` impide que actúen como claves y no altera las salidas válidas.
- `TemporalAttentionBias` recibe tiempo relativo lineal, aunque el encoding use
  `log1p`, porque modela intervalos estacionarios entre pares de timestamps.
- La ruta SDPA causal combina la máscara triangular con el bias temporal.
- `FeatureEmbedding` aplica `LayerNorm` después de proyectar a `d_model`; no se
  normaliza `d_in` antes de la proyección, ya que en modo evento `d_in=1` y se
  eliminaría toda la señal de valor.

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

El generador produce **8 escenarios** en cada modalidad:

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

Recorre los datasets descubiertos y los modelos solicitados, realiza un paso de
entrenamiento cuando corresponde y valida el cálculo del score sin consultar el
split test. El reporte queda en `model_contract_validation.csv`.

### 3. Benchmark

```powershell
# Legacy: offsets medidos en cantidad de eventos
python scripts/benchmark_final.py
```

El resultado reanudable se escribe en `experiments/synthetic_benchmark/`. Por defecto, `Custom` y `EncDec-AR` usan las recetas `Optimized` congeladas en `configs/benchmark/synthetic_optuna_best.yaml`.

Este comando reproduce la tarea histórica. Para evaluar si el modelo usa el
tiempo no equiespaciado debe ejecutarse el protocolo físico descrito arriba.

```powershell
# Comparar solo familias con perfiles históricos
python scripts/benchmark_final.py --models Custom EncDec-AR --model-sizes Small Medium Large
```

### 4. Búsqueda de Hiperparámetros

```powershell
conda.exe run -n memoria python scripts/tune_synthetic_optuna.py `
    --families Custom Custom-QueryCross EncDec-AR `
    --horizon-profile standard_4 `
    --history-length 512
```

Por defecto ejecuta 250 trials por familia sobre 6 escenarios representativos y
se reanuda desde
`experiments/optuna_synthetic_fixed_task/optuna_studies.db`. El perfil de
horizontes y la longitud de historia son parte fija del nombre/contrato del
estudio: nunca se samplean como hiperparámetros. Cambiar cualquiera de ellos
crea otra tarea y sus scores no son directamente comparables.

Ese comando conserva deliberadamente la tarea histórica por offsets de evento.
Para optimizar la arquitectura nueva sobre consultas físicas continuas se usa
un estudio separado:

```powershell
conda.exe run -n memoria python scripts/tune_physical_optuna.py `
    --kinds univariate multivariate `
    --trials 50 `
    --epochs 10
```

`tune_physical_optuna.py` entrena con horizontes continuos sobre
`observations.parquet`, obtiene los targets desde `truth.parquet` y selecciona
exclusivamente por `val_rmse`. El objetivo no crea un loader de test. Cada study
incluye un fingerprint SHA-256 del contenido de cada dataset, tarea, seed,
límites de muestreo, código, entorno y versión del espacio de búsqueda; una
reanudación incompatible se rechaza en vez de mezclar trials. Por defecto usa
todos los datasets que coinciden con los filtros; `--limit-datasets-per-kind`
debe indicarse explícitamente para una prueba acotada. Para un smoke también
puede usarse `--max-observation-rows-per-split`, pero cualquiera de esos caps
altera la tarea y no produce un resultado científico final.

---

## Protocolo de Benchmark

### Legacy: offsets por evento

Esta tabla describe exclusivamente
`configs/benchmark/synthetic_confirmatory_protocol.yaml`, congelado para
reproducibilidad histórica:

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

Estas recetas provienen de estudios históricos cuyos perfiles de horizonte no
eran idénticos entre familias. Se conservan para reproducir corridas anteriores,
pero no constituyen por sí solas una comparación justa. Los estudios nuevos
fijan `--horizon-profile` y `--history-length` para todas las familias.

### Protocolo físico

| Parámetro | Valor |
|-----------|-------|
| **Historia** | Duración 8.0; hasta 256 eventos univariados o 512 multivariados representativos |
| **Horizontes training** | 4 queries `log_uniform` en `[0.25, 8.0]` por muestra |
| **Horizontes evaluación** | `[0.25, 1.0, 3.0, 8.0]` en unidades del timestamp |
| **Orden de queries** | Permutado reproduciblemente; no codifica el horizonte |
| **Target** | `truth.parquet`, interpolación lineal por canal |
| **Precisión** | Absoluto `float64`; relativo/recentrado antes de `float32` |
| **Comparación causal** | QueryCross/BasisDecoder, CTSSM, NoTime, QueryOnly y Persistence |
| **Extensión probabilística** | QueryCross-Gaussian y BasisDecoder-Gaussian |
| **Métricas punto** | RMSE, MAE, por dataset/canal/horizonte/gap/densidad |
| **Métricas Gaussian** | NLL, CRPS, sigma y coberturas 90/95 % en el runner físico |

El manifiesto completo está en
`configs/benchmark/synthetic_physical_protocol.yaml`. Primero debe verificarse
identificabilidad con `configs/benchmark/temporal_identifiability.yaml`; entrenar
una red sobre una tarea donde el orden revela el horizonte no corrige el diseño
experimental.

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
| `horizon_profile` | **Fijo por estudio**: short_2, standard_4 o extended_8 |
| `history_length` | **Fijo por estudio**: valor de `--history-length` |
| `learning_rate` | [5e-5, 7e-4] log-uniform |
| `weight_decay` | [1e-5, 1e-2] log-uniform |
| `warmup_epochs` | [1, 5] |

### Configuración Optuna

- **6 datasets representativos**: 3 univariados + 3 multivariados
- **Familias**: Custom, Custom-QueryCross y EncDec-AR
- **250 trials/familia** con pruning (MedianPruner, 25 startup trials)
- **Storage**: SQLite (`experiments/optuna_synthetic_fixed_task/optuna_studies.db`)
- **Objetivo**: minimizar `mean_val_rmse` sobre los 6 escenarios
- **Contrato fijo**: mismo perfil de horizontes e historia para cada trial y familia
- **Alcance**: conserva la tarea histórica por offsets de evento; el tuning del
  protocolo físico debe hacerse con `tune_physical_optuna.py`, no se infiere de
  este estudio legacy
- **Recetas ganadoras**: escritas en `best_custom.json` y `best_encdec-ar.json`

### Optuna físico para QueryCross y BasisDecoder

- **Objetivo**: promedio de `val_rmse` en los datasets seleccionados; test no
  participa en selección.
- **Tarea fija**: duración histórica, consultas continuas, fuente de verdad,
  datasets y caps forman parte del fingerprint del study.
- **Cobertura por defecto**: todos los datasets que coinciden con los filtros;
  `--limit-datasets-per-kind` es únicamente un cap explícito.
- **Arquitectura**: `d_model/heads`, capas encoder/cross, ancho FFN, dropout,
  bases de lag Fourier/RBF, encoding histórico y CTSSM.
- **Optimización**: learning rate y weight decay de AdamW con cosine schedule.
- **Presupuesto reanudable**: `--trials` cuenta sólo trials `COMPLETE`; registros
  `RUNNING`, `PRUNED` o `FAIL` no pueden agotar silenciosamente el objetivo. El
  storage usa heartbeat y un límite finito de reintentos.
- **Storage**: `experiments/optuna_physical/physical_optuna.db` y receta
  `best_physical.yaml`.

---

## Pipeline de Datos

### Datasets

#### `TimeSeriesDataset` (Dense)
- **Entrada**: `values [T, D_total]`, `timestamps [T]`
- **Split**: primeras `input_dim` columnas = features, siguientes `output_dim` = targets
- **Config**: `history_length`, `target_offset_choices`, `stride`, `min_history_length`, `num_targets`
- **Multi-target**: Muestra K offsets combinados por ejemplo
- **Modo físico**: `history_duration`, `max_history_events`,
  `target_horizon_choices` o rango min/max, `target_match_mode` y
  `randomize_query_order`
- **Truth independiente**: acepta `target_timestamps` y `targets` distintos de
  las observaciones de entrada

#### `EventTimeSeriesDataset` (Sparse/Event)
- **Entrada**: Observaciones como tokens `(sensor_id, timestamp, value)`
- **Validación**: `~torch.isnan(values)` para contar eventos válidos
- **BucketBatchSampler**: Agrupa por longitud aproximada para minimizar padding
- **Historia física acotada**: subsampling `uniform_time`, `uniform_index` o
  `random`, preservando el último evento y metadatos de densidad

### Sequence Builder

```python
# Dense mode
SequenceBuilder(input_dim=D, target_token_value="zeros", num_target_tokens=1)

# Event mode
SequenceBuilder(input_dim=1, use_sensor_ids=True, num_sensors=M, num_target_tokens=M)
```

Concatena historia + targets al final de la secuencia. `target_token_value` puede ser `"zeros"` o `"last"` (copia del último valor).

Los timestamps permanecen en `float64` hasta seleccionar la ventana. El builder
los recentra respecto del tiempo histórico de referencia antes de convertirlos
a la precisión usada por la red. `TimeSeriesQueryCrossAttention` consume la
misma salida, pero separa historia y queries antes de codificar.

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
| `test_structured_metrics.py` | Métricas por horizonte/canal y calibración gaussiana |
| `test_synthetic_analysis.py` | Análisis estadístico |
| `test_experimental_architectures.py` | Ablaciones experimentales de arquitectura |
| `test_architecture_guardrails.py` | Decisiones deliberadas de masking, bias y normalización |
| `test_time_encoding_ablations.py` | Ablaciones de encoding temporal |
| `test_query_cross_attention.py` | Queries independientes, lags relativos, ablaciones y CTSSM |
| `test_physical_time_dataset.py` | Horizontes/duración físicos, truth independiente, subsampling y precisión |
| `test_temporal_identifiability_protocol.py` | Aleatorización de slots, controles, corruptelas y estratos |
| `test_fixed_task_tuning.py` | Optuna mantiene fija la tarea entre trials y familias |
| `test_physical_optuna.py` | Fingerprint, reanudación segura y aislamiento de test en Optuna físico |

```powershell
conda.exe run -n memoria python -m pytest -q
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
11. **Queries físicas independientes**: Cross-attention query--historia sin slot ordinal como horizonte oculto
12. **Historia por duración**: Cobertura física con cantidad de eventos acotada y densidad observable
13. **Precisión temporal explícita**: `float64` absoluto y `float32` sólo después del recentrado local
14. **Diagnóstico de identificabilidad**: Controles ordinales, corruptelas de timestamps y métricas por dificultad temporal

---

## Enlaces Rápidos

| Recurso | Ruta |
|---------|------|
| Motor del benchmark | `scripts/benchmark_synthetic.py` |
| Benchmark con tiempo físico | `scripts/benchmark_physical_models.py` |
| Orquestador final de tesis | `scripts/run_thesis_physical_benchmark.py` |
| Auditoría de identificabilidad | `scripts/temporal_identifiability_benchmark.py` |
| Generador de datos | `scripts/generate_synthetic_benchmarks.py` |
| Búsqueda Optuna legacy | `scripts/tune_synthetic_optuna.py` |
| Búsqueda Optuna física | `scripts/tune_physical_optuna.py` |
| Recetas ganadoras | `configs/benchmark/synthetic_optuna_best.yaml` |
| Protocolo legacy/event-offset | `configs/benchmark/synthetic_confirmatory_protocol.yaml` |
| Protocolo de tiempo físico | `configs/benchmark/synthetic_physical_protocol.yaml` |
| Configuración del runner físico | `configs/benchmark/physical_models.yaml` |
| Protocolo de identificabilidad | `configs/benchmark/temporal_identifiability.yaml` |
| Protocolo final congelado | `configs/benchmark/thesis_physical_final.yaml` |
| Arquitectura base | `configs/model/synthetic_transformer.yaml` |
| Resultados del benchmark | `experiments/synthetic_benchmark/` |
| Estudios Optuna de tarea fija | `experiments/optuna_synthetic_fixed_task/` |
| Estudios Optuna físicos | `experiments/optuna_physical/` |
| Salida física final | `experiments/thesis_physical_benchmark/` |
| Tesis LaTeX | `latex/main.tex` |
| Papers referenciados | `papers/` |
