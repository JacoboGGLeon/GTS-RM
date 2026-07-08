# Checkpoint 7 — Notebook global parametrizable

## Objetivo

Entregar una sola entrada ejecutable para las cuatro arquitecturas globales,
sin duplicar notebooks ni reimplementar la lógica ya validada en
`GlobalManager`.

## Notebook

Se añadió:

```text
code_03_GLOBAL_DRL.ipynb
```

El parámetro único de arquitectura acepta:

```text
mlp
mlp_vae
rnn
rnn_bi
```

El notebook está preparado para Papermill mediante una única celda con tag
`parameters` y también conserva anotaciones `@param` compatibles con formularios
de Colab/SageMaker.

## Flujo ejecutado

```text
global_series_long + calendario financiero
    ↓
split fijo por cross_key_id
    ├─ train_series
    ├─ validation_seen_series
    ├─ validation_unseen_series
    └─ test_unseen_series
    ↓
HPO global
    ↓
modelo nuevo con hiperparámetros ganadores
    ↓
warm-up global
    ↓
fine-tuning curricular con replay
    ↓
consolidación
    ↓
backtest seen + validación unseen + test unseen
    ↓
persistencia reproducible
```

## Soporte reutilizable

Se añadió `global_notebook.py` para mantener el notebook delgado y testeable.
Incluye:

- lectura CSV/Parquet desde disco local o S3;
- detección opcional del último `global_series_long.parquet`;
- validación e inferencia de variables exógenas numéricas/bool;
- split de identidades calculado una sola vez antes del HPO;
- holdout temporal por serie para `validation_seen`;
- reconstrucción determinista de datasets para cada `window_size`;
- dataset reservado `test_unseen` fuera del HPO;
- sincronización opcional de un run local hacia S3.

## Control de leakage temporal

Para cada identidad vista:

```text
train targets            : hasta t_split - 1
validation_seen targets  : desde t_split
```

La ventana de validación puede utilizar la historia inmediatamente anterior al
corte, pero ningún target posterior al corte entra al entrenamiento. El corte
permanece idéntico para todos los trials; sólo cambia el tamaño de la ventana
histórica sugerida por HPO.

## Identidad contable

`cross_key_id`, `account_currency_id` y `tipo_serie` se usan únicamente para:

- partición seen/unseen;
- balanceo por serie;
- agregación macro de métricas;
- auditoría y reconstrucción de resultados.

El `forward()` continúa recibiendo exclusivamente:

```text
y_context
x_history
x_future
context_mask
```

## Artefactos

Cada ejecución genera:

```text
<run>/model/
    manifest.json
    model_state.pt
    metrics.json
    history.json
    hpo_summary.json
    split_manifest.json

<run>/reports/
    notebook_config.json
    dataset_summary.json
    evaluation_metrics.json
    evaluation_metrics.parquet
    forecast_validation_seen.parquet       # opcional
    forecast_validation_unseen.parquet     # opcional
    forecast_test_unseen.parquet           # opcional
```

El directorio `model/` conserva exactamente el contrato de persistencia del
Checkpoint 6.

## Defaults eficientes

El dataset de `code_01` utiliza hasta 20 niveles curriculares. Para evitar que un
run por defecto multiplique excesivamente el tiempo de entrenamiento, el
notebook usa:

```text
HPO_EPOCHS = 10
WARMUP_EPOCHS = 15
FINETUNE_EPOCHS_PER_LEVEL = 3
CONSOLIDATION_EPOCHS = 5
N_TRIALS = 15
```

Todos siguen siendo parámetros explícitos del notebook.

## Validación

- 7 pruebas nuevas pasan.
- El notebook fue ejecutado completo con datos sintéticos y parámetros mínimos.
- El smoke recorrió carga, split, HPO, curriculum, seen/unseen, persistencia y
  escritura de reportes.
- `nbformat.validate`: OK.
- Notebook sin outputs ni `execution_count` persistidos.
- Los 64 tests y módulos heredados permanecen byte a byte idénticos al bundle
  de Checkpoint 6; además se reejecutaron los gates rápidos de Checkpoints 0–4.
- `compileall`: OK.

## Fuera de alcance

No se modificó:

- `code_01.ipynb`;
- los cuatro `code_02_*` locales;
- `Engineer`, `Scientist`, `Manager` locales;
- arquitecturas, HPO, curriculum o `GlobalManager`;
- `monitor_codigo_01.ipynb`;
- `monitor_codigo_02.ipynb`.

El siguiente checkpoint debe implementar exclusivamente el monitor seen/unseen,
la comparación local/global y los gates finales.
