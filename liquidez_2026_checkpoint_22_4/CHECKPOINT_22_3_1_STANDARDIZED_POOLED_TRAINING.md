# Checkpoint 22.3.1 — Standardized Pooled Productive Training

## Objetivo

Estandarizar `pooled_balanced` como la única ruta productiva de los cuatro
notebooks globales antes de implementar autoregresión.

El cambio elimina una inconsistencia del contrato anterior: el notebook
exponía las fases `warm-up -> fine-tuning -> consolidation`, aunque en modo
`pooled` únicamente se ejecutaba la primera etapa y las llamadas posteriores no
agregaban epochs.

Este checkpoint es de **limpieza metodológica y contractual**. No cambia:

- los encoders por modalidad;
- el decoder global;
- el residual decoder;
- los auxiliary heads;
- el objetivo de forecasting;
- el dataset de sliding windows.

## Workflow productivo oficial

```text
HPO proxy
    ↓
selección medium-fidelity de finalistas
    ↓
pooled full training
    ↓
pooled continuation opcional
    ↓
backtest
    ↓
forecast
```

El candidato seleccionado por HPO se reinicializa antes del entrenamiento
productivo. Los pesos de proxy y medium-fidelity se conservan como evidencia de
selección, pero no se reutilizan como punto inicial del modelo final.

## Configuración estandarizada

Los notebooks eliminan:

```text
WARM_EPOCHS
WARM_BATCH
FINE_EPOCHS
FINE_BATCH
CONSOLIDATION_EPOCHS
REPLAY_FRACTION
FINETUNE_LR_FACTOR
CONSOLIDATION_LR_FACTOR
TRAINING_ORDER
```

La superficie productiva queda:

```python
TRAINING_STRATEGY = "pooled_balanced"
POOLED_TRAIN_EPOCHS = 60
POOLED_TRAIN_BATCH = 512
POOLED_CONTINUATION_EPOCHS = 0
POOLED_CONTINUATION_LR_FACTOR = 0.20
```

Los 60 epochs recuperan el presupuesto antes repartido entre warm-up y
fine-tuning, pero ahora se aplican sobre una única distribución pooled
balanceada y estable.

## Schedule interno

Se introduce `GlobalTrainingScheduleConfig`.

Para `training_order="pooled_balanced"` produce:

```text
pooled_full_training
phase = productive_training
all curriculum levels
balanced sampler
learning_rate_factor = 1.0
```

Cuando `pooled_continuation_epochs > 0`, agrega:

```text
pooled_continuation
phase = pooled_continuation
same levels
same sampler
same objective
lower learning rate
```

La continuación no cambia la distribución de datos ni introduce curriculum.

## Contrato Pydantic

El contrato del notebook sube a:

```text
schema_version = "22.3.1"
```

La máquina de estados pública queda:

```text
hpo_and_pooled_training -> backtest -> forecast
```

Se reemplazan:

```text
HPOWarmupRequest
FineTuneRequest
```

por:

```text
PooledTrainingRequest
```

El contrato rechaza schedules productivos diferentes de
`pooled_balanced` dentro de los cuatro notebooks globales.

## API pública

La ejecución oficial usa:

```python
workflow.run_hpo_and_train(...)
workflow.run_backtest(...)
workflow.run_forecast(...)
```

`GlobalManager.run_hpo_and_train()` ejecuta en una sola llamada:

1. HPO proxy;
2. selección medium-fidelity;
3. construcción nueva del candidato ganador;
4. pooled full training;
5. pooled continuation, si está habilitada.

El resultado primario se expone mediante:

```python
manager.run_results()
```

con las claves:

```text
training
backtest
forecast
df_forecasts
df_outliers
```

`legacy_results()` permanece únicamente para consumidores históricos.

## Ablations históricas

`curriculum` y `shuffled` no fueron eliminados del motor interno. Permanecen
como ablations reproducibles mediante `GlobalTrainingScheduleConfig`, pero ya
no aparecen como opciones del notebook productivo.

La clase `GlobalCurriculumConfig` conserva por defecto la ruta curricular para
pruebas y reproducción histórica.

## Persistencia

El schema de artefactos sube a:

```text
1.5
```

El manifest usa:

```text
training_schedule_config
```

El loader mantiene compatibilidad de lectura con artefactos `1.4` que todavía
contengan `curriculum_config`.

## Validación

La regresión se ejecutó en grupos aislados para evitar la acumulación de
recursos observada en el runner monolítico:

```text
203 passed
```

Incluye cuatro gates específicos de 22.3.1 para:

- construcción del schedule pooled;
- validación de `PooledTrainingRequest`;
- ejecución conjunta de entrenamiento principal y continuación;
- conservación explícita de las ablations curriculum y shuffled.

También se validó:

- `compileall` estricto;
- compilación de los cuatro notebooks;
- ausencia de outputs y execution counts;
- ausencia de parámetros y llamadas obsoletas en los notebooks;
- ausencia de caracteres de control en markdowns;
- round-trip de persistencia cubierto por la regresión existente.

## Siguiente checkpoint

```text
22.3.2b — Autoregressive Residual Refinement
```
