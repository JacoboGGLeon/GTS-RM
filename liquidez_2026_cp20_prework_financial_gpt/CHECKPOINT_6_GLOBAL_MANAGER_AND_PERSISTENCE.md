# Checkpoint 6 — GlobalManager y persistencia

## Objetivo

Orquestar la ruta global completa ya implementada y persistir un único modelo
por arquitectura, sin crear todavía notebook productivo ni modificar monitores.

## Flujo implementado

```text
dataset_factory(window)
    ↓
GlobalHPOTrainer: un estudio global
    ↓
ventana e hiperparámetros ganadores
    ↓
reconstrucción del dataset ganador
    ↓
GlobalCurriculumTrainer
    ├─ warm-up
    ├─ fine-tuning por nivel + replay
    └─ consolidación
    ↓
un único state_dict global
```

## `GlobalManager`

Expone:

```text
fit_global(...)
backtest_seen(...)
backtest_unseen(...)
evaluate(...)
forecast(...)
save_model(...)
load_model(...)
run_summary()
```

El manager no construye SQL, rutas S3, calendarios ni splits hardcodeados. Recibe
un `dataset_factory` y un manifiesto de split externos.

## Forecast

`forecast()` genera una tabla larga auditable:

```text
cross_key_id
account_currency_id
tipo_serie
cutoff
horizon_step
prediction
actual
prediction_scaled
actual_scaled
center
scale
```

La predicción se devuelve en escala original mediante el `center` y `scale`
calculados sólo con el contexto histórico.

## Persistencia

Cada run contiene exactamente:

```text
manifest.json
model_state.pt
metrics.json
history.json
hpo_summary.json
split_manifest.json
```

`model_state.pt` guarda un solo `state_dict`, no diccionarios de modelos por
serie. `manifest.json` registra:

- arquitectura y dimensiones;
- orden exacto de variables exógenas;
- configuración del modelo, training y curriculum;
- candidato ganador de HPO;
- digest de pesos y SHA-256 del archivo;
- metadata del run.

La carga verifica checksum, digest y compatibilidad estricta del `state_dict`
antes de habilitar inferencia.

## Seen / unseen

- `backtest_seen()` usa ventanas futuras de identidades vistas.
- `backtest_unseen()` usa identidades excluidas del entrenamiento.
- después de cargar un artefacto, el usuario debe proporcionar explícitamente
  el dataset de evaluación; los datos originales no se serializan junto al
  modelo.

## Identidad

`cross_key_id`, cuenta, divisa y tipo de serie sólo participan en:

- construcción y separación de datasets;
- agregación de métricas;
- auditoría y salida de forecast.

Nunca entran al `forward()`.

## Fuera de alcance

No se implementó:

- notebook global parametrizable;
- conexión directa a S3;
- monitor comparativo local/global;
- selección final por budget;
- cambios en `code_01`, `code_02_*` o sus monitores.

## Gate

```bash
python -m unittest discover -s tests -p "test_checkpoint_*.py" -v
python -m compileall -q .
```

El siguiente checkpoint debe implementar únicamente el notebook global
parametrizable sobre estos contratos.
