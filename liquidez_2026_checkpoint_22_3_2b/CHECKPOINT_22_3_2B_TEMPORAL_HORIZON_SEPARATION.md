# Checkpoint 22.3.2b — Temporal Horizon Separation

## Objetivo

Separar cuatro cantidades temporales que antes podían confundirse dentro de los cuatro notebooks globales:

| Variable visible | Responsabilidad |
|---|---|
| `MAX_WINDOW_SIZE` | Máximo contexto histórico que el HPO puede seleccionar para el encoder. |
| `TRAINING_STRIDE` | Desplazamiento entre orígenes consecutivos al construir sliding windows. |
| `ROLLOUT_CHUNK_SIZE` | Longitud del target de entrenamiento y número de pasos emitidos por un `forward`. |
| `FORECAST_HORIZON` | Número total máximo de pasos futuros solicitados y exportados. |

El cambio no introduce todavía un decoder autoregresivo entrenable. Mantiene el decoder directo multi-step dentro de cada bloque y utiliza el recorrido por bloques ya existente durante la inferencia.

## Semántica del entrenamiento

Para:

```python
FORECAST_HORIZON = 25
ROLLOUT_CHUNK_SIZE = 3
TRAINING_STRIDE = 1
```

cada muestra aprende un target de tres puntos:

```text
[y(t-W+1) ... y(t)]   -> [y(t+1), y(t+2), y(t+3)]
[y(t-W+2) ... y(t+1)] -> [y(t+2), y(t+3), y(t+4)]
```

`TRAINING_STRIDE=1` desplaza el origen una fecha. No modifica la longitud del target ni el número total de puntos pedidos al forecast.

Las dimensiones del contrato tensorial son:

```text
y_context : [batch, window_size, 1]
x_history : [batch, window_size, exogenous_dim]
x_future  : [batch, rollout_chunk_size, exogenous_dim]
y_target  : [batch, rollout_chunk_size, 1]
x_static  : [batch, static_dim]
```

## Semántica de la inferencia

El modelo emite `ROLLOUT_CHUNK_SIZE=3` puntos por bloque. Para cubrir 25 pasos:

```text
bloque 1 -> t+1  ... t+3
bloque 2 -> t+4  ... t+6
...
bloque 8 -> t+22 ... t+24
bloque 9 -> t+25 (se recorta el excedente)
```

Por tanto:

```text
rollout_blocks_max = ceil(FORECAST_HORIZON / ROLLOUT_CHUNK_SIZE)
                   = ceil(25 / 3)
                   = 9
```

Dentro del bloque, los puntos se producen conjuntamente. Entre bloques, el forecast agrega la media MC predicha al contexto y vuelve a ejecutar el modelo. Esta recursión por bloques no equivale todavía a teacher forcing, scheduled sampling o refinamiento autoregresivo paso a paso.

## Horizonte máximo

`FORECAST_HORIZON` funciona también como gate:

- en modo `n_steps`, la solicitud no puede superarlo;
- en modo rango de fechas, el número de timestamps resueltos sobre `TemporalAxis` tampoco puede superarlo.

Esto evita que un rango explícito ejecute accidentalmente un rollout mucho mayor al autorizado por el run.

## Contratos Pydantic

Se agregó:

```python
TemporalForecastConfig(
    forecast_horizon=25,
    rollout_chunk_size=3,
    training_stride=1,
)
```

Valida que:

```text
1 <= rollout_chunk_size <= forecast_horizon
training_stride >= 1
```

`GlobalActiveConfiguration` y `GlobalNotebookRunContract` verifican coherencia con:

```text
GlobalNotebookConfig.horizon          == rollout_chunk_size
GlobalNotebookConfig.forecast_horizon == forecast_horizon
GlobalNotebookConfig.stride           == training_stride
```

El nombre interno `horizon` se conserva en datasets, modelos y artefactos antiguos porque representa la dimensión fija de salida del modelo. La superficie del notebook utiliza `ROLLOUT_CHUNK_SIZE` para expresar su significado real y evitar confundirlo con el horizonte total.

## Archivos modificados

- `global_surface_config.py`
- `global_notebook.py`
- `global_pipeline.py`
- `global_manager.py`
- `global_monitoring.py`
- `code_03_GLOBAL_MLP_E_D.ipynb`
- `code_03_GLOBAL_MLP_VaE_D.ipynb`
- `code_03_GLOBAL_RNN_E_D.ipynb`
- `code_03_GLOBAL_RNNBi_E_D.ipynb`
- pruebas históricas de contrato actualizadas
- `tests/test_checkpoint_22_3_2b_temporal_horizon_separation.py`

## Alcance preservado

No se modificaron:

- los encoders por modalidad;
- el decoder global;
- el residual decoder;
- los heads auxiliares;
- las pérdidas;
- el sampler pooled balanced;
- el presupuesto HPO;
- el esquema de pesos persistidos.

El siguiente cambio arquitectónico continúa siendo `22.4 — Autoregressive Residual Refinement`. El hotfix de visualización `22.3.3 — Forecast Monitoring and Plot Safety` puede ejecutarse antes si se prioriza corregir los gráficos observados.
