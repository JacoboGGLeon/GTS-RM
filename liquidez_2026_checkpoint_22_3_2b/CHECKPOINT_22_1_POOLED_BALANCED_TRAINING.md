# Checkpoint 22.1 — Pooled Balanced Training Schedule

Objetivo: agregar una ruta rápida y precisa de entrenamiento global que no dependa de la secuencia `warmup -> fine-tune curricular -> consolidation`.

## Decisión

Se agrega `TRAINING_ORDER = "pooled"` a `GlobalCurriculumConfig`.

Con `training_order="pooled"`, el entrenamiento productivo construye una sola etapa:

```text
pooled_balanced_all_levels
current_levels = todos los niveles curriculares
replay_levels = ()
replay_fraction = 0.0
epochs = WARM_EPOCHS
learning_rate_factor = 1.0
```

Esto conserva el sampler balanceado por serie/grupo/nivel, pero evita los saltos por nivel y el posible forgetting entre fases.

## Config copiada para la siguiente corrida

Las cuatro notebooks globales quedan con la configuración rápida solicitada:

```python
USE_STATIC_CONTEXT = True
USE_LOCAL_RESIDUAL_DECODER = True
USE_HPO = True
HPO_TRIALS = 30
HPO_EPOCHS = 3
HPO_BATCH = 512
WARM_EPOCHS = 25
FINE_EPOCHS = 35
TRAINING_ORDER = "pooled"
```

`FINE_EPOCHS` y `CONSOLIDATION_EPOCHS` se conservan para comparar contra `training_order="curriculum"`, pero en modo `pooled` sólo se usa `WARM_EPOCHS` como presupuesto productivo total.

## Por qué

El curriculum actual es útil cuando el modelo se vuelve inestable, pero también puede sesgar al modelo hacia niveles fáciles o causar forgetting. Para GTRM Stage 2 necesitamos una comparación limpia:

```text
A: curriculum + residual
B: pooled balanced + residual
```

El HPO se mantiene porque sí está funcionando como proxy: explora configuraciones, poda trials débiles, encuentra una región estable y luego el entrenamiento productivo usa el mejor candidato.
