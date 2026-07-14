# Checkpoint 22.2 — Agnostic Event/Magnitude/Direction Auxiliary Heads

## Objetivo

Agregar tres heads auxiliares agnósticos al tipo de serie para mejorar la representación histórica sin convertir todavía el modelo en multi-output saldo/variación.

Los heads son:

| Orden | Head | Dificultad | Target derivado |
|---:|---|---|---|
| 1 | `event_head` | Fácil | `abs(y_future_scaled) > threshold` sobre movimiento escalado |
| 2 | `magnitude_head` | Fácil | `asinh(abs(movement_scaled))`, `log1p(abs(...))` o `abs(...)` |
| 3 | `direction_head` | Media | `sign(movement_scaled)` como negativo/neutro/positivo |

## Dónde se generan los datos

Los targets se generan en `GlobalWindowDataset.__getitem__`, no en `code_01`, porque dependen de:

- `window_size`;
- `horizon`;
- escala causal de la ventana;
- último valor del contexto;
- `tipo_serie`.

La función central es:

```python
build_agnostic_auxiliary_targets(...)
```

El movimiento agnóstico se define así:

```text
saldo:      movement = y_future_scaled - y_context_scaled[-1]
variacion:  movement = y_future_scaled
```

## Flags nuevas

En las notebooks globales:

```python
USE_EVENT_HEAD = True
EVENT_LOSS_WEIGHT = 0.10
USE_MAGNITUDE_HEAD = True
MAGNITUDE_LOSS_WEIGHT = 0.10
USE_DIRECTION_HEAD = True
DIRECTION_LOSS_WEIGHT = 0.05
AUXILIARY_HEAD_HIDDEN_SIZE = 32
AUXILIARY_HEAD_NUM_LAYERS = 1
AUXILIARY_HEAD_DROPOUT_RATE = 0.0
EVENT_THRESHOLD = 1.0
MAGNITUDE_TRANSFORM = "asinh"
```

## Contrato de salida

Cuando se activan, los heads aparecen en `output["extras"]`:

```text
event_logits:      [batch, horizon, 1]
magnitude_pred:   [batch, horizon, 1]
direction_logits: [batch, horizon, 3]
```

## Loss compuesta

La loss principal sigue siendo el forecast final:

```text
value_loss(y_pred, y_future)
```

A eso se agregan términos auxiliares ponderados:

```text
+ EVENT_LOSS_WEIGHT * BCEWithLogits(event_logits, event_target)
+ MAGNITUDE_LOSS_WEIGHT * Huber(magnitude_pred, magnitude_target)
+ DIRECTION_LOSS_WEIGHT * CrossEntropy(direction_logits, direction_target)
```

Los heads son supervisión auxiliar: no cambian todavía la fórmula final:

```text
forecast = y_global + delta_local
```

## Validación

```bash
python -m pytest -q tests/test_checkpoint_22_2_agnostic_auxiliary_heads.py
python -W error::SyntaxWarning -m compileall -q .
```

Resultado esperado:

```text
3 passed
compileall PASS
```
