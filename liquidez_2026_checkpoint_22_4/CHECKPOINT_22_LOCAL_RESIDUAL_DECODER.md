# Checkpoint 22 — Local Residual Decoder

## Objetivo

Activar el Stage 2 de GTRM sin romper el contrato Stage 1:

```text
y_context, x_history, x_future, x_static
        ↓
Global Encoder / history_embedding
        ↓
Global Forecast Decoder → y_global
Local Residual Decoder  → delta_local
        ↓
y_pred = y_global + delta_local
```

El residual local es un corrector pequeño, causal y libre de `cross_key_id`.
No sustituye al modelo global: sólo corrige residuales sistemáticos por ventana.

## Cambios principales

### `global_models.py`

- Agrega constants de salida para:
  - `y_global`
  - `delta_local`
- Agrega `_configure_local_residual_decoder(...)`.
- Agrega `_apply_local_residual(...)`.
- Las cuatro arquitecturas globales soportan ahora:
  - `use_local_residual_decoder=False` → comportamiento global puro.
  - `use_local_residual_decoder=True` → `y_pred = y_global + delta_local`.
- `history_embedding` sigue siendo obligatorio.
- El `forward` conserva los mismos cuatro inputs canónicos.

### `global_training.py`

- Extiende `GlobalTrainingConfig` con:
  - `use_local_residual_decoder`
  - `local_residual_lambda`
  - `global_aux_alpha`
  - `local_residual_hidden_size`
  - `local_residual_num_layers`
  - `local_residual_dropout_rate`
- Propaga esos parámetros al `model_config` usado por HPO/candidatos.
- Refactoriza `global_forecast_loss` para incluir:

```text
point_loss(y_pred, y)
+ local_residual_lambda * mean(abs(delta_local))
+ global_aux_alpha * point_loss(y_global, y)
+ pérdidas auxiliares ya existentes: KL / reconstruction
```

### `gtrm_config.py`

- `GTRMModelConfig.validate(stage=2)` permite `use_local_residual_decoder=True`.
- Stage 2 sigue rechazando:
  - `use_quantile_head=True`
  - `use_patch_tokenizer=True`
  - `use_self_supervised_pretraining=True`

### `global_notebook.py` y notebooks `code_03_GLOBAL_*`

- Agrega `GTRM_STAGE = 2`.
- Expone flags y parámetros de residual local en la celda principal:
  - `USE_LOCAL_RESIDUAL_DECODER`
  - `LOCAL_RESIDUAL_LAMBDA`
  - `GLOBAL_AUX_ALPHA`
  - `LOCAL_RESIDUAL_HIDDEN_SIZE`
  - `LOCAL_RESIDUAL_NUM_LAYERS`
  - `LOCAL_RESIDUAL_DROPOUT_RATE`
- Mantiene default conservador:

```python
USE_LOCAL_RESIDUAL_DECODER = False
```

Para correr Stage 2 real se activa manualmente:

```python
USE_LOCAL_RESIDUAL_DECODER = True
```

## Contrato de salida

Con residual apagado:

```python
output = {
    "y_pred": ...,
    "extras": {
        "history_embedding": ...,
        "use_local_residual_decoder": False,
    },
}
```

Con residual activado:

```python
output = {
    "y_pred": y_global + delta_local,
    "extras": {
        "history_embedding": ...,
        "use_local_residual_decoder": True,
        "y_global": y_global,
        "delta_local": delta_local,
        "local_residual_lambda": ...,
        "global_aux_alpha": ...,
    },
}
```

## Validación

Comandos ejecutados:

```bash
python -m pytest -q tests/test_checkpoint_22_local_residual_decoder.py
python -m pytest -q tests/test_checkpoint_21_1_gtrm_config_architecture.py tests/test_checkpoint_22_local_residual_decoder.py
python -W error::SyntaxWarning -m compileall -q .
```

Resultados en este contenedor:

```text
Checkpoint 22 targeted tests: 3 passed, 2 skipped
Checkpoint 21.1 + 22 targeted tests: 7 passed, 2 skipped
compileall strict: PASS
```

Los skips corresponden a pruebas que importan `global_training.py`, porque este
contenedor no tiene `polars`. En el entorno real del proyecto deben correr sin
skip.

## Criterio de aceptación experimental

Checkpoint 22 no se acepta sólo por bajar WMAPE global. Debe demostrar:

```text
M2 = Global + Local Residual
vs
M1 = Global Only
```

con énfasis en:

- `%series_improved`
- `macro_MASE`
- `P90 error`
- `WMAPE`
- magnitud media de `delta_local`
- cohortes donde el residual ayuda o daña

El residual debe ser pequeño y útil. Si `delta_local` domina `y_global`, el head
está reemplazando al global y debe regularizarse más.
