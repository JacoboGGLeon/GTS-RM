# Checkpoint 22.2.1 — Auxiliary Loss Block + HPO de coeficientes

## Objetivo

Checkpoint 22.2 agregó los heads agnósticos `event_head`, `magnitude_head` y `direction_head`. Checkpoint 22.2.1 ordena su contribución a la loss como un bloque auxiliar normalizado e hiperparametrizable, sin cambiar el forecast final.

El forecast sigue siendo:

```text
y_pred = y_global + delta_local
```

Los heads auxiliares siguen siendo supervisión auxiliar para mejorar `history_embedding`, no componentes directos de `y_pred`.

## Loss antes de 22.2.1

```text
loss = value_loss
     + LOCAL_RESIDUAL_LAMBDA * residual_regularization
     + GLOBAL_AUX_ALPHA * global_aux_loss
     + EVENT_LOSS_WEIGHT * event_loss
     + MAGNITUDE_LOSS_WEIGHT * magnitude_loss
     + DIRECTION_LOSS_WEIGHT * direction_loss
```

## Loss desde 22.2.1

Default recomendado:

```text
loss = value_loss
     + LOCAL_RESIDUAL_LAMBDA * residual_regularization
     + GLOBAL_AUX_ALPHA * global_aux_loss
     + AUXILIARY_LOSS_WEIGHT * (
           EVENT_LOSS_SHARE * event_loss
         + MAGNITUDE_LOSS_SHARE * magnitude_loss
         + DIRECTION_LOSS_SHARE * direction_loss
       )
```

con la restricción:

```text
EVENT_LOSS_SHARE + MAGNITUDE_LOSS_SHARE + DIRECTION_LOSS_SHARE = 1.0
```

Default:

```python
USE_AUXILIARY_LOSS_BLOCK = True
AUXILIARY_LOSS_WEIGHT = 0.20
EVENT_LOSS_SHARE = 0.40
MAGNITUDE_LOSS_SHARE = 0.40
DIRECTION_LOSS_SHARE = 0.20
HPO_AUXILIARY_LOSS_WEIGHTS = True
```

Los pesos legacy permanecen disponibles sólo si:

```python
USE_AUXILIARY_LOSS_BLOCK = False
```

## HPO

Cuando `HPO_AUXILIARY_LOSS_WEIGHTS=True`, el HPO proxy puede explorar:

```text
auxiliary_loss_weight ∈ {0.0, 0.05, 0.10, 0.20, 0.30}
event_loss_share_raw ∈ [0.05, 1.0]
magnitude_loss_share_raw ∈ [0.05, 1.0]
direction_loss_share_raw ∈ [0.05, 1.0]
```

Los valores raw se normalizan para que las shares efectivas sumen 1.0.

## Compatibilidad

- No cambia el contrato de `forward`.
- No cambia `y_pred`.
- No rompe el esquema legacy de pesos directos.
- Las notebooks exponen las nuevas variables en la celda de configuración.
- `LOCAL_RESIDUAL_LAMBDA` y `GLOBAL_AUX_ALPHA` siguen separados porque son regularizadores estructurales del forecast, no shares de heads auxiliares.

## Archivos modificados

- `global_training.py`
- `global_models.py`
- `code_03_GLOBAL_MLP_E_D.ipynb`
- `code_03_GLOBAL_MLP_VaE_D.ipynb`
- `code_03_GLOBAL_RNN_E_D.ipynb`
- `code_03_GLOBAL_RNNBi_E_D.ipynb`

## Tests

- `tests/test_checkpoint_22_2_1_auxiliary_loss_block_hpo.py`
