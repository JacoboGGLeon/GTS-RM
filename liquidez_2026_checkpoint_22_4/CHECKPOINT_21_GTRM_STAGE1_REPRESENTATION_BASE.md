# Checkpoint 21 — GTRM Stage 1 Representation Base

## Objetivo

Cerrar la base global de representación de GTRM antes de agregar residual local,
cuantiles, patching continuo o pretraining autosupervisado.

Este checkpoint no cambia la filosofía del Checkpoint 20: mantiene el modelo
global supervisado, el escalado causal, el sampler balanceado y la selección por
`robust_macro_mase`. Lo que agrega es un contrato explícito de representación:
toda arquitectura global debe devolver un `history_embedding` causal, finito y
exportable.

## Contrato operativo

El `forward` sigue recibiendo únicamente los cuatro tensores canónicos:

```python
(y_context, x_history, x_future, x_static)
```

Los identificadores siguen prohibidos en el `forward`:

- `cross_key_id`
- `account_currency_id`
- `divisa`
- `tipo_serie` crudo
- `serie`

La salida canónica del modelo es:

```python
{
    "y_pred": Tensor[batch, horizon, 1],
    "extras": {
        "history_embedding": Tensor[batch, latent_dim],
        ...
    }
}
```

## Flags GTRM

Se formalizan las flags de la matriz de ablation del manifiesto:

```python
{
    "use_static_context": True,
    "use_patch_tokenizer": False,
    "use_local_residual_decoder": False,
    "use_quantile_head": False,
    "use_self_supervised_pretraining": False,
}
```

En Checkpoint 21 sólo `use_static_context` tiene efecto real. Las demás quedan
como hooks explícitos y validados para etapas futuras:

- `use_local_residual_decoder` → Stage 2
- `use_quantile_head` → Stage 3
- `use_self_supervised_pretraining` → Stage 4
- `use_patch_tokenizer` → Stage 5 / ablation posterior

## Cambios implementados

### `global_contracts.py`

- Agrega constantes de salida:
  - `GLOBAL_OUTPUT_FIELD = "y_pred"`
  - `HISTORY_EMBEDDING_FIELD = "history_embedding"`
  - `RECONSTRUCTION_FIELD = "context_reconstruction"`
- Agrega `GTRM_STAGE_FLAGS` y `DEFAULT_GTRM_STAGE1_FLAGS`.
- Agrega `default_gtrm_stage1_flags()`.
- Agrega `validate_gtrm_stage_flags()`.
- Extiende `GlobalModelContract` con `output_field` y `latent_field`.

### `global_models.py`

- Reutiliza constantes canónicas de salida desde `global_contracts.py`.
- Agrega `get_history_embedding(output)`.
- Agrega `validate_global_model_output(...)`.
- Agrega `GlobalForecastModel.representation_contract()`.

### `global_data.py`

- `StaticFeatureEncoder` ahora soporta modo deshabilitado.
- `GlobalWindowDataset(..., use_static_context=False)` conserva el contrato de
  `forward`, pero emite `x_static = zeros([1])` con feature name
  `static_context_disabled`.
- Default: `use_static_context=True`, compatible con Checkpoint 20.

### `global_notebook.py`

- `GlobalNotebookDatasetFactory` acepta `use_static_context=True/False`.
- El resumen de la factoría reporta `use_static_context`.
- Los datasets creados por la factoría reciben la bandera explícitamente.

### `gtrm_representation.py`

Nuevo módulo para cerrar el Stage 1:

- `GTRMStage1Config`
- `gtrm_stage1_manifest(...)`
- `collect_history_embeddings(...)`

`collect_history_embeddings` exporta embeddings por ventana junto con metadata
sólo para análisis posterior. Los ids nunca se pasan al `forward`.

### Tests

Nuevo gate dedicado:

```text
tests/test_checkpoint_21_gtrm_stage1_representation_base.py
```

Cubre:

- flags Stage 1 y rechazo de residual local en Stage 1;
- `use_static_context=False` sin romper `MODEL_INPUT_FIELDS`;
- las cuatro arquitecturas devuelven `history_embedding` válido;
- exportación de embeddings con metadata;
- manifiesto GTRM Stage 1 con métricas de aceptación.

## Criterio de aceptación experimental

Checkpoint 21 cierra la implementación de la base global de representación,
pero el cierre experimental requiere un reporte Stage 1 con:

- `robust_macro_mase`
- `raw_macro_wmape`
- `p90_series_error`
- `%series_improved`
- cortes por `tipo_serie`, `divisa`, `grupo`, `nivel_curriculum` y seen/unseen

No se debe avanzar a residual local si la base global no es competitiva contra
baselines en precisión por serie.

## Próximo checkpoint

Checkpoint 22 — `Global + Local Residual Decoder`.

Sólo después de validar que el `history_embedding` y el decoder global son
competitivos contra baselines.
