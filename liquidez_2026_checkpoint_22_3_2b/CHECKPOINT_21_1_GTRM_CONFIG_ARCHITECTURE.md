# Checkpoint 21.1 — GTRM Config Architecture

## Objetivo

Centralizar la arquitectura de configuración de GTRM en un único objeto para que
notebooks, dataset factory, manifiesto de representación y etapas futuras usen la
misma fuente de verdad.

## Nuevo contrato

Se agrega `gtrm_config.py` con `GTRMModelConfig`:

```python
GTRMModelConfig(
    architecture="rnn",
    use_static_context=True,
    use_calendar_encoder=True,
    use_patch_tokenizer=False,
    use_local_residual_decoder=False,
    use_quantile_head=False,
    use_self_supervised_pretraining=False,
    loss_type="huber",
    use_hpo=True,
    latent_dim=None,
)
```

Las flags futuras permanecen apagadas en Stage 1. Si se intenta activar
`use_local_residual_decoder`, `use_quantile_head`, `use_patch_tokenizer` o
`use_self_supervised_pretraining` durante Stage 1, la configuración falla rápido.

## Dónde se ve en notebooks

En cada `code_03_GLOBAL_*.ipynb`, la celda `#@title Configuración general — Financial-GPT`
ahora contiene:

```python
# GTRM — arquitectura modular Stage 1 / 21.1
USE_STATIC_CONTEXT = True
USE_CALENDAR_ENCODER = True
USE_PATCH_TOKENIZER = False
USE_LOCAL_RESIDUAL_DECODER = False
USE_QUANTILE_HEAD = False
USE_SELF_SUPERVISED_PRETRAINING = False
USE_HPO = True
LATENT_DIM = None
```

Luego el notebook construye:

```python
gtrm_model_config = GTRMModelConfig.from_notebook_globals(globals())
gtrm_model_config.validate(stage=1)
```

y lo pasa a:

- `GlobalNotebookConfig`
- `GlobalNotebookDatasetFactory`

## Decisiones

- `use_static_context=True` queda como default porque el contexto estático actual
  no contiene identidad dura: usa tipo de serie, divisa, escala causal y edad.
- `cross_key_id`, `account_currency_id`, `divisa` y `tipo_serie` siguen siendo
  metadata/auditoría; no entran como identificadores duros al forward.
- No se implementa residual local, quantiles, patch tokenizer ni SSL en este
  checkpoint. Sólo se crean hooks limpios para activarlos después.

## Validación esperada

```bash
python -m unittest tests/test_checkpoint_21_1_gtrm_config_architecture.py -v
python -m compileall -q .
```
