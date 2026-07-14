# Checkpoint 22.3.2 — Active Configuration and HPO Surface Hardening

## Objetivo

Limpiar la superficie de configuración de los cuatro notebooks globales y
hacer explícita la diferencia entre:

- capacidades realmente implementadas;
- defaults usados como fallback o ablation;
- espacio arquitectónico que Optuna explora;
- presupuesto de búsqueda;
- entrenamiento pooled productivo;
- inferencia y visualización.

Este checkpoint no cambia el `forward`, los encoders, los decoders, los heads,
los losses ni la construcción causal de ventanas.

## Problema corregido

La celda principal mezclaba controles con semánticas diferentes:

```text
FUTURE_ENCODER_DIM = 32
STATIC_ENCODER_DIM = 16
FUSION_HIDDEN_SIZE = 64
```

parecían fijar la arquitectura final, aunque Optuna los reemplazaba cuando el
HPO de encoders estaba activo. También se mostraban hooks sin implementación
productiva en Stage 2.3:

```text
USE_CALENDAR_ENCODER
USE_PATCH_TOKENIZER
USE_QUANTILE_HEAD
USE_SELF_SUPERVISED_PRETRAINING
USE_HPO
LATENT_DIM
```

Esto hacía que la celda no representara fielmente el comportamiento real.

## Nueva superficie Pydantic

Se agrega `global_surface_config.py` con contratos estrictos e inmutables:

```text
ModelFeatureConfig
ModalityEncoderDefaults
ModalityEncoderHPOSpace
ResidualDecoderConfig
AuxiliaryHeadsConfig
TrainingBudgetConfig
InferenceConfig
GlobalActiveConfiguration
```

Pydantic se usa sólo en la frontera de configuración. Tensores, batches y
objetos internos de PyTorch permanecen fuera del contrato para evitar
sobrecoste durante entrenamiento.

## Defaults frente a HPO

Los notebooks ahora separan:

```python
MODALITY_ENCODER_DEFAULTS = {...}
MODALITY_ENCODER_HPO_SPACE = {...}
```

La semántica es:

```text
HPO enabled = False
    -> se usan los defaults

HPO enabled = True
    -> los defaults inicializan el contrato base
    -> Optuna reemplaza dimensiones/capas/dropout/activación
    -> el candidato ganador define la arquitectura productiva
```

El espacio HPO ya no está hardcodeado dentro de
`suggest_global_candidate()`. Se valida mediante
`ModalityEncoderHPOSpace`, se serializa dentro de `GlobalHPOConfig` y se pasa
explícitamente al generador de candidatos.

## Espacio multi-encoder vigente

```text
target encoder dimension:       16, 32, 64, 128
historical encoder dimension:   16, 32, 64, 128
future encoder dimension:       16, 32, 64, 128
static encoder dimension:        8, 16, 32, 64
fusion hidden size:             32, 64, 128, 256

target/historical/future layers: 1..3
static layers:                    1..2
fusion layers:                    1..3
dropout:                          0.00..0.35
activation:                       relu, gelu, silu, tanh
```

## Presupuesto HPO actualizado

La búsqueda se amplía para corresponder con el tamaño del espacio
arquitectónico:

```python
HPO_TRIALS = 80
HPO_EPOCHS = 5
HPO_WINDOWS_PER_SERIES = 8
HPO_VALIDATION_WINDOWS_PER_SERIES = 5
HPO_FINALISTS = 8
HPO_FIDELITY_EPOCHS = 12
HPO_FIDELITY_WINDOWS_PER_SERIES = 16
```

El entrenamiento productivo permanece:

```python
TRAINING_STRATEGY = "pooled_balanced"
POOLED_TRAIN_EPOCHS = 60
POOLED_CONTINUATION_EPOCHS = 0
```

No se introducen warm-up, fine-tuning o consolidation nominales.

## Contrato raíz

`GlobalNotebookRunContract` sube a:

```text
schema_version = 22.3.2
```

El contrato puede incluir `surface=active_config` y valida coherencia entre:

- superficie activa y `GTRMModelConfig`;
- defaults Pydantic y `GlobalTrainingConfig`;
- espacio Pydantic y `GlobalHPOConfig`;
- presupuesto y configuración proxy/medium-fidelity;
- presupuesto y schedule pooled.

Los notebooks persisten adicionalmente:

```text
reports/active_configuration.json
```

`execution_config.json` referencia la configuración activa completa, en lugar
de duplicar decenas de variables sueltas.

## Controles retirados de los notebooks

```text
USE_CALENDAR_ENCODER
USE_PATCH_TOKENIZER
USE_QUANTILE_HEAD
USE_SELF_SUPERVISED_PRETRAINING
USE_HPO
LATENT_DIM
HPO_MODALITY_ENCODER_ARCHITECTURE
```

Los hooks históricos pueden permanecer en componentes internos por
compatibilidad, pero ya no aparecen como controles productivos.

## Validación

La regresión completa se ejecutó por grupos aislados:

```text
207 passed
```

Además:

- 4 gates específicos de 22.3.2;
- `compileall` estricto;
- compilación de las cuatro notebooks;
- outputs vacíos y `execution_count=None`;
- ausencia de controles muertos en la superficie activa;
- validación de espacios HPO personalizados;
- persistencia serializable del nuevo contrato.

## Siguiente checkpoint

```text
22.3.2b — Autoregressive Residual Refinement
```
