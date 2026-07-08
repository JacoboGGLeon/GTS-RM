# Prework Summary for Jacobo

## Qué conserva tu avance

El CP20 ya trae:

- Global long schema.
- Split causal seen/unseen por identidad.
- Scaler causal reversible.
- Calendar/exogenous context/future.
- Static context semántico y no identificador.
- Cuatro modelos globales.
- HPO proxy + medium fidelity.
- Sampler balanceado.
- Curriculum vs shuffled.
- Objective coherente `robust_macro_mase`.
- Monitor único Financial-GPT.

## Qué falta para el nuevo plan

No hay que rehacer Stage 0 ni Stage 1. Lo importante ahora es:

1. Meter flags sin romper CP20.
2. Agregar `Local Residual Decoder`.
3. Hacer diagnóstico por serie M1 vs M2.
4. Después agregar quantile head.
5. Después hacer SSL real.
6. Al final probar patch tokenizer continuo y static context ablation.

## Siguiente checkpoint recomendado

```text
Checkpoint 21 — Feature flags and CP20 compatibility gate
```

No debe cambiar la predicción. Sólo prepara el sistema para variantes:

```python
use_patch_tokenizer=False
use_static_context=True
use_local_residual_decoder=False
use_quantile_head=False
use_self_supervised_pretraining=False
```

## Primer checkpoint con ganancia esperada de precisión individual

```text
Checkpoint 23 — Local residual decoder
```

Ese es el que implementa:

```text
y_pred = y_global + delta_local
```

con regularización para que `delta_local` corrija, no domine.

