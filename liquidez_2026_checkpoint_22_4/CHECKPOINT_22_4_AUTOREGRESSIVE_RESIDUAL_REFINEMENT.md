# Checkpoint 22.4 — Autoregressive Residual Refinement

## Objetivo

Convertir el residual local paralelo de Stage 2 en un refinamiento causal a lo
largo del horizonte y reducir el HPO a decisiones con señal suficiente para
seleccionar un modelo productivo.

## Modelo

El pronóstico conserva la descomposición:

```text
y_pred[t] = y_global[t] + delta_local[t]
```

`delta_local[t]` se obtiene con celdas GRU compartidas y depende de:

- `history_embedding` causal;
- covariables conocidas del horizonte `t`;
- `y_global[t-1]`;
- `delta_local[t-1]`.

El decoder no usa `y_target`, identidad de serie ni teacher forcing. Cambiar
covariables de un horizonte posterior no puede modificar residuos anteriores.
El modo paralelo anterior permanece disponible sólo como ablation compatible.

## HPO compacto

Se mantienen como ejes de búsqueda:

- ventana: `5, 10, 15, 20`;
- dimensión latente: `32, 64`;
- dropout global: `0.00..0.20`;
- activación: `gelu, silu`;
- learning rate;
- capacidad principal compacta por familia;
- capacidad de fusión y contexto estático.

Los encoders temporales target/history/future comparten dimensión y profundidad
por defecto. Salen del espacio HPO:

- weight decay;
- capacidad y peso del autoencoder auxiliar;
- shares de heads auxiliares;
- dimensiones/capas temporales independientes;
- decoder RNN depth;
- anchos encoder/decoder MLP separados.

Estas decisiones quedan como política interna con defaults seguros. Los
notebooks conservan controles de usuario para fuentes de datos, horizonte,
splits, presupuesto de cómputo, entrenamiento e inferencia.

## Presupuesto recomendado

```text
36 trials proxy x 4 epochs
4 finalistas x 8 epochs medium-fidelity
6 ventanas/serie proxy
12 ventanas/serie medium-fidelity
```

## Compatibilidad

El contrato de salida, losses, persistencia, backtest y forecast no cambia. Los
pesos de un residual paralelo anterior no son compatibles con el nuevo módulo
autoregresivo y requieren reentrenamiento.
