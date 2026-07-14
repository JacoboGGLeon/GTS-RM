# Checkpoint 22.3 — Modality-Specific Input Encoders

## Objetivo

Separar la representación de cada modalidad antes de construir el
`history_embedding` compartido de GTRM/GTS-RM.

El checkpoint evita que `y_context`, `x_history` y `x_static` se mezclen como
variables crudas desde la primera capa. También codifica `x_future` por paso del
horizonte antes de entregarlo a los decoders y auxiliary heads.

Este checkpoint **no agrega autoregresión** y **no cambia la definición del
forecast final**:

```text
y_pred = y_global + delta_local
```

## Arquitectura implementada

Cuando `use_modality_specific_encoders=True`:

```text
y_context  -> Target Encoder -------------------------┐
x_history  -> Historical Exogenous Encoder ---------┤
x_static   -> Static Context Encoder ----------------┤
                                                       ├-> Fusion Encoder
                                                       │      -> history_embedding
x_future + posición_h -> Future Exogenous Encoder ----┘
                              -> future_embedding[h]
```

El contexto futuro entregado al decoder se construye como:

```text
future_context[h] = concat(future_embedding[h], static_embedding)
```

Y el forecast directo conserva:

```text
concat(history_embedding, future_context[h])
    -> global decoder
    -> y_global[h]
```

El residual decoder y los heads `event`, `magnitude` y `direction` reciben el
mismo `future_context` codificado. No se crean rutas paralelas con inputs crudos.

## Implementación por arquitectura

### MLP y MLP-VAE

- `y_context`: MLP sobre la ventana objetivo aplanada.
- `x_history`: MLP sobre las covariables históricas aplanadas.
- `x_static`: MLP sobre el contexto semántico estático.
- `x_future`: MLP compartido aplicado a cada paso futuro junto con su posición
  normalizada dentro del horizonte.
- fusión: MLP explícito que produce `history_embedding`.
- MLP-VAE: la fusión produce el estado compartido usado por `mu` y `logvar`.

### RNN y RNN-Bi

- `y_context`: GRU exclusiva para la serie objetivo.
- `x_history`: GRU exclusiva para covariables históricas.
- `x_static`: MLP.
- `x_future`: MLP por paso futuro.
- fusión: MLP que produce `history_embedding`.
- RNN-Bi usa bidireccionalidad sólo dentro del contexto histórico observado y
  proyecta cada estado al ancho contractual configurado.
- el decoder futuro continúa siendo GRU unidireccional y directo
  multi-horizonte.

## Contrato y diagnósticos

`output["extras"]` conserva `history_embedding` y, cuando la separación está
activa, expone:

- `target_embedding`;
- `historical_exogenous_embedding`;
- `future_exogenous_embedding`;
- `static_context_embedding`;
- `use_modality_specific_encoders`.

`representation_contract()` identifica el stage como:

```text
GTRM_STAGE_2_3_MODALITY_SPECIFIC_INPUT_ENCODERS
```

Los identificadores contables siguen fuera de `forward`.

## Configuración y ablation

Se agrega la bandera:

```python
USE_MODALITY_SPECIFIC_ENCODERS = True
```

La ruta anterior permanece disponible con:

```python
USE_MODALITY_SPECIFIC_ENCODERS = False
```

Esto permite comparar de manera controlada:

```text
joint raw-input encoder
vs
modality-specific encoders + fusion
```

La arquitectura corresponde a Stage 22.3. El contrato Pydantic vigente del notebook sube posteriormente a `schema_version="22.3.1"` y exige que la
bandera del modelo coincida con la bandera del entrenamiento.

## HPO agregado

Con `HPO_MODALITY_ENCODER_ARCHITECTURE=True`, Optuna explora:

- dimensión de `target_encoder`;
- dimensión de `historical_exogenous_encoder`;
- dimensión de `future_exogenous_encoder`;
- dimensión de `static_context_encoder`;
- tamaño oculto de fusión;
- capas de cada encoder;
- capas de fusión;
- dropout de encoders;
- activación de encoders.

Los encoders y decoders previos continúan dentro del espacio HPO. Este
checkpoint amplía capacidad estructural sin aumentar todavía el número de
trials o epochs.

## Compatibilidad

- El contrato público de `forward` no cambia.
- `history_embedding` conserva shape `[batch, latent_dim]`.
- El horizonte conserva shape `[batch, horizon, 1]`.
- El decoder global sigue siendo directo multi-horizonte.
- El residual decoder y auxiliary heads conservan sus losses.
- La ruta legacy sigue disponible para ablation.
- Los `state_dict` de la nueva ruta cargan estrictamente al reconstruir la misma
  configuración.

## Validación

Se agregaron siete tests específicos para:

- validación de stage y bandera;
- shapes en las cuatro arquitecturas;
- gradientes desde los cuatro inputs;
- independencia previa a la fusión;
- espacio HPO completo;
- ruta legacy de ablation;
- notebooks y round-trip estricto de pesos.

La regresión completa fue ejecutada por grupos aislados para evitar la
acumulación de recursos observada en la ejecución monolítica:

```text
199 passed
```

También pasaron:

- `compileall` estricto;
- compilación de las cuatro notebooks;
- outputs y execution counts vacíos.

## Fuera de alcance

El siguiente checkpoint implementará:

```text
22.4 — Autoregressive Residual Refinement
```

Todavía no se implementan:

- teacher forcing;
- scheduled sampling;
- consumo de la predicción anterior;
- rolling-origin evaluation nuevo;
- aumento del presupuesto HPO o de entrenamiento.
