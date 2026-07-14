# Checkpoint 2 — Dataset global de ventanas

## Objetivo

Implementar la capa de datos del futuro modelo global sin crear arquitecturas, HPO ni ciclos de entrenamiento.

## Componentes nuevos

### `ContextScaler`

Normaliza cada muestra utilizando exclusivamente su contexto histórico:

```text
center = mediana(y_context)
scale  = IQR(y_context) / 1.349
```

Si la ventana es constante utiliza una escala mínima. Los mismos parámetros transforman `y_future` y permiten regresar el forecast a escala original.

No se ajusta ningún scaler con la historia completa de una cuenta ni con valores futuros.

### `GlobalSeriesSplit`

Construye una partición reproducible por `cross_key_id`:

```text
train_series
validation_seen_series
validation_unseen_series
test_unseen_series
```

`validation_seen_series` conserva las identidades de entrenamiento porque su evaluación deberá realizarse sobre fechas futuras. Las particiones unseen son completamente disjuntas.

### `GlobalWindowDataset`

Cada muestra contiene:

```text
model_inputs:
    y_context       [window_size, 1]
    x_history       [window_size, num_exogenous]
    x_future        [horizon, num_exogenous]
    context_mask    [window_size, 1]

targets:
    y_future        [horizon, 1] normalizado con el contexto
    y_future_raw    [horizon, 1] en escala original

metadata:
    cross_key_id
    account_currency_id
    tipo_serie
    cutoff
    center
    scale
    difficulty_score
    nivel_curriculum
    grupo
```

Los identificadores permanecen exclusivamente en `metadata` y nunca entran a `model_inputs`.

El calendario financiero puede suministrarse como un dataframe independiente, con una fila por `fecha`. Sus variables se alinean de manera causal como contexto histórico y covariables futuras conocidas.

### `SeriesBalancedSampler`

Muestrea en dos pasos:

1. selecciona una serie uniformemente;
2. selecciona una ventana de esa serie.

Esto evita que una serie larga domine el entrenamiento sólo por producir más ventanas.

## Invariantes

- Ninguna ventana cruza entre `cross_key_id`.
- El `cross_key_id` no aparece como feature.
- La normalización utiliza únicamente `y_context`.
- `x_history` y `x_future` se alinean por fecha.
- Fechas faltantes en el calendario exógeno producen error explícito.
- Los splits zero-shot se realizan antes de crear ventanas.
- No se agregaron modelos ni dependencias de Optuna.

## No regresión

Permanecen byte a byte idénticos al Checkpoint 1:

- `code_01.ipynb`
- los cuatro `code_02_*`
- `engineer.py`
- `scientist.py`
- `manager.py`
- `models.py`
- `suggests.py`
- `losses.py`
- `tools.py`
- ambos monitores
- `global_contracts.py`
- `global_long_schema.py`

## Gate

```bash
python -m unittest discover -s tests -p "test_checkpoint_*.py" -v
python -m compileall -q .
```

Resultado del checkpoint:

```text
28 tests passed
compileall OK
```

El siguiente checkpoint debe implementar exclusivamente las cuatro arquitecturas globales detrás del contrato común. Todavía no debe incorporar HPO, curriculum learning ni orquestación.
