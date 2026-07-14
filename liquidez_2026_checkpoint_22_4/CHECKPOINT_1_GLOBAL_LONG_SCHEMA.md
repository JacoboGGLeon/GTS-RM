# Checkpoint 1 — Esquema largo canónico

## Objetivo

Canonizar y validar el dataset largo que alimentará posteriormente al modelo global, sin crear ventanas y sin modificar el entrenamiento local existente.

## Mapeo aplicado

El `series_long` legacy conserva esta semántica:

```text
cross_key_id = cuenta + divisa
serie       = cuenta + divisa + tipo_serie
```

La nueva salida `global_series_long` usa:

```text
account_currency_id = cuenta + divisa
cross_key_id         = cuenta + divisa + tipo_serie
```

El `cross_key_id` se conserva exclusivamente como metadata de agrupación, partición, reconstrucción y auditoría. No es una feature del modelo.

## Esquema exacto

```text
fecha
account_currency_id
cross_key_id
tipo_serie
target
difficulty_score
nivel_curriculum
grupo
```

Correspondencias desde el formato legacy:

```text
total_amount      -> target
curriculum_bucket -> nivel_curriculum
```

`target` se mantiene en escala original. `total_amount_robust` no se usa en la salida canónica porque la normalización contextual debe calcularse sólo con la ventana histórica en Checkpoint 2.

## Validaciones implementadas

- columnas y orden canónicos;
- dataset no vacío;
- ausencia de nulos;
- identificadores no vacíos;
- `tipo_serie` restringido a `saldo` y `variacion`;
- `cross_key_id = account_currency_id + tipo_serie`;
- target y dificultad finitos;
- dificultad dentro de `[0, 1]`;
- nivel curricular entero y mayor o igual a 1;
- unicidad de `(cross_key_id, fecha)`.

## Salidas nuevas de `code_01.ipynb`

```text
global_series_long.csv
global_series_long.parquet
global_series_long_validation.json
```

Las salidas legacy se mantienen intactas para los cuatro `code_02_*`.

## No regresión

No se modificaron:

- `code_02_MLP_E_D.ipynb`
- `code_02_MLP_VaE_D.ipynb`
- `code_02_RNN_E_D.ipynb`
- `code_02_RNNBi_E_D.ipynb`
- `engineer.py`
- `scientist.py`
- `manager.py`
- `models.py`
- `suggests.py`
- `losses.py`
- `tools.py`
- monitores existentes

## Gate

```bash
python -m unittest discover -s tests -p "test_checkpoint_*.py" -v
python -m compileall -q .
```

El siguiente checkpoint debe implementar exclusivamente el dataset de ventanas global, el split por series y el sampler balanceado. Todavía no debe crear arquitecturas ni entrenamiento global.
