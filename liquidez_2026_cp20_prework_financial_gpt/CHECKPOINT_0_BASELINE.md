# Checkpoint 0 — Baseline y contratos del modelo global

## Objetivo

Congelar el comportamiento actual y fijar los contratos mínimos de la futura variante global sin modificar el entrenamiento existente.

## Estado real del bundle

- `code_01.ipynb` ya genera `series_long.csv`, `series_long.parquet` y sus variantes robustas.
- Los cuatro `code_02_*` siguen entrenando un modelo independiente por serie.
- `Engineer` mantiene datasets y transformadores separados por serie.
- `Scientist` mantiene `models[serie]` y configuraciones por serie.
- `Manager` recorre las series para HPO, warm-up, fine-tuning, backtest y forecast.
- Los cuatro nombres de arquitectura actuales son `mlp`, `mlp_vae`, `rnn` y `rnn_bi`.
- Los notebooks no contienen outputs persistidos.

## Hallazgo de esquema

El formato long actual conserva:

- `cross_key_id`: cuenta + divisa.
- `tipo_serie`: `saldo` o `variacion`.
- `serie`: cuenta + divisa + tipo de serie.

El contrato solicitado define, para la variante global:

```text
account_currency_id = cuenta + divisa
cross_key_id         = cuenta + divisa + tipo_serie
```

La migración física del esquema queda para el Checkpoint 1. En Checkpoint 0 sólo se fija la regla para evitar cambios ambiguos.

## Invariantes aprobados

1. `cross_key_id` es metadata de agrupación, partición, auditoría y reconstrucción.
2. `cross_key_id`, `account_currency_id`, `tipo_serie` y `serie` no pueden entrar a `forward()`.
3. Los inputs autorizados del futuro modelo son:
   - `y_context`
   - `x_history`
   - `x_future`
   - `context_mask`
4. Las cuatro arquitecturas globales compartirán el mismo contrato.
5. Los notebooks y clases locales permanecen intactos hasta superar el baseline.

## Gate del checkpoint

Ejecutar:

```bash
python -m unittest discover -s tests -p "test_checkpoint_0_*.py" -v
python -m compileall -q .
```

El siguiente checkpoint debe limitarse a canonizar y validar el dataset long. No debe implementar todavía ventanas, sampler, modelos ni entrenamiento global.
