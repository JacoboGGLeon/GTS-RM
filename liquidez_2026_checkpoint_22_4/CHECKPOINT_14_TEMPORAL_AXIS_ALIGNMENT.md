# Checkpoint 14 — Temporal Axis Alignment

## Objetivo

Eliminar toda suposición de frecuencia dentro de Financial-GPT. El modelo no
interpreta fines de semana, festivos ni días hábiles: cada fila del proveedor
temporal define un paso válido.

## Implementación

- `TemporalAxis`: timestamps válidos y covariables ordenadas.
- `ForecastRequest`: forecast por `n_steps` o por rango de timestamps.
- `TemporalWindowAligner`: intersección auditable target/exógenas sin modificar
  el frame fuente.
- `InsufficientFutureContextError`: error explícito cuando el proveedor no
  contiene el horizonte requerido.
- Forecast MC-Dropout por pasos del eje, sin `freq="D"`, `freq="B"`, weekdays ni
  aritmética fija de días.
- Reporte estructurado y visible del backtest.
- Reporte de cobertura temporal por `cross_key_id`.

## Contrato

```text
series targets + TemporalAxis
            ↓
TemporalWindowAligner
            ↓
y_context + x_history + x_future + context_mask
```

`cross_key_id` permanece exclusivamente como metadata, partición y trazabilidad.

## Salidas nuevas

```text
reports/temporal_alignment_report.parquet
reports/backtest_run_report.json
```

## Compatibilidad

Los cuatro notebooks globales aceptan:

- ambos `FC_START` y `FC_END`: timestamps válidos del eje dentro del rango;
- ambos vacíos: los próximos `HORIZON` pasos del eje por serie.

Los cuatro notebooks locales, modelos, HPO, currículo, monitores y persistencia
S3 no cambian.
