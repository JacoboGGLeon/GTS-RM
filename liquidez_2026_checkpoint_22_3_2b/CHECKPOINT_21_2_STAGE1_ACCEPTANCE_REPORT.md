# Checkpoint 21.2 — GTRM Stage 1 Acceptance Report

## Objetivo

Cerrar la evidencia mínima del **Global Representation Base** antes de agregar
residual local, cuantiles, patching o preentrenamiento. Este checkpoint no cambia
el entrenamiento ni la arquitectura: convierte las métricas por serie del monitor
en un gate reproducible.

## Nuevo módulo

- `gtrm_acceptance.py`

Funciones y clases principales:

- `Stage1AcceptanceCriteria`
- `Stage1AcceptanceSummary`
- `Stage1AcceptanceReport`
- `build_stage1_acceptance_report(...)`

## Criterio de aceptación

El reporte compara, por cada `cross_key_id`, el mejor candidato global contra el
mejor baseline disponible usando `MASE` por defecto.

Métricas de cierre:

- `%series_improved`
- `macro_model_metric` vs `macro_baseline_metric`
- `p90_model_metric` vs `p90_baseline_metric`
- `wmape_model` vs `wmape_baseline`

Defaults:

```python
Stage1AcceptanceCriteria(
    primary_metric="MASE",
    wmape_metric="WMAPE",
    min_percent_series_improved=55.0,
    max_macro_mase_relative_regression=0.0,
    max_p90_relative_regression=0.0,
    max_wmape_relative_regression=0.02,
)
```

## Salidas auditables

`Stage1AcceptanceReport.write(output_dir)` genera:

- `stage1_acceptance_summary.json`
- `stage1_acceptance_criteria.json`
- `stage1_acceptance_by_series.csv`
- `stage1_acceptance_by_<cohort>.csv` para cortes opcionales

## Cohortes recomendadas

- `tipo_serie`
- `divisa`
- `grupo`
- `nivel_curriculum`

## Regla de roadmap

Si el Stage 1 no pasa este gate, no se debe avanzar a `local_residual_decoder`.
Primero se corrige el Global Representation Base: scaler, sampler, loss,
contexto estático, HPO o arquitectura.
