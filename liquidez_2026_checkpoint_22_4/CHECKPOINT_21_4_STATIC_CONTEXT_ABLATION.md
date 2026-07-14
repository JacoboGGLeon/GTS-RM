# Checkpoint 21.4 — Static Context Ablation

## Objetivo

Formalizar la comparación de Stage 1 entre:

- **GTRM-A**: `use_static_context=False`
- **GTRM-B**: `use_static_context=True`

El objetivo no es agregar arquitectura nueva. Es decidir, con evidencia por serie,
si `x_static` debe quedarse como default del Global Representation Base.

## Artefactos agregados

- `gtrm_static_ablation.py`
- `tests/test_checkpoint_21_4_static_context_ablation.py`

## Salidas del reporte

`build_static_context_ablation_report(...).write(output_dir)` genera:

- `static_context_ablation_summary.json`
- `static_context_ablation_criteria.json`
- `static_context_ablation_by_series.csv`
- `static_context_ablation_by_<cohort>.csv`

## Criterio default

```python
StaticContextAblationCriteria(
    primary_metric="MASE",
    wmape_metric="WMAPE",
    min_percent_series_improved=50.0,
    max_macro_relative_regression=0.0,
    max_p90_relative_regression=0.0,
    max_wmape_relative_regression=0.01,
)
```

Se acepta `use_static_context=True` como default sólo si mejora una proporción
razonable de series y no deteriora macro/P90. WMAPE es secundario y admite una
tolerancia menor porque puede moverse por pocas series de alto volumen.

## Uso esperado

Ejecutar dos corridas comparables del mismo candidato global:

1. `USE_STATIC_CONTEXT = False`
2. `USE_STATIC_CONTEXT = True`

Unir las métricas por serie del monitor en un solo DataFrame con columna
`use_static_context`, y después correr:

```python
from gtrm_static_ablation import build_static_context_ablation_report

report = build_static_context_ablation_report(
    metrics_by_series,
    cohort_columns=("tipo_serie", "divisa", "grupo", "nivel_curriculum"),
)
report.write("gtrm_static_context_ablation")
```

## Decisión operacional

Este checkpoint no cambia el forward ni activa etapas futuras. Si el reporte
acepta GTRM-B, `USE_STATIC_CONTEXT=True` se mantiene como default para cerrar
Stage 1. Si no lo acepta, se debe revisar por cohorte antes de avanzar al
residual local.
