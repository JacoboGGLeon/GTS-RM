# Checkpoint 21.3 — History Embedding Export & Diagnostics

Este checkpoint convierte `history_embedding` en un artefacto auditable de GTRM
Stage 1. No cambia entrenamiento, arquitectura ni loss.

## Entregables

- `gtrm_embedding_diagnostics.py`
  - `embedding_columns(...)`
  - `validate_history_embedding_frame(...)`
  - `build_history_embedding_diagnostics(...)`
  - `write_history_embedding_artifacts(...)`
- `tests/test_checkpoint_21_3_history_embedding_diagnostics.py`

## Artefactos generados

Al ejecutar `collect_history_embeddings(...)` y luego
`write_history_embedding_artifacts(...)`, se generan:

- `history_embeddings.csv`
- `history_embeddings.parquet` cuando el entorno tenga engine de Parquet
- `history_embeddings_schema.json`
- `history_embeddings_parquet_status.json`
- `history_embedding_diagnostics_summary.json`
- `history_embedding_dimension_report.csv`
- `history_embedding_by_series_summary.csv`
- `history_embedding_by_<cohort>.csv`

## Diagnósticos

El reporte mide:

- número de embeddings y dimensión latente;
- fracción de embeddings casi duplicados;
- dimensiones colapsadas por varianza casi cero;
- norma media/p50/p90/máxima del embedding;
- drift sucesivo del embedding por serie;
- resumen por cohortes como `tipo_serie`, `divisa`, `grupo` y `nivel_curriculum`.

## Regla de diseño

Los identificadores (`cross_key_id`, `account_currency_id`) pueden viajar en
metadata de exportación, pero siguen prohibidos como inputs del `forward`.
