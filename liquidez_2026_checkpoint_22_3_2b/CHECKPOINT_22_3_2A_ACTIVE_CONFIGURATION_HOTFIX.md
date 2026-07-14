# Checkpoint 22.3.2a — Active Configuration Hotfix

## Objetivo

Limpiar la superficie activa de los cuatro notebooks globales sin modificar la arquitectura, el HPO, el entrenamiento pooled ni la inferencia.

## Cambios

- Se retiró `GTRM_STAGE` de la celda de parámetros: Stage 2 es una propiedad interna del contrato vigente, no una decisión del usuario.
- Se corrigió la referencia circular `active_config.budget.hpo_timeout_seconds`.
- `HPO_TIMEOUT_SECONDS` queda conectado directamente a `TrainingBudgetConfig`.
- Cada bloque Pydantic se construye y valida por separado antes de ensamblar `GlobalActiveConfiguration`.
- Se mantienen únicamente controles con efecto real o ablations implementadas.
- No se cambian modelos, losses, datasets, sampling, presupuestos ni outputs.

## Flujo de configuración

```text
variables visibles
  -> contratos Pydantic independientes
  -> GlobalActiveConfiguration
  -> contratos internos de entrenamiento/HPO
```
