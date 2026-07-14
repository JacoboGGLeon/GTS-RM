# Checkpoint 9 — Original Methodology Alignment

## Objetivo

Recuperar la intención computacional del pipeline local: explorar muchas
configuraciones con un warm-up barato y reservar el cómputo pesado para el
warm-up/fine-tuning productivo del candidato ganador.

## Cambios

- HPO proxy de 3 epochs por trial.
- 4 ventanas balanceadas por serie y epoch.
- 1 origen reciente de validación por serie seen/unseen.
- `HyperbandPruner` con `trial.report()` y `trial.should_prune()`.
- Cache de datasets por `window_size` dentro del estudio.
- Sin refit duplicado al terminar Optuna: el resultado del mejor trial se usa
  sólo como evidencia; el curriculum crea un modelo productivo nuevo.
- `batch_size`, `loss=Huber` y `optimizer=AdamW` quedan fuera del espacio HPO.
- Objetivo Optuna: macro sMAPE en escala original, 50% seen + 50% unseen.
- Escalamiento contextual robusto y reversible calculado sólo con `y_context`:
  IQR, MAD, cambio absoluto medio y piso relativo a la magnitud.
- Para `variacion`, el centro es cero; para `saldo`, la mediana del contexto.
- Manifiesto explícito de elegibilidad por `cross_key_id`.

## Salidas nuevas del notebook

- `reports/hpo_trials.parquet`
- `reports/best_candidate.json`
- `reports/scaler_contract.json`
- `reports/eligibility_manifest.parquet`

## No incluido

- Save/load directo en S3: Checkpoint 10.
- Cuatro notebooks globales separados y monitor local-vs-global: Checkpoint 11.
- Los cuatro notebooks locales `code_02_*` permanecen intactos.

## Validación

- 84 pruebas pasaron.
- `compileall`: OK.
- 9 notebooks sin outputs persistidos.
- Los cuatro notebooks locales permanecen byte a byte idénticos al Checkpoint 8.
