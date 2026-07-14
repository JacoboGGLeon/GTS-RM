# Checkpoint 22.2.3 — Notebook Contract Hardening

## Objetivo

Limpiar y endurecer el notebook global antes de ampliar la arquitectura. Este
checkpoint **no cambia** encoders, decoders, losses, targets auxiliares,
sampling, HPO ni forecast. Su función es convertir el notebook en un consumidor
de contratos tipados y de una API pública ordenada.

## Problema corregido

Los cuatro notebooks globales construían configuraciones válidas, pero después:

- ejecutaban métodos privados del manager;
- leían atributos privados para backtest, forecast y outliers;
- repartían validaciones entre varias celdas;
- no tenían una máquina de estados que impidiera adelantar o repetir fases;
- no persistían un contrato completo y versionado del run.

## Implementación

### 1. Contrato Pydantic único

Se agregó `global_pipeline.py` con `GlobalNotebookRunContract`, un modelo
Pydantic estricto, inmutable y con `extra="forbid"`.

Valida en conjunto:

- `GlobalNotebookConfig`;
- `GTRMModelConfig`;
- `GlobalTrainingConfig`;
- `GlobalHPOConfig`;
- `GlobalCurriculumConfig`;
- igualdad de arquitectura;
- igualdad del model config anidado;
- coherencia entre loss del modelo y loss de entrenamiento;
- coherencia entre epochs proxy de entrenamiento y HPO;
- una sola métrica para HPO y selección productiva;
- flags de residual y auxiliary heads;
- `event_threshold` y `magnitude_transform`.

### 2. Requests Pydantic por fase

Se agregaron contratos separados:

- `HPOWarmupRequest`;
- `FineTuneRequest`;
- `BacktestRequest`;
- `ForecastRequest`.

`ForecastRequest` exige exactamente uno de estos modos:

- `start_date + end_date`; o
- `n_steps`.

### 3. Orden único de ejecución

`GlobalTrainingWorkflow` impone:

```text
HPO + warm-up
    → fine-tuning + consolidación
    → backtest
    → forecast
```

Una fase adelantada, repetida o posterior a la finalización genera error.

### 4. API pública del manager

Se expusieron métodos públicos:

- `run_hpo_and_warmup()`;
- `run_finetune()`;
- `run_backtest()`;
- `run_future_forecast()`.

Los métodos privados históricos siguen existiendo para compatibilidad interna,
pero ya no aparecen en los notebooks.

También se agregaron vistas públicas de sólo lectura:

- `backtest_results`;
- `future_results`;
- `forecast_frame`;
- `outliers_frame`.

### 5. Notebooks endurecidos

Los cuatro notebooks globales ahora:

- instalan `pydantic>=2.6` cuando se habilita instalación;
- construyen un `GlobalNotebookRunContract`;
- ejecutan un `GlobalTrainingWorkflow`;
- no invocan métodos privados;
- no leen atributos privados;
- persisten `notebook_run_contract.json`;
- persisten `workflow_snapshot.json`;
- mantienen outputs y execution counts vacíos.

## Artefactos modificados

- `global_pipeline.py`;
- `global_manager.py`;
- `code_03_GLOBAL_MLP_E_D.ipynb`;
- `code_03_GLOBAL_MLP_VaE_D.ipynb`;
- `code_03_GLOBAL_RNN_E_D.ipynb`;
- `code_03_GLOBAL_RNNBi_E_D.ipynb`;
- tests históricos ajustados al contrato público;
- `tests/test_checkpoint_22_2_3_notebook_contract_hardening.py`.

## Fuera de alcance

Este checkpoint no implementa todavía:

- encoders específicos por modalidad;
- refinamiento residual autoregresivo;
- scheduled sampling;
- rolling-origin evaluation nuevo;
- ampliación del espacio HPO;
- aumento del presupuesto de entrenamiento.

El siguiente checkpoint arquitectónico puede partir de una frontera de notebook
estable y auditable.
