# Checkpoint 15 — Training and Evaluation Integrity

## Objetivo

Corregir los problemas observados en la primera corrida real de Financial-GPT sin volver a modelos por cuenta:

1. usar la misma métrica de selección durante HPO, warm-up, fine-tuning y scheduler;
2. estabilizar series de variación intermitentes;
3. evaluar el backtest sólo sobre test;
4. calcular bandas y outliers con referencia de train;
5. separar saldo y variación mediante dos cabezas, conservando un encoder global;
6. reportar cobertura y anchura de intervalos;
7. recuperar la configuración general explícita de los notebooks originales.

## Cambios

### Dos cabezas de tarea

Cada arquitectura global conserva un encoder compartido y expone:

- `saldo` head;
- `variacion` head.

`cross_key_id` y `account_currency_id` siguen fuera de `forward()`. `tipo_serie` no se convierte en embedding ni se entrega al encoder: sólo selecciona externamente la cabeza correspondiente para cada muestra.

### Transformación de variación

- `saldo`: escalamiento robusto lineal por ventana.
- `variacion`: `signed_log1p` reversible antes del escalamiento robusto.
- ventanas de variación constantes usan un piso de escala igual a `1.0` en el dominio logarítmico.

Todos los estadísticos continúan ajustándose únicamente con `y_context`.

### Métrica coherente

`GlobalTrainingConfig.selection_metric` controla:

- mejor checkpoint;
- early stopping;
- `ReduceLROnPlateau`;
- warm-up;
- fine-tuning;
- consolidación.

El valor predeterminado es `raw_macro_smape`, igual al HPO proxy.

### Pérdidas configurables

Los cuatro notebooks globales aceptan:

- `rmse`;
- `mae`;
- `mse`;
- `smape`;
- `wmape`;
- `log_cosh`;
- `huber`.

La pérdida de entrenamiento y la métrica de selección son contratos independientes.

### Backtest e incertidumbre

Las métricas por serie se calculan exclusivamente con `isTrain=False`.

Se añaden:

- `PICP`;
- `MPIW`;
- `Winkler`.

Las bandas de outliers de Backtest, Forecast y Backtest+Forecast se ajustan con observaciones reales de train. `df_outliers` también se reconstruye desde la referencia de train.

### Configuración de notebooks

Los cuatro `code_03_GLOBAL_*` exponen:

- `N_MONTE_CARLO`;
- `HPO_TRIALS`, `HPO_EPOCHS`, `HPO_WINDOWS_PER_SERIES`, `HPO_BATCH`;
- `WARM_EPOCHS`, `WARM_BATCH`;
- `FINE_EPOCHS`, `FINE_BATCH`;
- `LOSS_FUNCTION`;
- `SELECTION_METRIC`;
- `SHOW_PLOTS`, `PLOT_MAX_SERIES`.

`FINE_EPOCHS` es un presupuesto total que se reparte entre los niveles curriculares disponibles. HPO y warm-up usan batches independientes.

La configuración efectiva se persiste como `reports/execution_config.json` y también viaja en el manifiesto del modelo.

## Compatibilidad

Los cuatro notebooks locales y los dos monitores oficiales no fueron modificados.

El esquema de artefacto global sube de `1.0` a `1.1`, porque las arquitecturas ahora contienen dos cabezas. Los pesos globales de Checkpoint 14 no son cargables como Checkpoint 15 y deben reentrenarse. Los datasets, splits y artefactos locales siguen siendo compatibles.

## Validación realizada

- 47 pruebas focalizadas de contratos, datos, scaling, HPO, eje temporal e integridad: `passed`.
- Smoke de las cuatro arquitecturas con dos cabezas: `passed`.
- Las siete pérdidas producen escalares finitos: `passed`.
- Backtest test-only e interval metrics: `passed`.
- Cuatro notebooks globales válidos, compilables y sin outputs: `passed`.
- `compileall`: `passed` (permanece el SyntaxWarning histórico de documentación en `losses.py`).

La suite monolítica pesada no se completó en este runner por su límite continuo; las áreas modificadas se validaron mediante pruebas focalizadas y smoke tests.
