# Checkpoint 8 — Seen/unseen monitoring, MC-Dropout and final gates

## Alcance ejecutado

Este checkpoint recupera el contrato analítico de los `code_02_*` sin volver al
entrenamiento de un modelo por serie. Se mantiene un único `state_dict` global
por arquitectura y se separan resultados únicamente para evaluación, auditoría
y visualización.

## Flujo 5/5

`code_03_GLOBAL_DRL.ipynb` ejecuta:

1. HPO global y warm-up.
2. Fine-tuning curricular con replay y consolidación.
3. Backtest rolling train/test con MC-Dropout.
4. Forecast futuro real con MC-Dropout.
5. Tres visualizaciones legacy por `cross_key_id`.

La sesión curricular conserva el mismo modelo y el mismo optimizador entre las
fases 1 y 2. Los hashes de estado verifican la continuidad de pesos.

## Contrato de resultados

```python
results = {
    "warm": manager._warmup_results,
    "fine": manager._finetune_results,
    "backtest": manager._backtest_results,
    "forecast": manager._future_results,
    "df_forecasts": manager._df_forecasts,
    "df_outliers": manager._df_outliers,
}
```

## Backtest y forecast

- El backtest agrega ventanas superpuestas por serie y fecha.
- `isTrain=True/False` conserva la separación visual train/test.
- MC-Dropout entrega media, intervalo, varianza predictiva y sesgo cuadrático.
- El forecast comienza después de la última observación disponible.
- Si el rango supera el horizonte directo, se encadenan bloques manteniendo la
  salida directa dentro de cada bloque.
- `cross_key_id`, cuenta, divisa y tipo de serie no entran al `forward()`.

## Outliers y visualización

Se conserva el cálculo jerárquico legacy 3σ → 2σ → 1σ y las tres figuras:

1. Backtest.
2. Forecast.
3. Backtest + Forecast.

Las funciones existentes de `Tools` se reutilizan para mantener apariencia,
bandas, histogramas, intervalos y marcadores.

## Monitor final

`monitor_codigo_03_GLOBAL_DRL.ipynb` y `global_monitor.py`:

- cargan dos o más runs globales terminados;
- comparan métricas seen/unseen;
- rankean arquitecturas por serie;
- eligen un ganador por `cross_key_id`;
- construyen el forecast ensemble usando solamente la arquitectura ganadora.

## Validación

- Checkpoints 0–4: **48 passed**.
- Checkpoints 5–8: **29 passed**.
- Total: **77 passed**.
- Smoke real de las tres visualizaciones con backend no interactivo: OK.
- Forecast futuro posterior al último dato: OK.
- Monitor y ensemble por ganador: OK.
- `compileall`: OK.
- Notebooks sin outputs persistidos: OK.
- ZIP integrity: OK.

Las pruebas se ejecutaron en dos grupos para evitar el límite continuo del
runner. No hubo fallos.
