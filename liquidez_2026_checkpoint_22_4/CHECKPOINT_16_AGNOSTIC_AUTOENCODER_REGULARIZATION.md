# Checkpoint 16 — Agnostic forecast head + auxiliary autoencoder regularization

## Objetivo

Eliminar las dos cabezas de forecasting especializadas por `tipo_serie` y recuperar un contrato completamente agnóstico:

```text
y_context + x_history + x_future + context_mask
                    ↓
              encoder global
                    ↓
           history_embedding
              ↙             ↘
 forecast head único     autoencoder head auxiliar
```

`cross_key_id`, `account_currency_id` y `tipo_serie` permanecen fuera de `forward()`. El vocabulario de `tipo_serie` queda abierto: saldo, variación, liquidez, contrato o cualquier serie no vacía usan el mismo contrato.

## Arquitectura

Las cuatro arquitecturas globales exponen una sola predicción:

```text
y_pred: [batch, horizon, 1]
```

Desde el mismo espacio latente, un decoder auxiliar reconstruye el target histórico normalizado:

```text
context_reconstruction: [batch, window_size, 1]
```

La reconstrucción usa únicamente posiciones observadas de `context_mask`.

## Preprocesamiento agnóstico

Todas las series usan el mismo transformador reversible por ventana:

```text
signed_log1p → mediana contextual → escala robusta contextual
```

`series_type` ya no decide la transformación ni el centro. Sólo permanece como metadata y trazabilidad.

## Función objetivo

```text
loss_total = loss_forecast
           + beta_ae * loss_reconstruction
           + beta_kl * loss_kl      # sólo MLP-VAE
```

El autoencoder es un regularizador de entrenamiento. Backtest, forecast y MC-Dropout consumen exclusivamente `y_pred`.

## HPO

Optuna ajusta para todas las arquitecturas:

```text
beta_ae
  rango logarítmico: 1e-5 .. 1.0

ae_hidden_size
  32 | 64 | 128 | 256

ae_num_layers
  1 .. 3
```

Para `mlp_vae`, la regularización variacional se mantiene separada:

```text
beta_kl: 1e-4 .. 1.0
```

## Persistencia

`model_config` conserva los hiperparámetros del autoencoder dentro del manifiesto y artefactos S3. El esquema cambia a:

```text
artifact_schema_version = 1.2
```

Los pesos de Checkpoint 15 no son compatibles porque desaparecen las cabezas `saldo`/`variacion` y se incorpora el decoder auxiliar.

## Validación

```text
Checkpoints 0–8:   77 passed
Checkpoints 9–16:  47 passed
Total:             124 passed
compileall:        passed
```

Se comprobó:

- una sola cabeza de forecasting en MLP, MLP-VAE, RNN y RNNBi;
- vocabulario abierto de tipos de serie y escalamiento sin ramas por tipo;
- ausencia de `task_predictions`;
- reconstrucción causal del contexto desde el espacio latente;
- gradiente del autoencoder hacia el encoder compartido;
- `beta_ae=0` desactiva la penalización sin cambiar el contrato;
- HPO de importancia y capacidad del decoder auxiliar;
- separación de `beta_ae` y `beta_kl` en MLP-VAE;
- compatibilidad con training, curriculum, manager, MC-Dropout y S3.
