# Checkpoint 4 — HPO y entrenamiento global

## Objetivo

Implementar un único ciclo de entrenamiento compartido y un único estudio
Optuna por arquitectura global. Este checkpoint no agrega curriculum learning,
orquestación, notebooks ni persistencia en S3.

## Componentes nuevos

### `GlobalTrainingConfig`

Configura entrenamiento reproducible con:

- AdamW;
- MAE, Huber o MSE;
- gradient clipping;
- ReduceLROnPlateau;
- early stopping;
- semilla y selección explícita de dispositivo;
- número balanceado de muestras por época.

### `GlobalDatasetBundle`

Agrupa exclusivamente:

```text
train
validation_seen
validation_unseen
```

Valida que:

- las tres particiones compartan ventana, horizonte y calendario;
- `validation_seen` utilice identidades presentes en entrenamiento;
- `validation_unseen` sea completamente disjunto;
- el conjunto test unseen no participe en HPO.

La separación temporal concreta de `validation_seen` deberá construirse en la
capa de orquestación posterior.

### `GlobalTrainer`

Construye y entrena un solo modelo global:

```text
1 architecture
1 model_config
1 state_dict
N cross_key_id compartiendo pesos
```

Cada época utiliza `SeriesBalancedSampler`, que primero selecciona una serie y
luego una ventana. No existe `models[serie]` ni `best_params[serie]`.

El target de entrenamiento es `y_future` normalizado con estadísticas del
contexto histórico. En VAE se suma exactamente `weighted_kl`; no se duplica la
regularización KL.

### Validación macro

Las métricas se calculan primero por `cross_key_id` y después se promedian:

```text
macro_mae
macro_rmse
raw_macro_mae
raw_macro_rmse
raw_macro_wmape
```

También se reporta `micro_mae` como diagnóstico. El objetivo HPO es:

```text
mean(
    validation_seen.macro_mae,
    validation_unseen.macro_mae,
)
```

Por tanto, ni una serie larga ni una sola partición dominan la selección.

`cross_key_id` se utiliza únicamente para balancear ejemplos y agregar métricas;
nunca entra al `forward` del modelo.

### `GlobalHPOTrainer`

Ejecuta exactamente un estudio Optuna para una arquitectura. Cada trial sugiere:

```text
window_size
latent_dim
arquitectura interna
learning_rate
weight_decay
batch_size
loss
dropout
```

Al finalizar, reconstruye el mejor candidato y entrena un único modelo global
final con esa configuración. Los trials son candidatos temporales de HPO, no
modelos persistentes por serie.

## Arquitecturas soportadas

```text
mlp
mlp_vae
rnn
rnn_bi
```

Las cuatro usan el contrato directo multi-horizonte creado en Checkpoint 3.

## Fuera de alcance

No se implementó:

- orden por `difficulty_score`;
- curriculum learning;
- replay entre etapas;
- manager global;
- guardado/carga de artefactos;
- notebooks de ejecución;
- cambios en `code_02_*`.

## Gate

```bash
python -m unittest discover -s tests -p "test_checkpoint_*.py" -v
python -m compileall -q .
```

El siguiente checkpoint debe implementar únicamente curriculum learning con
replay sobre este entrenador compartido.
