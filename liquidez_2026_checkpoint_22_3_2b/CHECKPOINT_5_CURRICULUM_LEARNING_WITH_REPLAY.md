# Checkpoint 5 — Curriculum learning con replay

## Objetivo

Agregar warm-up y fine-tuning curricular al modelo global seleccionado por HPO,
sin crear modelos por serie y sin tocar todavía manager, persistencia, notebooks
o monitor final.

## Flujo implementado

```text
HPO global
    ↓
mejor configuración
    ↓
modelo nuevo construido una sola vez
    ↓
warm-up con el nivel curricular más sencillo
    ↓
fine-tuning por cada nivel posterior + replay de niveles previos
    ↓
consolidación balanceada con todos los niveles
    ↓
modelo global final
```

El modelo y el estado del optimizador continúan entre etapas. No se reinician
pesos. Al terminar cada fase se restauran el mejor `state_dict` y el estado del
optimizador según la validación macro `seen + unseen`.

## Componentes

### `GlobalCurriculumConfig`

Define:

- épocas de warm-up;
- épocas de fine-tuning por nivel;
- épocas de consolidación;
- fracción de replay;
- reducción del learning rate para fine-tuning y consolidación.

Los learning rates son monótonos: una etapa posterior nunca vuelve a elevar un
learning rate que el scheduler ya haya reducido.

### `CurriculumReplaySampler`

Para cada muestra:

1. elige pool actual o replay;
2. elige una serie uniformemente dentro del pool;
3. elige una ventana de esa serie.

Por tanto, ni una serie larga ni un nivel con muchas ventanas domina los
gradientes.

### `GlobalCurriculumTrainer`

Construye exactamente un modelo y ejecuta:

```text
warmup_level_<min>
finetune_level_<siguiente> + replay
...
consolidation_all_levels
```

El nivel curricular y `cross_key_id` sólo controlan sampling y métricas. No son
inputs del modelo.

### Evidencia de continuidad

Cada etapa registra:

- digest de pesos al inicio;
- digest de pesos al final;
- mejor época y score;
- muestras actuales y de replay;
- learning rate;
- validación seen/unseen.

El digest final de una etapa debe ser exactamente el digest inicial de la
siguiente.

### Integración con HPO

`fit_best_candidate_with_curriculum(...)` toma el ganador de Checkpoint 4 y lo
reentrena productivamente desde cero con warm-up, curriculum y replay. Los pesos
de los trials no se reutilizan.

## Cambios de datos

`GlobalWindowDataset` expone de forma read-only:

```text
series_curriculum_levels
series_difficulty_scores
```

También valida que dificultad, nivel curricular, grupo, tipo e identidad sean
constantes dentro de cada serie.

## Fuera de alcance

No se implementó:

- `GlobalManager`;
- guardado/carga de artefactos;
- S3;
- notebook parametrizable;
- monitor final;
- cambios en los cuatro `code_02_*` locales.

## Gate

```bash
python -m unittest discover -s tests -p "test_checkpoint_*.py" -v
python -m compileall -q .
```

El siguiente checkpoint debe implementar únicamente manager global y
persistencia.
