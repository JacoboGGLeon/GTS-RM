# Roadmap from CP20 to the New Financial-GPT Plan

This roadmap maps the requested stages to the existing CP20 bundle. The checkpoint numbering starts at 21 because CP20 is the current base.

## Stage 0 — Baselines / data contract

Most of Stage 0 already exists in CP20.

| Requested checkpoint | CP20 status | Next action |
|---|---|---|
| 0.1 temporal axis causal | Implemented via `temporal_axis.py`, `global_long_schema.py`, `GlobalNotebookDatasetFactory`. | Keep. Add docs/tests only if behavior changes. |
| 0.2 temporal + series-held-out split | Implemented via `GlobalSeriesSplit`, grouped by `account_currency_id`. | Keep. |
| 0.3 causal scaler + inverse scaler | Implemented via `ContextScaler`. | Keep. Optional future ablation: scaler type. |
| 0.4 window generator | Implemented via `GlobalWindowDataset`. | Keep. |
| 0.5 baselines | Implemented in monitor layer. | Keep. Only extend metrics if needed. |

### Checkpoint 21 — Feature flags and CP20 compatibility gate

Goal: introduce one feature flag contract without changing default behavior.

Files likely touched:

- `global_contracts.py`
- new `financial_gpt_flags.py` or equivalent
- `global_models.py`
- `global_training.py`
- four `code_03_GLOBAL_*` notebooks
- tests

Acceptance:

- default flags reproduce CP20 forward behavior;
- all four architectures still produce `[B,H,1]`;
- `use_static_context=False` can zero/drop `x_static` through a controlled path, but default is `True`;
- `use_patch_tokenizer`, `use_local_residual_decoder`, `use_quantile_head`, `use_self_supervised_pretraining` exist as config flags even if some are not active yet;
- no identifier is added to model inputs.

## Stage 1 — Global supervised training

Already implemented.

### Checkpoint 22 — Stage 1 contract documentation and regression tests

Goal: freeze CP20's global-only model as the baseline `M1`.

Acceptance:

- produces `y_pred` only as final forecast;
- exports `history_embedding`;
- robust_macro_mase remains training/HPO selection metric;
- monitor can compare all four global architectures.

This may be mostly tests and docs.

## Stage 2 — Global + Local Residual Training

This is the most important new feature.

### Checkpoint 23 — Local residual decoder in all four architectures

Goal: add optional residual head.

Files likely touched:

- `global_models.py`
- `global_training.py`
- `global_manager.py` if artifact schema needs metadata
- notebooks
- tests

Acceptance:

- `use_local_residual_decoder=False` is byte/behavior compatible with CP20 as much as possible;
- when true, output includes:
  - `y_pred = y_global + delta_local`;
  - `extras["y_global"]`;
  - `extras["delta_local"]`;
  - `extras["use_local_residual_decoder"]`;
- `delta_local` has shape `[B,H,1]`;
- `cross_key_id` is still not used in `forward`;
- residual regularization is included in `global_forecast_loss`.

### Checkpoint 24 — Residual diagnostics and M1 vs M2 comparison

Goal: prove the residual head improves individual series instead of only improving volume-weighted aggregate metrics.

Add diagnostics:

- mean absolute `delta_local`;
- ratio `abs(delta_local) / abs(y_global)`;
- per-series improvement vs M1;
- percentage of series improved;
- groups where residual helps/hurts.

Artifacts:

- `residual_diagnostics.csv/parquet`;
- monitor JSON fields for residual models;
- ablation table M1 vs M2.

## Stage 5 partial — HPO for Stage 1 and Stage 2

### Checkpoint 25 — Controlled HPO for global-only and residual models

Goal: allow HPO to tune only safe variables.

Add HPO search space for:

- `use_local_residual_decoder` fixed per experiment, not random mixed with everything;
- `local_hidden_dim`;
- `local_num_layers`;
- `local_dropout`;
- `local_residual_lambda`;
- `global_aux_alpha`.

Acceptance:

- HPO does not decide leakage-sensitive contracts;
- `robust_macro_mase` remains primary objective;
- per-series improvement is tracked as diagnostic.

## Stage 3 — Quantile/probabilistic head

### Checkpoint 26 — Quantile decoder and pinball loss

Goal: add optional quantile head without breaking point forecast.

Flags:

```python
use_quantile_head: bool
quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
quantile_weight: float
quantile_crossing_penalty: float
```

Acceptance:

- output includes `quantiles` only when enabled;
- point `y_pred` is still available;
- pinball loss is added;
- quantile crossing is penalized or prevented;
- monitor reports coverage and interval width.

## Stage 4 — Self-supervised pretraining

CP20 already has an auxiliary autoencoder regularizer. That is not full pretraining.

### Checkpoint 27 — Masked reconstruction dataset and pretraining loop

Goal: true encoder pretraining before supervised forecasting.

Acceptance:

- can train `series_encoder + reconstruction_head` without forecast decoder;
- saves `pretrained_series_encoder.pt`;
- fine-tuning can load encoder weights;
- comparison `without_ssl` vs `with_ssl` is persisted.

## Stage 5 final — Patch tokenizer, static ablation, SSL/quantile ablations

### Checkpoint 28 — Continuous patch tokenizer

Goal: optional continuous patch projection, not Chronos-style discrete tokenization.

Flags:

```python
use_patch_tokenizer: bool
patch_size: int
patch_stride: int
patch_dim: int
```

Acceptance:

- default `False` preserves CP20;
- when true, encoder receives patch embeddings;
- no value quantization/vocabulary.

### Checkpoint 29 — Static context ablation

Goal: make `x_static` truly optional.

Acceptance:

- default `True` preserves CP20;
- `False` either supplies zero-width static features safely or uses a zero-vector adapter without changing public model input fields;
- ablation compares static on/off;
- no `cross_key_id` embedding.

### Checkpoint 30 — Final ablation matrix

Required variants:

| Variant | Patch | Static | Local residual | Quantile | SSL |
|---|---:|---:|---:|---:|---:|
| M1 | False | True | False | False | False |
| M2 | False | True | True | False | False |
| M3 | True | True | True | False | False |
| M4 | True | False/True ablation | True | False | False |
| M5 | True | True | True | True | False |
| M6 | True | True | True | True | True |

Decision criteria:

- `robust_macro_mase`;
- `weighted_WMAPE` raw scale;
- P90 per-series error;
- `% series improved vs baseline`;
- stability/cost/simplicity.

