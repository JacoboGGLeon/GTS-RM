# Checkpoint 19 — Causal Representation and No Leakage

## Objective

Replace the aggressive target transformation with a minimal causal representation and ensure that every learned statistic is fitted only after the train/validation/test partition has been fixed.

## Implemented contract

### Target

For every window, using only `y_context`:

```text
scale = max(mean(abs(y_context)), mean(abs(diff(y_context))), 1.0)
y_scaled = y / scale
y_raw = y_scaled * scale
```

There is no `signed_log1p`, rolling median centering, or nonlinear inverse. The auxiliary autoencoder, when enabled, reconstructs `y_context_scaled`.

### Model inputs

```text
y_context
x_history
x_future
x_static
```

`context_mask` no longer enters the encoder because the canonical schema rejects null and non-finite targets.

`x_static` contains no account identifier:

```text
one_hot(tipo_serie)
one_hot(divisa)
log_scale_bounded
series_age_bounded
```

The category vocabulary is fitted on train and includes unknown buckets for unseen categories.

### Leakage barriers

The execution order is now:

```text
temporal alignment and eligibility
→ split by account_currency_id
→ seen temporal holdout
→ train-only difficulty
→ train-only static vocabulary
→ train-date exogenous normalization
→ window construction
```

Saldo and variación from the same account-currency identity are always assigned to the same seen/unseen partition.

Curriculum difficulty is calculated separately for each final `cross_key_id`, only from train targets. Easy series are no longer removed from the global dataset; the legacy threshold remains local to `code_02`.

Calendar binary features remain 0/1. Other numeric calendar features are standardized with pooled train dates only. The selected feature list, statistics, and calendar checksum are persisted.

## Data schema

The canonical global-long schema adds:

```text
divisa
series_age_step
```

Checkpoint 18 canonical datasets can be upgraded at load time when currency can be inferred. Existing model weights cannot be reused because the forward contract now includes `x_static` and the artifact schema is `1.3`.

## Persisted evidence

Each global run now writes:

```text
static_feature_contract.json
exogenous_contract.json
difficulty_train_only.parquet
scaler_contract.json
dataset_summary.json
```

## Unchanged scope

The checkpoint does not change:

- the four global architectures;
- direct 25-step forecasting;
- HPO, warm-up, curriculum/replay, and consolidation budgets;
- optional auxiliary autoencoder and VAE KL;
- MC-Dropout;
- `NUM_WORKERS = 0`;
- S3 persistence and Monitor 03.

## Required rerun

```text
1. Run code_01.ipynb to regenerate global_series_long.parquet.
2. Run all four code_03_GLOBAL_*.ipynb notebooks.
3. Run monitor_codigo_03_FINANCIAL_GPT.ipynb.
```

Do not mix Checkpoint 18 model states with Checkpoint 19 datasets or inference code.
