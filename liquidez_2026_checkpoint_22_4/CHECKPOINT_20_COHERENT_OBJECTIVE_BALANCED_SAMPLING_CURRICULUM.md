# Checkpoint 20 — Coherent Objective, Balanced Sampling and Curriculum Control

## Goal

Align the complete Financial-GFM training methodology without changing the four model architectures:

```text
training loss
→ HPO screening
→ medium-fidelity finalist selection
→ early stopping
→ productive training
→ monitor winner selection
```

The primary selection criterion is now one causal, non-redundant metric:

```text
robust_macro_mase
```

At monitor level the equivalent raw-scale metric is `MASE`, calculated with a denominator fitted only from observations before the comparison period.

## Implemented changes

### 1. Causal robust MASE

Each series receives a fixed denominator fitted from its pre-holdout history:

```text
max(
    mean absolute first difference,
    1% of mean absolute level,
    1e-6
)
```

The level floor prevents constant and near-constant financial series from producing an undefined or explosive MASE.

Evidence is persisted in:

```text
mase_scale_causal.parquet
```

### 2. One selection objective

The default selection metric is:

```python
SELECTION_METRIC = "robust_macro_mase"
```

It is shared by:

- Optuna objective;
- pruning reports;
- early stopping;
- learning-rate scheduling;
- best-checkpoint selection;
- productive monitoring.

MAE, RMSE, sMAPE, WMAPE, EVS and R² remain diagnostic metrics. They no longer vote repeatedly through a redundant rank sum.

### 3. Balanced global sampler

Training samples are drawn hierarchically:

```text
series type
→ curriculum level
→ cross_key_id
→ window
```

When both types exist, `saldo` and `variacion` receive the same expected sampling share. Long series, populous curriculum levels and series with many windows cannot dominate solely because they generate more rows.

Replay preserves its configured current/replay proportion and applies the same balancing contract inside each pool.

### 4. HPO in two fidelities

The HPO flow is now:

```text
150 proxy trials with Hyperband
→ top 5 completed trials
→ 8-epoch medium-fidelity comparison
→ selected candidate
→ new productive model trained from scratch
```

Default validation coverage increased from one to three windows per series.

The Optuna study records:

- proxy ranking;
- selected finalist trial;
- medium-fidelity scores;
- objective metric;
- finalist budget.

### 5. Controlled curriculum ablation

`GlobalCurriculumConfig` now supports:

```python
training_order = "curriculum"  # or "shuffled"
```

Both orders preserve:

- total epochs;
- samples per epoch;
- batch size;
- optimizer-step budget;
- seed;
- learning-rate factors;
- datasets and validation partitions.

The shuffled control mixes all available levels at every stage and disables replay. The helper `compare_curriculum_vs_shuffled` performs the paired comparison.

### 6. Monitor selection

`financial_gpt_monitor.json` remains the single monitor document. Winner selection now exposes:

```text
selection_metric = MASE
selection_score
selection_rank
```

The previous aggregate `rank_score` is removed from the Financial-GPT monitor because MAE/WMAPE and RMSE/R² produced duplicated votes.

## Productive notebook defaults

```python
HPO_VALIDATION_WINDOWS_PER_SERIES = 3
HPO_FINALISTS = 5
HPO_FIDELITY_EPOCHS = 8
HPO_FIDELITY_WINDOWS_PER_SERIES = 8
SELECTION_METRIC = "robust_macro_mase"
TRAINING_ORDER = "curriculum"
```

The four global candidates remain mandatory:

```text
GLOBAL_MLP_E_D
GLOBAL_MLP_VaE_D
GLOBAL_RNN_E_D
GLOBAL_RNNBi_E_D
```

No attention mechanism, local adapter, account embedding or architectural replacement was introduced.

## Compatibility

Artifact schema:

```text
1.4
```

Checkpoint 19 weights remain structurally loadable only when their model dimensions and static contracts match, but productive reruns are required because the HPO objective, sampler and winner criterion changed.

## Execution order

```text
code_01.ipynb
→ four code_03_GLOBAL notebooks
→ monitor_codigo_03_FINANCIAL_GPT.ipynb
```

For the curriculum ablation, execute matched runs with the same configuration and seed:

```text
TRAINING_ORDER = "curriculum"
TRAINING_ORDER = "shuffled"
```

Compare their `robust_macro_mase` before fixing the productive order.
