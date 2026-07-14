# Checkpoint 11 — Four global notebooks and local/global monitor

## Scope

This checkpoint materializes four explicit Financial-GPT notebooks while
preserving the four local `code_02` baselines byte for byte.

### Local baselines

- `code_02_MLP_E_D.ipynb`
- `code_02_MLP_VaE_D.ipynb`
- `code_02_RNN_E_D.ipynb`
- `code_02_RNNBi_E_D.ipynb`

### Global models

- `code_03_GLOBAL_MLP_E_D.ipynb`
- `code_03_GLOBAL_MLP_VaE_D.ipynb`
- `code_03_GLOBAL_RNN_E_D.ipynb`
- `code_03_GLOBAL_RNNBi_E_D.ipynb`

Each global notebook fixes exactly one architecture but preserves the same
five-step contract: HPO/warm-up, curriculum fine-tuning, MC-Dropout backtest,
future forecast and per-series visualization. All four save and load under the
required Financial-GPT S3 root.

## Final monitor

`monitor_codigo_03_FINANCIAL_GPT.ipynb` compares:

1. four local runs;
2. four global runs;
3. a rolling last-value naive baseline.

`financial_gpt_monitor.py` reloads raw backtest predictions and recalculates all
metrics on the exact intersection of test dates available for every candidate
of each `cross_key_id`. This prevents local/global rankings from using different
periods or relying on incompatible preaggregated metrics.

The final outputs are:

- `run_inventory.parquet`
- `comparison_coverage.parquet`
- `metrics_by_series.parquet`
- `winners_by_series.parquet`
- `winner_counts.parquet`
- `ensemble_forecast.parquet`

## Invariants

- Local notebooks are unchanged.
- Global model weights remain shared; only outputs are separated by series.
- `cross_key_id` is never a model input.
- A winner is selected independently for each `cross_key_id`.
- The naive baseline can win; neural candidates are not forced into the final
  ensemble when they do not beat persistence.
