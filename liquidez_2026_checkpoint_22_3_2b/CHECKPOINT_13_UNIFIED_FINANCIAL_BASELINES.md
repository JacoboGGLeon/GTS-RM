# Checkpoint 13 — Unified Financial Baselines

## Scope

Unify the benchmark contract of the two official monitors without mixing local
and global neural models.

- `monitor_codigo_02.ipynb`: four local models plus the common baselines.
- `monitor_codigo_03_FINANCIAL_GPT.ipynb`: four global models plus the same
  common baselines.

## Baselines

### `NAIVE_LAST_VALUE`

One-step backtest prediction uses the previous observed value. Future forecast
repeats the last observed value.

### `NAIVE_ZERO`

Available only when the series is identified as `variacion`. Backtest and future
forecast are zero. It is not admitted as a candidate for `saldo`.

### `SEASONAL_NAIVE_FINANCIAL`

Repeats the value observed `SEASONAL_PERIOD_DAYS` earlier. The default is seven
calendar days, matching the weekly financial cycle. When an exact seasonal
observation is unavailable, it falls back to the latest available historical
value. Multi-step future forecasts recursively repeat the learned seasonal
pattern.

## Fair comparison

For every `cross_key_id`, all model and baseline metrics are recomputed over the
exact intersection of test dates. `NAIVE_ZERO` is omitted from the candidate
set for non-variation series. Forecasts in the final ensemble come only from
the winning candidate.

## Official monitor scopes

```text
monitor_codigo_02.ipynb
  4 local models
  + NAIVE_LAST_VALUE
  + NAIVE_ZERO (variation only)
  + SEASONAL_NAIVE_FINANCIAL

monitor_codigo_03_FINANCIAL_GPT.ipynb
  4 global models
  + NAIVE_LAST_VALUE
  + NAIVE_ZERO (variation only)
  + SEASONAL_NAIVE_FINANCIAL
```

## Evidence

- 104 tests collected.
- 13 checkpoint 11–13 monitor tests passed.
- 39 checkpoint 0–3 inherited tests passed.
- Full monolithic suite reached the runner timeout without recording failures.
- Training notebooks and runtime modules outside the monitor scope are unchanged
  from Checkpoint 12.
- Python compile and notebook-clean gates pass.
