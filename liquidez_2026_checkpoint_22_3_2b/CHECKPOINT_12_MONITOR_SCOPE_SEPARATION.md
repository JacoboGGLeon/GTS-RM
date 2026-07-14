# Checkpoint 12 — Monitor Scope Separation

## Objective

Separate the two official monitoring workflows without changing any training
notebook or model implementation:

- `monitor_codigo_02.ipynb` evaluates only the four local `code_02` models;
- `monitor_codigo_03_FINANCIAL_GPT.ipynb` evaluates only the four global
  `code_03` models plus `NAIVE_LAST_VALUE`.

## Official candidate sets

### Local monitor

`monitor_codigo_02.ipynb` remains byte-identical and contains:

- `MLP_E_D`
- `MLP_VaE_D`
- `RNN_E_D`
- `RNNBi_E_D`

It does not discover global runs and does not add the naive baseline.

### Financial-GPT monitor

`monitor_codigo_03_FINANCIAL_GPT.ipynb` contains exactly:

- `GLOBAL_MLP_E_D`
- `GLOBAL_MLP_VaE_D`
- `GLOBAL_RNN_E_D`
- `GLOBAL_RNNBi_E_D`
- `NAIVE_LAST_VALUE`

It does not accept, discover or load local runs.

## Implementation

`financial_gpt_monitor.py` adds
`compare_global_financial_gpt_runs(...)`. It validates that exactly the four
registered global architectures are supplied, recalculates metrics on common
test dates per `cross_key_id`, ranks the four global candidates plus naive and
builds the final winner forecast.

The previous `compare_financial_gpt_runs(...)` API remains available only for
backward compatibility and extraordinary analysis; no official notebook calls
it.

## Non-regression

- Four local training notebooks unchanged.
- Four global training notebooks unchanged.
- `monitor_codigo_02.ipynb` SHA-256 unchanged.
- No model, dataset, HPO, curriculum, forecast or S3 code changed.

## Evidence

- 101 tests collected.
- 10 directly affected Checkpoint 11/12 tests pass.
- `compileall` passes.
- all notebooks are output-free.
- ZIP integrity passes.
