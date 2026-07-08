# MAC3_TEST

`MAC3_TEST` is the first release use case for GTS-RM.

This directory is intentionally not under `tutorials/`. It is the operational
case that drives the library extraction:

1. keep CP20 behavior frozen;
2. expose the validated global forecasting core through `gts_rm`;
3. produce a precise global model workflow;
4. derive tutorials only after the release path is stable.

## Release Objective

Build the first GTS-RM release around the existing CP20 global forecasting
pipeline, then iterate toward the GTRM manifesto:

- global representation via `history_embedding`;
- causal data contract;
- global supervised model first;
- local residual, quantiles, patching and SSL later through explicit flags.

## Current Dependency

The use case currently depends on the frozen CP20 bundle:

```text
liquidez_2026_cp20_prework_financial_gpt/
```

The package entry point is:

```python
import gts_rm
```

## CP22 Contract

CP21 locked this directory as the operational boundary. CP22 adds a stable
library facade over CP20. See `CONTRACT.md` and `manifest.json` for the
machine-readable and human-readable contracts.

The current directory layout is:

```text
configs/   versioned use-case configuration
data/      canonical input data location
artifacts/ model artifacts and persisted runs
reports/   evaluation and acceptance reports
runs/      run manifests and execution records
notebooks/ optional notebooks, not the source of truth
```

The public facade is:

```python
from gts_rm import data, models, training, evaluation, artifacts, config
```
