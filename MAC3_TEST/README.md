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
