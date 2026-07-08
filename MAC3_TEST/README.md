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

## CP25 Contract

CP21 locked this directory as the operational boundary. CP22 added a stable
library facade over CP20. CP23 added executable smoke workflows for the four
locked global model architectures. CP24 migrated operational configuration into
`MAC3_TEST/configs`. CP25 migrates the data contract into
`MAC3_TEST/configs/data_contract.json` and validates it through `gts_rm.data`.

The public facade is:

```python
from gts_rm import data, models, training, evaluation, artifacts, config
```

Load the migrated config bundle with:

```python
from gts_rm import config

bundle = config.load_mac3_config_bundle()
```

Load the data contract with:

```python
from gts_rm import data

contract = data.load_mac3_data_contract()
```

## Smoke Suite

Run the full smoke suite from the repository root:

```powershell
python -m MAC3_TEST.workflows.smoke_all_global_models
```

Each workflow loads config through `gts_rm.config`, builds the configured global
model through `gts_rm.models`, runs a synthetic forward pass, and writes JSON
evidence under `reports/` and `runs/`.
