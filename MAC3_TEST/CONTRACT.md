# CP23 - MAC3_TEST Smoke Workflow Contract

`MAC3_TEST` is the release-first use case for GTS-RM. It is not a tutorial.
The CP20 bundle remains the source of truth while the operational workflow
migrates into this directory.

## Scope

CP21 defined the case boundary. CP22 added the library facade. CP23 adds the
executable smoke workflow suite for the four locked CP20 global architectures:

- declare the use-case inputs, outputs and acceptance metrics;
- create stable directories for configs, data, artifacts, reports, runs and
  optional notebooks;
- keep all CP20 implementation files in place;
- expose the CP20 core through `gts_rm` wrappers;
- make the contract testable from the repository root;
- provide stable public modules: `gts_rm.config`, `gts_rm.data`,
  `gts_rm.models`, `gts_rm.training`, `gts_rm.evaluation` and
  `gts_rm.artifacts`;
- run one smoke workflow per architecture: `mlp`, `mlp_vae`, `rnn`, `rnn_bi`;
- run the aggregate suite through `MAC3_TEST.workflows.smoke_all_global_models`;
- write smoke evidence under `MAC3_TEST/reports` and `MAC3_TEST/runs`.

CP23 does not:

- move CP20 modules;
- change model behavior;
- train a new productive model;
- add residual, quantile, patching or SSL behavior.

## Smoke Workflow Contract

The CP23 workflow suite must:

- load architecture-specific configs from `MAC3_TEST/configs/smoke_global_*.json`;
- build each model through `gts_rm.models.build_global_model`;
- cover exactly `mlp`, `mlp_vae`, `rnn` and `rnn_bi`;
- use synthetic tensors only;
- validate `y_pred` shape and finiteness;
- validate `extras["history_embedding"]` exists and is finite;
- write one report under `reports/` per architecture;
- write one run record under `runs/` per architecture.

## Library Facade Contract

The use case must import operational capabilities from `gts_rm`, not directly
from the CP20 bundle.

```text
gts_rm.config      feature flags and stage configuration
gts_rm.data        schema, scaler, split, dataset and temporal axis
gts_rm.models      global model registry and builders
gts_rm.training    trainer, HPO and curriculum APIs
gts_rm.evaluation  validation, monitoring and comparison APIs
gts_rm.artifacts   manager, local artifacts and S3 persistence APIs
```

The facade is wrapper-first. CP20 remains the implementation source until later
checkpoints migrate internals module by module.

## Directory Contract

```text
MAC3_TEST/
├─ configs/
├─ data/
├─ artifacts/
├─ reports/
├─ runs/
├─ notebooks/
├─ workflows/
├─ CONTRACT.md
├─ README.md
├─ RELEASE_PLAN.md
└─ manifest.json
```

## Input Contract

The canonical panel must follow the CP20 global-long schema. Required columns
come from `global_contracts.GLOBAL_LONG_REQUIRED_COLUMNS`.

Calendar/exogenous features must be aligned causally through the CP20 temporal
axis and dataset factory. Future-known calendar features are allowed; future
target leakage is not.

## Output Contract

The use case produces:

- model artifacts under `MAC3_TEST/artifacts`;
- run records under `MAC3_TEST/runs`;
- evaluation reports under `MAC3_TEST/reports`.

Until a later checkpoint changes the artifact schema, persisted model behavior
continues to follow the CP20 manager/save/load contract.

## Model Contract

The CP20 lock remains active:

- `forward` inputs: `y_context`, `x_history`, `x_future`, `x_static`;
- output: `y_pred`;
- latent representation: `extras["history_embedding"]`;
- output shape: `[batch, horizon, 1]`;
- architectures: `mlp`, `mlp_vae`, `rnn`, `rnn_bi`;
- no `cross_key_id`, `account_currency_id`, raw `divisa`, raw `tipo_serie` or
  `serie` in `forward`.

## Acceptance Metrics

The first release gate tracks:

- `robust_macro_mase`;
- `raw_wmape`;
- `p90_series_error`;
- `series_improved_pct`.

The CP20 objective remains `robust_macro_mase` until a later checkpoint defines
a compound score.

## Migration Rule

Each migration step must leave CP20 tests passing and add a MAC3_TEST or
`gts_rm` test proving the new operational path works from the repository root.