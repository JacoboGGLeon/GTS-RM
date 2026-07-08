# CP24 - MAC3_TEST Config Migration Contract

`MAC3_TEST` is the release-first use case for GTS-RM. It is not a tutorial.
The CP20 bundle remains the implementation source while operational
configuration moves into `MAC3_TEST/configs` and is loaded through `gts_rm.config`.

## Scope

CP21 defined the use-case boundary. CP22 added the library facade. CP23 added
smoke workflows for the four locked CP20 global architectures. CP24 makes
configuration a library-facing contract:

- keep CP20 implementation files in place;
- keep CP20 model behavior unchanged;
- expose config loading through `gts_rm.config`;
- keep `MAC3_TEST/configs` as the source of truth for use-case execution;
- validate stage flags, training config, candidate configs, notebook execution
  configs and smoke configs from repository-root tests;
- keep smoke workflows facade-only through `gts_rm.config` and `gts_rm.models`.

CP24 does not:

- move CP20 modules;
- train a productive model;
- require real MAC3 input data;
- add residual, quantile, patching or SSL behavior.

## Migrated Configs

```text
MAC3_TEST/configs/base_cp20.json          CP20 locked contract summary
MAC3_TEST/configs/stage_cp20.json         FinancialGPTStageConfig flags
MAC3_TEST/configs/training_smoke.json     GlobalTrainingConfig smoke defaults
MAC3_TEST/configs/candidates_smoke.json   GlobalCandidateConfig per architecture
MAC3_TEST/configs/notebooks_mac3.json     GlobalNotebookConfig per architecture
MAC3_TEST/configs/smoke_global_*.json     executable smoke configs
MAC3_TEST/configs/acceptance.json         release evidence gates
```

The public loader is:

```python
from gts_rm import config

bundle = config.load_mac3_config_bundle()
bundle.validate()
```

The bundle must cover exactly `mlp`, `mlp_vae`, `rnn` and `rnn_bi`.

## Smoke Workflow Contract

The CP24 workflow suite must:

- load smoke configs through `gts_rm.config`;
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
gts_rm.config      config loading, feature flags and stage configuration
gts_rm.data        schema, scaler, split, dataset and temporal axis
gts_rm.models      global model registry and builders
gts_rm.training    trainer, HPO and curriculum APIs
gts_rm.evaluation  validation, monitoring and comparison APIs
gts_rm.artifacts   manager, local artifacts and S3 persistence APIs
```

The facade is wrapper-first. CP20 remains the implementation source until later
checkpoints migrate internals module by module.

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
