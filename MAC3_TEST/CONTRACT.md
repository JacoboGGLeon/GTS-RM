# CP25 - MAC3_TEST Data Contract Migration

`MAC3_TEST` is the release-first use case for GTS-RM. It is not a tutorial.
The CP20 bundle remains the implementation source while the operational data
contract moves into `MAC3_TEST/configs/data_contract.json` and is loaded through
`gts_rm.data`.

## Scope

CP21 defined the use-case boundary. CP22 added the library facade. CP23 added
smoke workflows for the four locked CP20 global architectures. CP24 migrated
runtime configs. CP25 migrates the data contract:

- keep CP20 implementation files in place;
- keep CP20 model and data behavior unchanged;
- expose the MAC3 data contract through `gts_rm.data`;
- keep `MAC3_TEST/configs/data_contract.json` as the use-case data schema source;
- validate global-long columns, model inputs, metadata fields and forbidden
  input fields against CP20 constants;
- align notebook configs with the same canonical panel and calendar URIs;
- avoid requiring real MAC3 data until a later ingestion checkpoint.

CP25 does not:

- load production MAC3 data;
- train a productive model;
- move CP20 modules;
- infer calendar frequency outside the provider temporal axis;
- add residual, quantile, patching or SSL behavior.

## Data Contract

```text
MAC3_TEST/configs/data_contract.json
```

The public loader is:

```python
from gts_rm import data

contract = data.load_mac3_data_contract()
contract.validate()
```

The contract pins:

- canonical global-long required columns from `GLOBAL_LONG_REQUIRED_COLUMNS`;
- model inputs: `y_context`, `x_history`, `x_future`, `x_static`;
- metadata-only identifiers and raw categories;
- forbidden model inputs: `cross_key_id`, `account_currency_id`, `divisa`,
  `tipo_serie`, `serie`;
- calendar date column: `fecha`;
- MAC3 exogenous columns: `month_sin`, `month_cos`, `is_month_end`;
- split unit: `account_currency_id`;
- future-known exogenous policy for direct multi-horizon forecasting.

## Migrated Configs

```text
MAC3_TEST/configs/base_cp20.json          CP20 locked contract summary
MAC3_TEST/configs/stage_cp20.json         FinancialGPTStageConfig flags
MAC3_TEST/configs/training_smoke.json     GlobalTrainingConfig smoke defaults
MAC3_TEST/configs/candidates_smoke.json   GlobalCandidateConfig per architecture
MAC3_TEST/configs/notebooks_mac3.json     GlobalNotebookConfig per architecture
MAC3_TEST/configs/data_contract.json      canonical MAC3 data schema contract
MAC3_TEST/configs/smoke_global_*.json     executable smoke configs
MAC3_TEST/configs/acceptance.json         release evidence gates
```

## Smoke Workflow Contract

The current workflow suite still uses synthetic tensors only. That is intentional:
CP25 validates the data contract boundary before introducing real ingestion.

## Library Facade Contract

The use case must import operational capabilities from `gts_rm`, not directly
from the CP20 bundle.

```text
gts_rm.config      config loading, feature flags and stage configuration
gts_rm.data        data contract, schema, scaler, split, dataset and temporal axis
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

## Migration Rule

Each migration step must leave CP20 tests passing and add a MAC3_TEST or
`gts_rm` test proving the new operational path works from the repository root.
