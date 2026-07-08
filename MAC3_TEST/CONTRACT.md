# CP26 - MAC3_TEST Model/Training Facade Migration

`MAC3_TEST` is the release-first use case for GTS-RM. It is not a tutorial.
The CP20 bundle remains the implementation source while model and training
entrypoints move behind stable `gts_rm.models` and `gts_rm.training` helpers.

## Scope

CP21 defined the use-case boundary. CP22 added the library facade. CP23 added
smoke workflows for the four locked CP20 global architectures. CP24 migrated
runtime configs. CP25 migrated the data contract. CP26 migrates the model and
training facade:

- keep CP20 implementation files in place;
- keep CP20 model/training behavior unchanged;
- build MAC3 smoke models through `gts_rm.models` from migrated configs;
- load MAC3 candidate/training configs through `gts_rm.training`;
- build CP20 `GlobalTrainer` instances through `gts_rm.training`;
- avoid executing productive training until a later workflow checkpoint.

CP26 does not:

- train a productive model;
- load production MAC3 data;
- move CP20 modules;
- add residual, quantile, patching or SSL behavior.

## Public Model Facade

```python
from gts_rm import models

specs = models.mac3_model_specs()
model = models.build_mac3_smoke_model("mlp")
model = models.build_global_model_from_config(payload)
```

The facade must cover exactly `mlp`, `mlp_vae`, `rnn` and `rnn_bi`.

## Public Training Facade

```python
from gts_rm import training

candidates = training.load_mac3_candidates()
candidate = training.get_mac3_candidate("rnn")
trainer = training.build_mac3_trainer("rnn")
summary = training.mac3_training_facade_summary()
```

The trainer is CP20 `GlobalTrainer` configured from `MAC3_TEST/configs`, but CP26
only validates construction. Real training remains deferred.

## Data And Config Contracts

The CP24 config bundle and CP25 data contract remain active:

```python
from gts_rm import config, data

bundle = config.load_mac3_config_bundle()
contract = data.load_mac3_data_contract()
```

## Smoke Workflow Contract

The current workflow suite still uses synthetic tensors only. That is intentional:
CP26 validates model/trainer construction before introducing real ingestion and
training execution.

## Library Facade Contract

The use case must import operational capabilities from `gts_rm`, not directly
from the CP20 bundle.

```text
gts_rm.config      config loading, feature flags and stage configuration
gts_rm.data        data contract, schema, scaler, split, dataset and temporal axis
gts_rm.models      model specs and builders over CP20 global architectures
gts_rm.training    candidate loading and trainer builders over CP20 training APIs
gts_rm.evaluation  validation, monitoring and comparison APIs
gts_rm.artifacts   manager, local artifacts and S3 persistence APIs
```

The facade is wrapper-first. CP20 remains the implementation source until later
checkpoints migrate internals module by module.

## Migration Rule

Each migration step must leave CP20 tests passing and add a MAC3_TEST or
`gts_rm` test proving the new operational path works from the repository root.
