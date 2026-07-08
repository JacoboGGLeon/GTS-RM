# CP27 - MAC3_TEST Acceptance Report

`MAC3_TEST` is the release-first use case for GTS-RM. It is not a tutorial.
CP27 records the first acceptance report for the migrated wrapper-first API.

## Scope

CP27 validates and documents the state reached after CP24-CP26:

- config facade migrated through `gts_rm.config`;
- data contract migrated through `gts_rm.data`;
- model construction migrated through `gts_rm.models`;
- trainer construction migrated through `gts_rm.training`;
- CP20 behavior preserved through the full pytest suite;
- acceptance evidence captured under `MAC3_TEST/reports`.

CP27 does not:

- train a productive model;
- ingest real MAC3 data;
- move CP20 implementation modules;
- add residual, quantile, patching or SSL behavior.

## Acceptance Artifacts

```text
MAC3_TEST/reports/CP27_ACCEPTANCE_REPORT.md
MAC3_TEST/reports/api_coverage.svg
```

The acceptance report records:

- smoke suite status;
- compileall status;
- full pytest status;
- API coverage badge generated from a focused public API probe;
- explicit answer on API migration and use-case behavior preservation.

## Verdict

Accepted.

We migrated the public API surface correctly while keeping current use-case
functionality intact. The implementation remains CP20-backed and wrapper-first.
Real data ingestion and productive training remain deferred.

## Public Entrypoints Accepted

```python
from gts_rm import config, data, models, training

bundle = config.load_mac3_config_bundle()
contract = data.load_mac3_data_contract()
model = models.build_mac3_smoke_model("mlp")
trainer = training.build_mac3_trainer("mlp")
```

## Migration Rule

Each migration step must leave CP20 tests passing and add a MAC3_TEST or
`gts_rm` test proving the new operational path works from the repository root.
