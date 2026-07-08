# MAC3_TEST

`MAC3_TEST` is the first release use case for GTS-RM.

This directory is intentionally not under `tutorials/`. It is the operational
case that drives the library extraction:

1. keep CP20 behavior frozen;
2. expose the validated global forecasting core through `gts_rm`;
3. produce a precise global model workflow;
4. derive tutorials only after the release path is stable.

## CP27 Status

CP27 adds the acceptance report for the migrated API facade.

```text
MAC3_TEST/reports/CP27_ACCEPTANCE_REPORT.md
MAC3_TEST/reports/api_coverage.svg
```

Current verdict: accepted.

The API facade has been migrated correctly for the current release path while
preserving the existing CP20-backed use-case behavior. Real MAC3 data ingestion
and productive training are still intentionally deferred.

## Public Facade

```python
from gts_rm import data, models, training, evaluation, artifacts, config

bundle = config.load_mac3_config_bundle()
contract = data.load_mac3_data_contract()
model = models.build_mac3_smoke_model("mlp")
trainer = training.build_mac3_trainer("mlp")
```

## Smoke Suite

Run the full smoke suite from the repository root:

```powershell
python -m MAC3_TEST.workflows.smoke_all_global_models
```
