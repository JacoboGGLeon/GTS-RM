# CP27 - MAC3_TEST Acceptance Report

![API coverage](api_coverage.svg)

Generated: 2026-07-08 America/Mexico_City

## Verdict

Accepted.

The GTS-RM API facade was migrated correctly for `MAC3_TEST` while preserving the use-case behavior validated so far. CP20 remains the implementation source; `MAC3_TEST` now consumes stable `gts_rm` entrypoints for config, data contract, model construction and trainer construction.

## Current Status

- Repository branch: `main`
- Baseline: `CP20`
- Current checkpoint: `CP27`
- Use case: `MAC3_TEST`
- Migration mode: wrapper-first
- Productive training execution: deferred
- Real MAC3 data ingestion: deferred

## Evidence

| Check | Command | Result |
| --- | --- | --- |
| Smoke suite | `python -m MAC3_TEST.workflows.smoke_all_global_models` | PASS |
| Syntax compile | `python -m compileall -q .` | PASS |
| Full pytest | `python -m pytest -q` | PASS: 166 passed, 1 warning, 5 subtests passed |
| API coverage probe | focused `sys.settrace` probe over `src/gts_rm` public API calls | PASS: 32.04% |

Known warning:

```text
global_monitoring.py:354 UserWarning: NumPy array is not writable...
```

This warning is pre-existing and does not fail the suite.

## API Coverage Badge

The badge is generated from a focused API probe because `coverage.py` is not installed in the environment. The probe exercises public `gts_rm` facade calls and measures AST statement-line coverage over `src/gts_rm`.

- Tool: focused `sys.settrace` API probe
- Scope: `src/gts_rm` API facade lines exercised by CP27 public API probe
- Covered statements: 140
- Total statements: 437
- Coverage: 32.04%

Per-file API coverage:

| File | Coverage |
| --- | ---: |
| `src/gts_rm/config.py` | 60.16% |
| `src/gts_rm/data.py` | 36.36% |
| `src/gts_rm/models.py` | 36.36% |
| `src/gts_rm/training.py` | 54.55% |
| Other facade modules | 0.00% in this API probe |

## Migration Answer

Yes: we migrated the API surface correctly while keeping the current use-case functionality intact.

More precisely:

- CP24 migrated operational config loading into `gts_rm.config`.
- CP25 migrated the MAC3 data contract into `gts_rm.data`.
- CP26 migrated model and training construction into `gts_rm.models` and `gts_rm.training`.
- CP27 confirms those migrated entrypoints remain executable and the full CP20 suite still passes.

What has not happened yet:

- No productive model has been trained through `MAC3_TEST`.
- No real MAC3 data ingestion workflow has been introduced.
- Tutorials remain deferred until the release path is stable.

## Accepted Public Entrypoints

```python
from gts_rm import config, data, models, training

bundle = config.load_mac3_config_bundle()
contract = data.load_mac3_data_contract()
model = models.build_mac3_smoke_model("mlp")
trainer = training.build_mac3_trainer("mlp")
```
