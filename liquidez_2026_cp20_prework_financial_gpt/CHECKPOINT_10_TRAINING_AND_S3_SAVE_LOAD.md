# Checkpoint 10 — Productive training and atomic S3 save/load

## Scope

This checkpoint keeps the productive Financial-GPT path established in Checkpoint 9:

```text
fast proxy HPO
→ new model with winning configuration
→ full warm-up
→ curriculum fine-tuning with replay
→ consolidation
```

It adds verified persistence under:

```text
s3://ada-us-east-1-sbx-live-mx-m6hn-data/users/mi31883/financial_gpt/
```

No local `code_02_*` notebook was modified. The four explicit global notebooks remain reserved for Checkpoint 11.

## S3 layout

```text
financial_gpt/
└── <architecture>/
    ├── runs/
    │   └── <run_id>/
    │       ├── model/
    │       │   ├── manifest.json
    │       │   ├── model_state.pt
    │       │   ├── metrics.json
    │       │   ├── history.json
    │       │   ├── hpo_summary.json
    │       │   └── split_manifest.json
    │       ├── evidence/
    │       │   ├── run_summary.json
    │       │   ├── best_candidate.json
    │       │   ├── split_manifest.json
    │       │   ├── curriculum_history.json
    │       │   ├── training_history.parquet
    │       │   └── metrics.parquet
    │       ├── reports/                 # optional notebook reports
    │       ├── artifact_checksums.json
    │       └── _SUCCESS
    └── latest.json
```

## Atomicity contract

S3 has no atomic directory rename. The run therefore uses a commit marker:

1. upload model, evidence and optional reports;
2. upload `artifact_checksums.json`;
3. verify the remote size of every object;
4. write `_SUCCESS`;
5. update `<architecture>/latest.json`.

A loader rejects a run when:

- `_SUCCESS` is absent;
- the checksum manifest was altered;
- any downloaded model/evidence file changed size or SHA-256;
- the local model manifest or `state_dict` digest is inconsistent.

A committed `run_id` is immutable.

## API

```python
run_uri = manager.save_model_s3(
    "s3://ada-us-east-1-sbx-live-mx-m6hn-data/users/mi31883/financial_gpt",
    run_id=run_name,
    reports_dir=reports_dir,
)

loaded = GlobalManager.load_model_s3(
    run_uri,
    map_location="cpu",
)

latest = GlobalManager.load_latest_model_s3(
    ARCHITECTURE,
    map_location="cpu",
)
```

`load_model_s3` downloads model and compact evidence, but only checks the remote sizes of potentially large notebook reports. This keeps inference startup efficient while preserving run integrity.

## Expected gain

- no retraining is needed for inference;
- interrupted uploads cannot be loaded as valid runs;
- exact model identity is checked with SHA-256 and `state_dict` digest;
- every architecture has a stable `latest.json` pointer;
- model, split, HPO, curriculum and metrics remain auditable together;
- all artifacts stay under the requested user S3 prefix.

## Validation

- 91 tests passed across checkpoints 0–10;
- atomic upload order verified with an in-memory S3 client;
- exact save/load and `latest` roundtrip verified;
- missing `_SUCCESS` rejected;
- tampered `model_state.pt` rejected;
- committed `run_id` overwrite rejected;
- notebook uses the requested S3 root and validates a post-upload load;
- the four local notebooks remain present and unchanged in scope.
