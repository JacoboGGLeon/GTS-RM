# MAC3_TEST Release Plan

## Phase 1: Baseline

Status: complete.

- CP20 contract locked.
- Full suite passing.
- Sentinel test added.

## Phase 2: Library Extraction

Status: complete.

- Add installable `gts_rm` package.
- Keep CP20 files in place.
- Provide stable imports for the current global model core.
- Add package-level tests.

## Phase 3: Use-Case Contract

Status: complete.

- Lock `MAC3_TEST` as a release-first case.
- Define inputs, outputs, configs and acceptance metrics.
- Validate the manifest from tests.

## Phase 4: Library Facade

Status: complete.

- Expose CP20 through stable `gts_rm.*` modules.
- Keep CP20 as the implementation source.
- Make MAC3_TEST use facade imports.

## Phase 5: Use-Case Workflow

Status: complete.

- Add MAC3_TEST smoke workflows for `mlp`, `mlp_vae`, `rnn` and `rnn_bi`.
- Add an aggregate smoke suite for all locked architectures.
- Define expected data inputs.
- Define release acceptance metrics.
- Produce reproducible smoke workflow evidence under `reports/` and `runs/`.

## Phase 6: Config Migration

Status: complete.

- Move stage, training, candidate, notebook and smoke defaults into
  `MAC3_TEST/configs`.
- Add `gts_rm.config.load_mac3_config_bundle`.
- Validate migrated configs from tests and smoke workflows.
- Keep CP20 behavior unchanged.

## Phase 7: Tutorials

Status: deferred.

Tutorials should be extracted from the working use case after the release path
is stable.
