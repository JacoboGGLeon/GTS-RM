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

Status: in progress.

- Lock `MAC3_TEST` as a release-first case.
- Define inputs, outputs, configs and acceptance metrics.
- Validate the manifest from tests.

## Phase 4: Use-Case Workflow

Status: pending.

- Add a MAC3_TEST training/evaluation runbook.
- Define expected data inputs.
- Define release acceptance metrics.
- Produce a reproducible smoke workflow.

## Phase 5: Tutorials

Status: deferred.

Tutorials should be extracted from the working use case after the release path
is stable.
