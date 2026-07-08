# Checkpoint 17 — Numerical Stability and HPO Recovery

## Problem corrected

A productive warm-up could finish a finite training epoch but overflow while
inverting `signed_log1p` predictions for raw-scale validation. The resulting
`inf / inf` made `raw_macro_smape = nan` and aborted the run after HPO had
already completed.

## Changes

- Numerically safe inverse transform with derived float64 bounds.
- Stable raw sMAPE and explicit clipping diagnostics.
- Rollback + learning-rate reduction when an epoch is non-finite.
- Finite-parameter gate after every optimizer step.
- HPO evidence committed to `GlobalManager` before productive warm-up.
- Optuna study persisted in a run-local SQLite database.
- Four global notebooks expose generic recovery controls.

No rule depends on `saldo`, `variacion`, weekdays, account identities or a
specific financial calendar.
