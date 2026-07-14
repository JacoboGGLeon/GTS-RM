# GTS-RM

GTS-RM is a Global Time-Series Representation Model for heterogeneous financial forecasting.
The current repository keeps the original CP20 bundle intact and exposes it
through a small installable package so the first release can move forward
without rewriting validated code.

## Current Status

- CP20 baseline is frozen in `liquidez_2026_cp20_prework_financial_gpt/BASELINE_CP20_LOCK.md`.
- The first use case is `MAC3_TEST`.
- The library scaffold lives in `src/gts_rm`.
- Legacy CP20 modules remain in `liquidez_2026_cp20_prework_financial_gpt`.
- CP22.3.2b and the P0 pre-architecture gate live in `liquidez_2026_checkpoint_22_3_2b`.

## Install

```powershell
python -m pip install -e ".[dev]"
```

## Validate

```powershell
python -m compileall -q .
python -m pytest -q
```

## Direction

This is release-first. Tutorials will come after the use case and library API
are stable enough to teach from.
