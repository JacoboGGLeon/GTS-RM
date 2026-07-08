# CP20 Baseline Lock

This document freezes the current CP20 bundle as the baseline for the GTS-RM library work.
Future phases should extend this contract through explicit checkpoints, not by changing
the baseline silently.

## Locked Commit

- Local baseline commit before this lock: `cf2af8a Initial GTS-RM bundle`
- Baseline bundle: `liquidez_2026_cp20_prework_financial_gpt`
- Validation date: 2026-07-08

## Frozen Runtime Contract

- `MODEL_INPUT_FIELDS` stays exactly:
  - `y_context`
  - `x_history`
  - `x_future`
  - `x_static`
- Forbidden model inputs stay metadata-only:
  - `cross_key_id`
  - `account_currency_id`
  - raw `divisa`
  - raw `tipo_serie`
  - `serie`
- Supported global architectures stay:
  - `mlp`
  - `mlp_vae`
  - `rnn`
  - `rnn_bi`
- Model output keeps `y_pred` as the final point forecast.
- `y_pred` keeps shape `[batch, horizon, 1]`.
- `extras["history_embedding"]` remains the canonical representation output.
- CP20-compatible flags preserve default behavior:
  - `use_causal_scaler=True`
  - `use_calendar_future=True`
  - `use_static_context=True`
  - `use_patch_tokenizer=False`
  - `use_local_residual_decoder=False`
  - `use_quantile_head=False`
  - `use_self_supervised_pretraining=False`
  - `use_auxiliary_autoencoder=True`

## Frozen Data and Evaluation Contract

- Long-format schema validation remains owned by `global_long_schema.py`.
- Temporal alignment remains causal and axis-based, not calendar-day guessing.
- Train/validation/test split keeps grouped identity semantics by account/currency.
- `ContextScaler` remains causal and reversible from context-only parameters.
- `GlobalWindowDataset` remains the canonical window generator.
- `GlobalBalancedSampler` remains the default training sampler.
- `robust_macro_mase` remains the primary HPO/objective metric.
- Monitor output remains one portable JSON artifact unless a later checkpoint
  explicitly changes the artifact schema.
- Local and S3 model persistence must continue validating checksums/digests.

## Validation Evidence

Commands run from the current Windows workspace:

```powershell
python -m compileall -q .
python -m pytest -q
```

Result:

```text
148 passed, 1 warning in 74.82s
```

The warning is the existing PyTorch non-writable NumPy array warning in
`global_monitoring.py`; it is not a CP20 regression.

## Change Policy

Any later phase that changes these items must:

1. Add a new checkpoint document.
2. Update or add targeted tests.
3. Preserve backward-compatible defaults where possible.
4. Bump artifact schema only when persisted outputs/checkpoints change.
5. Keep notebooks output-free and `execution_count` null when edited.
