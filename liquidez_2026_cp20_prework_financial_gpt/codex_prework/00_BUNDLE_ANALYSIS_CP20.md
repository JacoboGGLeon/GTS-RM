# CP20 Bundle Analysis — Financial-GPT Prework

Base bundle inspected:

`liquidez_2026_codes_checkpoint_20_coherent_objective_balanced_sampling_curriculum.zip`

## Executive summary

This bundle is not a raw prototype. It already contains a mature **global forecasting** pipeline:

- canonical long schema for Financial-GFM;
- train/validation/test identity split by `account_currency_id`, keeping `saldo` and `variacion` together;
- causal linear context scaling fitted only on `y_context`;
- calendar/exogenous alignment with train-only scaling;
- non-identifying static context (`tipo_serie`, `divisa`, bounded log scale, bounded series age);
- four global architectures: `mlp`, `mlp_vae`, `rnn`, `rnn_bi`;
- direct multi-horizon output `[batch, horizon, 1]`;
- optional auxiliary autoencoder regularizer on the latent space;
- robust causal MASE objective;
- hierarchical balanced sampler by series type, curriculum level, series and window;
- HPO with proxy screening + medium-fidelity finalist selection;
- curriculum-vs-shuffled control;
- one Financial-GPT monitor JSON.

The next plan should **not restart the pipeline**. It should add feature flags and new heads around the existing CP20 contract.

## Current runtime contract

`global_contracts.py` defines the model inputs:

```python
MODEL_INPUT_FIELDS = (
    "y_context",
    "x_history",
    "x_future",
    "x_static",
)
```

Meaning:

- `y_context`: scaled historical target window `[B, L, 1]`.
- `x_history`: historical exogenous/calendar features `[B, L, F]`.
- `x_future`: future known exogenous/calendar features `[B, H, F]`.
- `x_static`: non-identifying static/causal features `[B, S]`.

Forbidden as model inputs:

```python
cross_key_id
account_currency_id
divisa raw column
tipo_serie raw column
serie
```

Important nuance: `divisa` and `tipo_serie` are forbidden as raw strings in `forward`, but they currently enter through `StaticFeatureEncoder` as train-fitted one-hot vectors. This is acceptable because they are semantic, not direct identity keys.

## Current representation status

### Already implemented

| Planned component | CP20 status |
|---|---|
| `x2: window serie` | Implemented as `y_context`. |
| causal scaler | Implemented by `ContextScaler`, linear and reversible. |
| inverse scale | Implemented by `ContextScaler.inverse_transform*`. |
| calendar context/future | Implemented as `x_history` and `x_future` from exogenous/calendar frame. |
| series encoder | Implemented inside each architecture. |
| `history_embedding` | Exposed as `output["extras"]["history_embedding"]`. |
| global forecast decoder | Implemented as `y_pred`. |
| static context | Already implemented, currently mandatory via `x_static`. |
| auxiliary reconstruction | Implemented as optional `use_auxiliary_autoencoder`; this is an in-training regularizer, not full SSL pretraining. |
| HPO | Implemented with proxy + medium-fidelity selection. |
| balanced sampling | Implemented in `GlobalBalancedSampler`. |
| curriculum | Implemented in `global_curriculum.py`. |

### Missing / not yet implemented

| Planned component | CP20 status | Recommendation |
|---|---|---|
| `use_patch_tokenizer` | Missing | Add later. Do not start here. |
| `use_static_context` flag | Missing; static is always on | Add a flag but default should preserve CP20 behavior: `True`. |
| `use_local_residual_decoder` | Missing | This is the most important next model feature. |
| `delta_local` output | Missing | Add to `output["extras"]` when enabled. |
| `forecast_scaled = y_global + delta_local` | Missing | Preserve `y_pred` as final forecast. Add `y_global` and `delta_local` diagnostics. |
| quantile/probabilistic head | Missing | Add after local residual is validated. |
| true self-supervised pretraining | Missing | Existing auxiliary AE is related but not equivalent. |
| observed/context mask | Removed by CP19 design | Reintroduce only if the raw data truly has missing target windows. Make it a flag and keep default off to preserve CP20. |

## Important finding

The user’s desired architecture:

```text
x2: window serie → scaler + mask → optional patch tokenizer → series_encoder → z_series
x1: calendar context/future → calendar encoder → z_calendar
z = concat(z_series, z_calendar, optional static)
z → global decoder + local residual decoder
```

is already partially implemented, but CP20 combines the roles differently:

```text
y_context + x_history → architecture-specific encoder → history_embedding
x_future + x_static + future position → decoder → y_pred
```

So the first next checkpoint should be an **interface/flags checkpoint**, not a rewrite.

## Test status in this sandbox

Static compile passes:

```bash
python -m compileall -q .
```

Pytest collection failed in this sandbox only because `polars` is not installed. The bundle itself includes `CHECKPOINT_20_VALIDATION.txt`, which reports 148 tests passed in the original validation environment.

Required packages for Codex/local test environment are listed in `codex_prework/requirements-codex.txt`.

## Do not break these CP20 invariants

1. `cross_key_id` remains metadata only.
2. `account_currency_id` remains metadata/split unit only.
3. Train/unseen split must remain grouped by `account_currency_id`.
4. Scalers must be causal and train/context only.
5. `y_pred` must keep shape `[B, H, 1]`.
6. Existing four architectures must remain supported.
7. Default flags must reproduce CP20 behavior unless explicitly changed.
8. Monitor must still export one `financial_gpt_monitor.json`.
9. HPO objective remains `robust_macro_mase` unless the checkpoint explicitly introduces a compound objective.

