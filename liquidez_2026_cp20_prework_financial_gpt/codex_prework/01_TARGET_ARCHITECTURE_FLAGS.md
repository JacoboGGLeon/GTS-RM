# Target Architecture with Feature Flags

The new Financial-GPT roadmap should be implemented through feature flags, not hard forks.

## Target conceptual graph

```text
x2: window serie
   ↓
causal scaler + mask
   ↓
continuous patch/tokenizer optional
   ↓
series_encoder
   ↓
history_embedding z_series

x1: calendario contexto/futuro
   ↓
calendar_projection / calendar_encoder
   ↓
calendar_embedding z_calendar

z = concat(z_series, z_calendar, static_context optional)

z
 ├── global_forecast_decoder → y_global
 └── local_residual_decoder → delta_local

forecast_scaled = y_global + delta_local
forecast = inverse_scale(forecast_scaled)
```

## CP20-compatible flag defaults

These defaults must preserve CP20 behavior:

```python
FEATURE_FLAGS = {
    "use_causal_scaler": True,
    "use_observed_mask": False,
    "use_context_mask": False,
    "use_patch_tokenizer": False,
    "use_calendar_encoder": True,
    "use_calendar_future": True,
    "use_static_context": True,
    "use_local_residual_decoder": False,
    "use_quantile_head": False,
    "use_self_supervised_pretraining": False,
    "use_auxiliary_autoencoder": True,
}
```

Why `use_static_context=True` by default?

CP20 already uses `x_static` with non-identifying semantic/causal features. Turning it off by default would silently change the trained contract and break existing artifacts/notebooks. The ablation can later set it to `False`.

Why masks default to `False`?

CP19/CP20 explicitly removed `context_mask` because targets are required finite after canonicalization. Reintroduce masks only as a controlled feature flag after validating missingness semantics.

## Output contract after local residual checkpoint

When `use_local_residual_decoder=False`:

```python
output = {
    "y_pred": y_global,
    "extras": {
        "history_embedding": z,
        "use_local_residual_decoder": False,
    },
}
```

When `use_local_residual_decoder=True`:

```python
output = {
    "y_pred": y_global + delta_local,
    "extras": {
        "history_embedding": z,
        "y_global": y_global,
        "delta_local": delta_local,
        "use_local_residual_decoder": True,
    },
    "losses": {
        "weighted_local_residual": lambda_local * mean_abs(delta_local),
        "weighted_global_aux": alpha_global * Huber(y, y_global),  # computed in training if target is available
    }
}
```

Important: `y_pred` remains the final forecast so existing evaluation and monitor code keep working.

## Loss strategy

Stage 1 global only:

```text
loss = Huber(y_scaled, y_pred)
```

Stage 2 local residual:

```text
y_pred = y_global + delta_local
loss = Huber(y_scaled, y_pred)
     + lambda_local * mean(abs(delta_local))
     + alpha_global * Huber(y_scaled, y_global)
```

In implementation, it may be cleaner to compute `global_aux` inside `global_forecast_loss` because the target tensor is available there.

