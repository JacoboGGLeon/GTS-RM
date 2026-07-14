# Checkpoint 18 — Optional auxiliary autoencoder and single monitor JSON

## Scope

This checkpoint makes the auxiliary context-reconstruction head optional in all four Financial-GPT architectures and consolidates monitor 03 output into one portable JSON document.

## Model configuration

Every global notebook exposes:

```python
USE_AUXILIARY_AUTOENCODER = True
```

The flag is passed through `GlobalTrainingConfig` into every HPO candidate.

- `True`: HPO tunes `beta_ae`, `ae_hidden_size`, and `ae_num_layers`; the reconstruction head and weighted reconstruction loss remain active.
- `False`: the head is not instantiated, the reconstruction loss is absent, and those HPO parameters are not sampled.
- For `mlp_vae`, disabling the auxiliary autoencoder does not disable the VAE KL term.

The default remains `True`, preserving compatibility with checkpoint 17 artifacts and schema 1.2.

## Monitor output

`FinancialGPTMonitorResult.write(...)` now creates exactly one file:

```text
financial_gpt_monitor.json
```

It contains:

- run inventory;
- comparison coverage;
- metrics by series;
- winners by series;
- winner counts;
- final ensemble forecast;
- compact summary and schema metadata.

No monitor parquet files are emitted by this method.

## Notebook defaults

The four global notebooks were synchronized with the supplied productive configurations. RNNBi retains its larger batch/sample budget; the other three architectures use the 256/16384 budget supplied by the user.

## Validation

Focused validation completed:

```text
30 passed
```

Coverage includes checkpoints 11–13, 16–18, all four architectures with the autoencoder enabled and disabled, VAE KL independence, HPO search-space gating, and single-file JSON monitor persistence.
