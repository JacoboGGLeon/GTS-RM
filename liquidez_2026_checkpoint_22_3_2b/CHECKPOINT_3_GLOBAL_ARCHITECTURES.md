# Checkpoint 3 — Global DRL architectures

## Scope

This checkpoint adds four global forecasting architectures behind one strict
model contract. It does not add HPO, optimizers, training loops, curriculum
learning, orchestration, notebooks, or persistence.

## Contract

Every architecture implements:

```python
forward(
    y_context,
    x_history,
    x_future,
    context_mask,
) -> {
    "y_pred": Tensor[batch, horizon, 1],
    "extras": {"history_embedding": Tensor[batch, latent_dim]},
}
```

`cross_key_id`, `account_currency_id`, and `tipo_serie` remain dataset metadata
and cannot enter the constructor or `forward` contract.

## Architectures

- `GlobalMLPEncoderDecoder`: flattened historical encoder plus shared direct
  future decoder.
- `GlobalMLPVAEEncoderDecoder`: variational historical representation with KL
  output; deterministic mean representation during evaluation.
- `GlobalRNNEncoderDecoder`: unidirectional GRU history encoder and causal GRU
  future decoder.
- `GlobalRNNBiEncoderDecoder`: bidirectional history encoder and the same causal,
  unidirectional future decoder.

All models use direct multi-horizon forecasting. No predicted target is fed back
as a future lag in this checkpoint.

## Causality and masking

- The encoder consumes only historical targets and historical exogenous data.
- The decoder consumes only known-future exogenous data plus a deterministic
  horizon position.
- Masked target values are zeroed before encoding and cannot alter predictions.
- Bidirectionality is allowed only inside the fully observed historical window.

## Compatibility

The existing local `models.py`, `Scientist`, `Manager`, and `code_02_*`
notebooks remain untouched and continue to represent the model-per-series
baseline.
