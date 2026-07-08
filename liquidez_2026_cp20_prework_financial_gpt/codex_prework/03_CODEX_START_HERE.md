# Codex Start Here — CP20 to Financial-GPT v2

## Objective

Start from CP20 and implement the new architecture plan incrementally. Do not rewrite the bundle.

First target:

```text
Checkpoint 21 — Feature flags and compatibility gate
```

Then:

```text
Checkpoint 23 — Local residual decoder
```

## Setup commands

Install dependencies in the Codex/runtime environment:

```bash
python -m pip install -r codex_prework/requirements-codex.txt
```

Run static compile:

```bash
python -m compileall -q .
```

Run tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Focused tests during model changes:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_checkpoint_19_causal_representation_no_leakage.py \
  tests/test_checkpoint_20_coherent_objective_sampling_curriculum.py
```

## Non-negotiable constraints

- Do not add `cross_key_id`, `account_currency_id`, raw `divisa`, or raw `tipo_serie` to `forward`.
- Keep `MODEL_INPUT_FIELDS` stable unless there is a deliberate checkpoint-level migration.
- Keep `y_pred` as the final point forecast.
- Keep shape `[batch, horizon, 1]`.
- Default flags must preserve CP20 behavior.
- Artifact schema should bump only when outputs/checkpoints change.
- Add tests before modifying notebook code.

## Suggested first Codex prompt

```text
You are working in this repository, starting from Checkpoint 20.
Implement Checkpoint 21: introduce a central Financial-GPT feature flag/config contract without changing default runtime behavior.

Read:
- codex_prework/00_BUNDLE_ANALYSIS_CP20.md
- codex_prework/01_TARGET_ARCHITECTURE_FLAGS.md
- codex_prework/02_ROADMAP_CHECKPOINTS_FROM_CP20.md

Tasks:
1. Add a small `financial_gpt_flags.py` module with immutable dataclasses for feature flags and stage config.
2. Wire the flags into model/training configs only enough to persist and validate them, not to change behavior yet.
3. Default flags must reproduce CP20:
   - use_static_context=True
   - use_patch_tokenizer=False
   - use_local_residual_decoder=False
   - use_quantile_head=False
   - use_self_supervised_pretraining=False
   - use_auxiliary_autoencoder=True
4. Add tests that all four architectures still pass the existing forward contract and that forbidden identifiers remain excluded.
5. Update checkpoint docs and GLOBAL_MODEL_CHECKPOINTS.csv with Checkpoint 21.
6. Keep notebooks output-free and execution_count null.
7. Run compileall and focused pytest.
```

## Suggested second Codex prompt

```text
Implement Checkpoint 23: optional local residual decoder.

Requirements:
1. Add `use_local_residual_decoder` to model configs, default False.
2. When False, outputs must match the old CP20 contract.
3. When True:
   - produce `y_global` and `delta_local` with shape [B,H,1];
   - final `y_pred = y_global + delta_local`;
   - expose `extras["y_global"]`, `extras["delta_local"]`, and `extras["use_local_residual_decoder"]`;
   - keep `history_embedding` exposed;
   - do not use `cross_key_id` or account identity.
4. Add regularization in `global_forecast_loss`:
   - `local_residual_lambda * mean(abs(delta_local))`;
   - optional `global_aux_alpha * Huber(y, y_global)`.
5. Add tests for all four architectures with the flag on/off.
6. Add residual diagnostics scaffolding if possible, but do not modify monitor winner logic yet.
```

