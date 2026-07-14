# Checkpoint 21.1.1 — Warning-free losses docstrings

## Purpose

Remove the pre-existing `SyntaxWarning: invalid escape sequence` emitted by
`losses.py` during strict Python compilation.

## Change

The `log_cosh_loss` docstring no longer uses LaTeX-style backslash escapes inside
a normal Python string. The mathematical explanation was rewritten in plain text:

- before: `\(\log(\cosh(e))\)`
- after: `log(cosh(e))`

No functional code, public API, imports, or loss formulas were changed.

## Regression guard

Added:

```text
tests/test_checkpoint_21_1_1_warning_free_losses.py
```

The test compiles `losses.py` with `SyntaxWarning` treated as an error, so this
class of warning cannot return silently.

## Validation

```bash
python -W error::SyntaxWarning -m py_compile losses.py
python -W error::SyntaxWarning -m compileall -q .
python -m pytest -q tests/test_checkpoint_21_1_1_warning_free_losses.py
```
