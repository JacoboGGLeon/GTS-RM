"""Checkpoint 21.1.1 — warning-free legacy loss catalogue.

This regression test keeps docstring/math notation changes from reintroducing
invalid Python escape warnings in losses.py.
"""
from __future__ import annotations

import py_compile
import warnings
from pathlib import Path


def test_losses_py_compiles_with_syntax_warnings_as_errors() -> None:
    root = Path(__file__).resolve().parents[1]
    losses_path = root / "losses.py"

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        py_compile.compile(str(losses_path), doraise=True)
