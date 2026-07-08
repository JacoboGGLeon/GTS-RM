from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cp20_bundle_root() -> Path:
    return repo_root() / "liquidez_2026_cp20_prework_financial_gpt"


def ensure_cp20_import_path() -> Path:
    root = cp20_bundle_root()
    if not root.exists():
        raise RuntimeError(f"CP20 bundle root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
