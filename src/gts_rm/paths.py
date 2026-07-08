from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CP20_BUNDLE_ROOT = REPO_ROOT / "liquidez_2026_cp20_prework_financial_gpt"
MAC3_TEST_ROOT = REPO_ROOT / "MAC3_TEST"

__all__ = ["REPO_ROOT", "CP20_BUNDLE_ROOT", "MAC3_TEST_ROOT"]
