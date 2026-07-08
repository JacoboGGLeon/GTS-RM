from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CP20 = ROOT / "liquidez_2026_cp20_prework_financial_gpt"

for path in (SRC, CP20):
    path_text = str(path)
    if path.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)
