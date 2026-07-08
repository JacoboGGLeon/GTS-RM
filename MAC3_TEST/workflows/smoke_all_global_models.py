from __future__ import annotations

from ._global_smoke import run_all_global_smokes


def run_smoke_suite(*, output_root: str | None = None) -> dict:
    return run_all_global_smokes(output_root=output_root)


if __name__ == "__main__":
    results = run_smoke_suite()
    raise SystemExit(0 if all(report["ok"] for report in results.values()) else 1)
