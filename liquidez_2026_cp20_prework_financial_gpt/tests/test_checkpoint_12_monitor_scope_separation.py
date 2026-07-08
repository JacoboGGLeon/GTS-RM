from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import polars as pl

from financial_gpt_monitor import (
    GLOBAL_ARCHITECTURES,
    NAIVE_CANDIDATE_ID,
    compare_global_financial_gpt_runs,
)


ROOT = Path(__file__).resolve().parents[1]

def _backtest_rows(candidate_id: str) -> pd.DataFrame:
    actual_by_series = {
        "S1": [1, 2, 3, 4, 5, 6],
        "S2": [10, 11, 12, 13, 14, 15],
        "S3": [20, 20, 20, 20, 20, 20],
    }
    rows = []
    for series_id, values in actual_by_series.items():
        for offset, actual in enumerate(values):
            is_train = offset < 3
            if is_train:
                prediction = float(actual)
            elif candidate_id == "GLOBAL_MLP_E_D" and series_id == "S1":
                prediction = float(actual)
            elif candidate_id == "GLOBAL_RNN_E_D" and series_id == "S2":
                prediction = float(actual)
            else:
                prediction = float(actual + 5)
            rows.append(
                {
                    "date": date(2026, 1, 1) + timedelta(days=offset),
                    "serie": series_id,
                    "cross_key_id": series_id,
                    "isTrain": is_train,
                    "actual_orig": float(actual),
                    "pred_orig": prediction,
                }
            )
    return pd.DataFrame(rows)


def _forecast_rows() -> pd.DataFrame:
    rows = []
    for series_id, base in (("S1", 7.0), ("S2", 16.0), ("S3", 25.0)):
        for offset in range(2):
            value = base + offset
            rows.append(
                {
                    "date": date(2026, 2, 1) + timedelta(days=offset),
                    "serie": series_id,
                    "cross_key_id": series_id,
                    "pred_orig": value,
                    "lower_ci": value - 1,
                    "upper_ci": value + 1,
                }
            )
    return pd.DataFrame(rows)


def _write_global_run(root: Path, candidate_id: str, architecture: str) -> Path:
    run = root / candidate_id
    (run / "model").mkdir(parents=True)
    (run / "reports").mkdir(parents=True)
    (run / "model" / "manifest.json").write_text(
        json.dumps({"architecture": architecture}), encoding="utf-8"
    )
    pl.DataFrame(_backtest_rows(candidate_id).to_dict("records")).write_parquet(
        run / "reports" / "backtest_mc_by_series.parquet"
    )
    pl.DataFrame(_forecast_rows().to_dict("records")).write_parquet(
        run / "reports" / "future_forecast_mc_by_series.parquet"
    )
    return run


class TestCheckpoint12MonitorScopeSeparation(unittest.TestCase):
    def test_monitor_02_is_local_only_with_common_baselines(self) -> None:
        path = ROOT / "monitor_codigo_02.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("discover_latest_local_runs(", code)
        self.assertIn("compare_local_financial_runs(", code)
        self.assertIn("INCLUDE_BASELINES = True", code)
        self.assertIn("SEASONAL_PERIOD_DAYS = 7", code)
        self.assertIn("assert monitor.run_inventory.height == 7", code)
        self.assertNotIn("GLOBAL_MLP_E_D", code)
        self.assertNotIn("discover_latest_global_runs(", code)

    def test_monitor_03_contains_only_global_discovery_and_comparison(self) -> None:
        path = ROOT / "monitor_codigo_03_FINANCIAL_GPT.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("discover_latest_global_runs(", code)
        self.assertIn("compare_global_financial_gpt_runs(", code)
        self.assertIn("assert monitor.run_inventory.height == 7", code)
        self.assertNotIn("discover_latest_local_runs(", code)
        self.assertNotIn("LOCAL_EXECUTION_ROOT", code)
        self.assertNotIn("local_run_uris=", code)
        self.assertTrue(
            all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        compile(code, str(path), "exec")

    def test_global_monitor_compares_exactly_four_global_plus_naive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_runs = {
                candidate_id: str(_write_global_run(root, candidate_id, architecture))
                for candidate_id, architecture in GLOBAL_ARCHITECTURES.items()
            }
            result = compare_global_financial_gpt_runs(
                global_run_uris=global_runs,
                metrics=("MAE", "RMSE", "WMAPE"),
                include_naive=True,
            )

        self.assertEqual(result.run_inventory.height, 7)
        self.assertEqual(set(result.run_inventory["family"].to_list()), {"global", "baseline"})
        self.assertEqual(result.winners_by_series.height, 3)
        winners = {
            row["cross_key_id"]: row["winner_candidate"]
            for row in result.winners_by_series.to_dicts()
        }
        self.assertEqual(winners["S1"], "GLOBAL_MLP_E_D")
        self.assertEqual(winners["S2"], "GLOBAL_RNN_E_D")
        self.assertEqual(winners["S3"], NAIVE_CANDIDATE_ID)
        counts = result.metrics_by_series.group_by("cross_key_id").len(name="count")
        self.assertTrue((counts["count"] == 6).all())
        self.assertNotIn("local", result.metrics_by_series["family"].unique().to_list())

    def test_global_monitor_requires_exactly_four_global_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four global runs"):
            compare_global_financial_gpt_runs(global_run_uris={})
        unexpected = dict(GLOBAL_ARCHITECTURES)
        unexpected["LOCAL_MLP_E_D"] = "/tmp/not-a-global-run"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            compare_global_financial_gpt_runs(global_run_uris=unexpected)


if __name__ == "__main__":
    unittest.main()
