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
    LOCAL_ALGORITHMS,
    NAIVE_LAST_VALUE_ID,
    NAIVE_ZERO_ID,
    SEASONAL_NAIVE_ID,
    compare_global_financial_gpt_runs,
    compare_local_financial_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def _backtest_rows(candidate_id: str) -> pd.DataFrame:
    origin = date(2026, 1, 1)
    weekly = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    saldo = weekly + weekly + weekly
    variation_train = [1.0, -1.0] * 7
    variation = variation_train + [0.0] * 7
    rows = []
    for series_id, values in {
        "ACC_MXN_saldo": saldo,
        "ACC_MXN_variacion": variation,
    }.items():
        for offset, actual in enumerate(values):
            is_train = offset < 14
            prediction = float(actual) if is_train else float(actual + 5.0)
            rows.append(
                {
                    "date": origin + timedelta(days=offset),
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
    for series_id in ("ACC_MXN_saldo", "ACC_MXN_variacion"):
        for offset in range(7):
            rows.append(
                {
                    "date": date(2026, 1, 22) + timedelta(days=offset),
                    "serie": series_id,
                    "cross_key_id": series_id,
                    "pred_orig": 999.0,
                    "lower_ci": 998.0,
                    "upper_ci": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _write_local_run(root: Path, algorithm: str) -> Path:
    run = root / algorithm
    data = run / "data"
    data.mkdir(parents=True)
    _backtest_rows(f"LOCAL_{algorithm}").to_csv(
        data / "backtest_timeseries.csv", index=True
    )
    _forecast_rows().to_csv(data / "forecast_timeseries.csv", index=True)
    return run


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


class TestCheckpoint13UnifiedFinancialBaselines(unittest.TestCase):
    def test_local_monitor_uses_three_baselines_with_zero_only_for_variation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {
                algorithm: str(_write_local_run(root, algorithm))
                for algorithm in LOCAL_ALGORITHMS
            }
            result = compare_local_financial_runs(
                local_run_uris=runs,
                metrics=("MAE", "RMSE", "WMAPE"),
                include_baselines=True,
                seasonal_period_days=7,
            )

        self.assertEqual(result.run_inventory.height, 7)
        metrics = result.metrics_by_series
        saldo_candidates = set(
            metrics.filter(pl.col("cross_key_id") == "ACC_MXN_saldo")[
                "candidate_id"
            ].to_list()
        )
        variation_candidates = set(
            metrics.filter(pl.col("cross_key_id") == "ACC_MXN_variacion")[
                "candidate_id"
            ].to_list()
        )
        self.assertNotIn(NAIVE_ZERO_ID, saldo_candidates)
        self.assertIn(NAIVE_ZERO_ID, variation_candidates)
        self.assertIn(NAIVE_LAST_VALUE_ID, saldo_candidates)
        self.assertIn(SEASONAL_NAIVE_ID, saldo_candidates)

        winners = {
            row["cross_key_id"]: row["winner_candidate"]
            for row in result.winners_by_series.to_dicts()
        }
        self.assertEqual(winners["ACC_MXN_saldo"], SEASONAL_NAIVE_ID)
        self.assertEqual(winners["ACC_MXN_variacion"], NAIVE_ZERO_ID)

    def test_global_monitor_has_same_baseline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {
                candidate_id: str(
                    _write_global_run(root, candidate_id, architecture)
                )
                for candidate_id, architecture in GLOBAL_ARCHITECTURES.items()
            }
            result = compare_global_financial_gpt_runs(
                global_run_uris=runs,
                metrics=("MAE", "RMSE", "WMAPE"),
                include_baselines=True,
                seasonal_period_days=7,
            )

        self.assertEqual(result.run_inventory.height, 7)
        self.assertEqual(
            set(
                result.run_inventory.filter(pl.col("family") == "baseline")[
                    "candidate_id"
                ].to_list()
            ),
            {NAIVE_LAST_VALUE_ID, NAIVE_ZERO_ID, SEASONAL_NAIVE_ID},
        )
        winners = {
            row["cross_key_id"]: row["winner_candidate"]
            for row in result.winners_by_series.to_dicts()
        }
        self.assertEqual(winners["ACC_MXN_saldo"], SEASONAL_NAIVE_ID)
        self.assertEqual(winners["ACC_MXN_variacion"], NAIVE_ZERO_ID)
        zero_forecast = result.ensemble_forecast.filter(
            pl.col("winner_candidate") == NAIVE_ZERO_ID
        )
        self.assertTrue((zero_forecast["pred_orig"] == 0.0).all())

    def test_official_monitors_remain_strictly_scoped(self) -> None:
        local_nb = json.loads(
            (ROOT / "monitor_codigo_02.ipynb").read_text(encoding="utf-8")
        )
        global_nb = json.loads(
            (ROOT / "monitor_codigo_03_FINANCIAL_GPT.ipynb").read_text(
                encoding="utf-8"
            )
        )
        local_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in local_nb["cells"]
            if cell["cell_type"] == "code"
        )
        global_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in global_nb["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("compare_local_financial_runs(", local_code)
        self.assertNotIn("compare_global_financial_gpt_runs(", local_code)
        self.assertIn("compare_global_financial_gpt_runs(", global_code)
        self.assertNotIn("compare_local_financial_runs(", global_code)
        for code in (local_code, global_code):
            self.assertIn("INCLUDE_BASELINES = True", code)
            self.assertIn("SEASONAL_PERIOD_DAYS = 7", code)
            self.assertIn("assert monitor.run_inventory.height == 7", code)


if __name__ == "__main__":
    unittest.main()
