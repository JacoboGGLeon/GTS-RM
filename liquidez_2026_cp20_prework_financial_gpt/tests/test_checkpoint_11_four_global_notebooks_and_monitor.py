from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import polars as pl

from financial_gpt_monitor import (
    GLOBAL_ARCHITECTURES,
    LOCAL_ALGORITHMS,
    NAIVE_CANDIDATE_ID,
    compare_financial_gpt_runs,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTEBOOK_HASHES = {
    "code_02_MLP_E_D.ipynb": "19f1fca1250691fab6690068afcfa0e67b1b18ae5146b8f3d3381713e2fa0183",
    "code_02_MLP_VaE_D.ipynb": "f5e33c29f19abf985eb4744232e81342edf0ddfda35adf7162176c2b9c4c6fe7",
    "code_02_RNN_E_D.ipynb": "f5835b0ab86344d2657d02a868f481d93908f7469c005c23b881905b26917208",
    "code_02_RNNBi_E_D.ipynb": "824d332d4eadc4935830e44e26189a42ad79568f270b15c3126f676ab872b0f3",
}
GLOBAL_NOTEBOOKS = {
    "code_03_GLOBAL_MLP_E_D.ipynb": ("mlp", "GLOBAL_MLP_E_D"),
    "code_03_GLOBAL_MLP_VaE_D.ipynb": ("mlp_vae", "GLOBAL_MLP_VaE_D"),
    "code_03_GLOBAL_RNN_E_D.ipynb": ("rnn", "GLOBAL_RNN_E_D"),
    "code_03_GLOBAL_RNNBi_E_D.ipynb": ("rnn_bi", "GLOBAL_RNNBi_E_D"),
}


def _backtest_rows(candidate: str) -> pd.DataFrame:
    origin = date(2026, 1, 1)
    series_values = {
        "S1": [1, 2, 3, 4, 5, 6],
        "S2": [10, 11, 12, 13, 14, 15],
        "S3": [20, 20, 20, 20, 20, 20],
    }
    rows = []
    for series_id, values in series_values.items():
        for offset, actual in enumerate(values):
            is_train = offset < 3
            if is_train:
                prediction = float(actual)
            elif candidate == "LOCAL_MLP_E_D" and series_id == "S1":
                prediction = float(actual)
            elif candidate == "GLOBAL_RNN_E_D" and series_id == "S2":
                prediction = float(actual)
            else:
                prediction = float(actual + 5)
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


def _forecast_rows(candidate: str) -> pd.DataFrame:
    rows = []
    for series_id, base in (("S1", 7.0), ("S2", 16.0), ("S3", 25.0)):
        for offset in range(2):
            rows.append(
                {
                    "date": date(2026, 2, 1) + timedelta(days=offset),
                    "serie": series_id,
                    "cross_key_id": series_id,
                    "pred_orig": base + offset,
                    "lower_ci": base + offset - 1,
                    "upper_ci": base + offset + 1,
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
    _forecast_rows(f"LOCAL_{algorithm}").to_csv(
        data / "forecast_timeseries.csv", index=True
    )
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
    pl.DataFrame(_forecast_rows(candidate_id).to_dict("records")).write_parquet(
        run / "reports" / "future_forecast_mc_by_series.parquet"
    )
    return run


class TestCheckpoint11Notebooks(unittest.TestCase):
    def test_exact_four_local_and_four_global_notebooks(self) -> None:
        for name, expected_hash in LOCAL_NOTEBOOK_HASHES.items():
            path = ROOT / name
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
        for name in GLOBAL_NOTEBOOKS:
            self.assertTrue((ROOT / name).is_file())
        self.assertFalse((ROOT / "code_03_GLOBAL_DRL.ipynb").exists())

    def test_global_notebooks_are_fixed_variants_with_same_contract(self) -> None:
        for name, (architecture, label) in GLOBAL_NOTEBOOKS.items():
            notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertIn(f'ARCHITECTURE = "{architecture}"', code)
            self.assertIn(f'GLOBAL_MODEL_LABEL = "{label}"', code)
            self.assertIn(f'NOTEBOOK_FILENAME = "{name}"', code)
            for token in (
                "manager._warmup_all(",
                "manager._finetune_all(",
                "manager._run_backtest(",
                "manager._run_forecast(",
                "manager.visualise(",
                "manager.save_model_s3(",
                "GlobalManager.load_model_s3(",
            ):
                self.assertIn(token, code)
            self.assertTrue(
                all(
                    not cell.get("outputs") and cell.get("execution_count") is None
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
            )
            compile(code, str(ROOT / name), "exec")

    def test_final_monitor_notebook_has_four_global_plus_naive_contract(self) -> None:
        path = ROOT / "monitor_codigo_03_FINANCIAL_GPT.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            "discover_latest_global_runs(",
            "compare_global_financial_gpt_runs(",
            'INCLUDE_BASELINES = True',
            'assert monitor.run_inventory.height == 7',
        ):
            self.assertIn(token, code)
        self.assertNotIn("discover_latest_local_runs(", code)
        self.assertNotIn("local_run_uris=", code)
        self.assertTrue(
            all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        compile(code, str(path), "exec")


class TestCheckpoint11FinalMonitor(unittest.TestCase):
    def test_monitor_compares_nine_candidates_and_selects_each_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_runs = {
                algorithm: str(_write_local_run(root / "local", algorithm))
                for algorithm in LOCAL_ALGORITHMS
            }
            global_runs = {
                candidate_id: str(
                    _write_global_run(root / "global", candidate_id, architecture)
                )
                for candidate_id, architecture in GLOBAL_ARCHITECTURES.items()
            }
            result = compare_financial_gpt_runs(
                local_run_uris=local_runs,
                global_run_uris=global_runs,
                metrics=("MAE", "RMSE", "WMAPE"),
                include_naive=True,
            )

        self.assertEqual(result.run_inventory.height, 11)
        self.assertEqual(result.winners_by_series.height, 3)
        winners = {
            row["cross_key_id"]: row["winner_candidate"]
            for row in result.winners_by_series.to_dicts()
        }
        self.assertEqual(winners["S1"], "LOCAL_MLP_E_D")
        self.assertEqual(winners["S2"], "GLOBAL_RNN_E_D")
        self.assertEqual(winners["S3"], NAIVE_CANDIDATE_ID)
        candidate_counts = (
            result.metrics_by_series.group_by("cross_key_id")
            .len(name="count")
            .to_dicts()
        )
        self.assertTrue(all(row["count"] == 10 for row in candidate_counts))
        self.assertEqual(result.ensemble_forecast["cross_key_id"].n_unique(), 3)
        naive = result.ensemble_forecast.filter(
            result.ensemble_forecast["winner_candidate"] == NAIVE_CANDIDATE_ID
        )
        self.assertTrue((naive["pred_orig"] == 20.0).all())

    def test_monitor_requires_all_four_local_and_global_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "four local and four global"):
            compare_financial_gpt_runs(
                local_run_uris={},
                global_run_uris={},
            )

    def test_result_writes_all_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_runs = {
                algorithm: str(_write_local_run(root / "local", algorithm))
                for algorithm in LOCAL_ALGORITHMS
            }
            global_runs = {
                candidate_id: str(
                    _write_global_run(root / "global", candidate_id, architecture)
                )
                for candidate_id, architecture in GLOBAL_ARCHITECTURES.items()
            }
            result = compare_financial_gpt_runs(
                local_run_uris=local_runs,
                global_run_uris=global_runs,
                metrics=("MAE", "RMSE", "WMAPE"),
            )
            output = result.write(root / "output")
            self.assertEqual(output.name, "financial_gpt_monitor.json")
            self.assertTrue(output.is_file())
            self.assertEqual(
                {path.name for path in output.parent.iterdir() if path.is_file()},
                {"financial_gpt_monitor.json"},
            )


if __name__ == "__main__":
    unittest.main()
