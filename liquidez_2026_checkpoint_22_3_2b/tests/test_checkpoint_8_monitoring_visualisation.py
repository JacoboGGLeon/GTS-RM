from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optuna
import pandas as pd
import polars as pl

from global_curriculum import GlobalCurriculumConfig
from global_manager import GlobalManager
from global_notebook import GlobalNotebookDatasetFactory, prepare_calendar_frame
from global_training import GlobalCandidateConfig, GlobalTrainingConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "code_03_GLOBAL_MLP_E_D.ipynb"


def make_global_long(num_series: int = 9, length: int = 18) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows = []
    for number in range(num_series):
        account = f"ACC{number:02d}_MXN"
        kind = "saldo" if number % 2 == 0 else "variacion"
        key = f"{account}_{kind}"
        for offset in range(length):
            rows.append({
                "fecha": origin + timedelta(days=offset),
                "account_currency_id": account,
                "cross_key_id": key,
                "tipo_serie": kind,
                "target": float(number * 20 + offset + (offset % 3) * 2),
                "difficulty_score": float(1 + number % 3) / 3.0,
                "nivel_curriculum": 1 + number % 3,
                "grupo": "Grupo_2",
            })
    return pl.DataFrame(rows)


def make_calendar(length: int = 40) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame({
        "fecha": [origin + timedelta(days=i) for i in range(length)],
        "es_quincena": [float((i + 1) % 15 == 0) for i in range(length)],
        "dia_habil": [float(i % 7 < 5) for i in range(length)],
    })


def tiny_candidate(trial: optuna.Trial, architecture: str, base: GlobalTrainingConfig) -> GlobalCandidateConfig:
    trial.suggest_categorical("window_size", [3])
    return GlobalCandidateConfig(
        window_size=3,
        model_config={
            "latent_dim": 4,
            "enc_hidden_size": 8,
            "enc_num_layers": 1,
            "dec_hidden_size": 8,
            "dec_num_layers": 1,
            "dropout_rate": 0.2,
            "activation": "gelu",
        },
        training_config=base,
    )


def build_manager() -> tuple[GlobalManager, GlobalNotebookDatasetFactory]:
    calendar, columns = prepare_calendar_frame(make_calendar())
    factory = GlobalNotebookDatasetFactory(
        make_global_long(),
        calendar,
        exogenous_columns=columns,
        horizon=2,
        seen_validation_size=4,
        validation_unseen_fraction=0.2,
        test_unseen_fraction=0.2,
        stride=2,
        seed=31,
        max_window_size=3,
    )
    training = GlobalTrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        loss="mae",
        patience=1,
        scheduler_patience=1,
        samples_per_epoch=12,
        num_workers=0,
        seed=31,
        device="cpu",
    )
    curriculum = GlobalCurriculumConfig(
        warmup_epochs=1,
        fine_tune_epochs_per_level=1,
        consolidation_epochs=1,
        replay_fraction=0.25,
        fine_tune_lr_factor=0.2,
        consolidation_lr_factor=0.1,
    )
    return GlobalManager(
        "mlp",
        base_training_config=training,
        curriculum_config=curriculum,
        candidate_factory=tiny_candidate,
        seed=31,
    ), factory


class TestCheckpoint8MonitoringVisualisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager, cls.factory = build_manager()
        cls.manager._warmup_all(
            cls.factory,
            n_trials=1,
            split_manifest=cls.factory.split,
            exogenous_columns=cls.factory.exogenous_columns,
            show_progress=False,
        )

    def test_warmup_and_finetune_are_separate_and_continuous(self) -> None:
        manager, factory = build_manager()
        manager._warmup_all(
            factory,
            n_trials=1,
            split_manifest=factory.split,
            exogenous_columns=factory.exogenous_columns,
            show_progress=False,
        )
        self.assertEqual(
            {stage.stage.phase for stage in manager.training_result.stages},
            {"warmup"},
        )
        warm_end = manager.training_result.stages[-1].end_state_digest
        result = manager._finetune_all(show_progress=False)
        self.assertIn("finetune", {stage.stage.phase for stage in result.stages})
        fine = next(stage for stage in result.stages if stage.stage.phase == "finetune")
        self.assertEqual(warm_end, fine.start_state_digest)
        self.assertFalse(isinstance(manager.model, dict))

    def test_backtest_and_future_forecast_preserve_series_contract(self) -> None:
        if not any(s.stage.phase == "finetune" for s in self.manager.training_result.stages):
            self.manager._finetune_all(show_progress=False)
        backtest = self.manager._run_backtest(n_mc=3, batch_size=16, device="cpu")
        frame = backtest["df_regression"]
        self.assertFalse(frame.empty)
        self.assertTrue({True, False}.issubset(set(frame["isTrain"].unique())))
        self.assertEqual(frame.groupby(["serie", "date", "isTrain"]).size().max(), 1)
        self.assertTrue({"lower_ci", "upper_ci", "var_pred"}.issubset(frame.columns))

        last = pd.Timestamp(self.factory.global_long["fecha"].max())
        forecasts = self.manager._run_forecast(
            start_date=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=(last + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            n_mc=3,
            batch_size=16,
            device="cpu",
        )
        self.assertTrue(forecasts)
        self.assertFalse(self.manager._df_forecasts.empty)
        self.assertEqual(set(self.manager._df_forecasts["cross_key_id"]), set(forecasts))
        self.assertTrue((self.manager._df_forecasts["date"] > last).all())
        self.assertFalse(self.manager._df_outliers.empty)

    def test_visualise_calls_the_three_legacy_plots_per_series(self) -> None:
        if not self.manager._backtest_results:
            self.test_backtest_and_future_forecast_preserve_series_contract()
        series_id = next(iter(self.manager._future_results))
        last = pd.Timestamp(self.factory.global_long["fecha"].max())
        with patch("tools.Tools.plot_backtest_for_serie") as bt, patch(
            "tools.Tools.plot_forecast_with_outliers"
        ) as fc, patch("tools.Tools.plot_backtest_forecast_with_outliers_for_serie") as both:
            self.manager.visualise(
                bt_start="2026-01-01",
                bt_end=last.strftime("%Y-%m-%d"),
                fc_start=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                fc_end=(last + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                series_ids=[series_id],
            )
        bt.assert_called_once()
        fc.assert_called_once()
        both.assert_called_once()

    def test_legacy_results_has_the_exact_public_keys(self) -> None:
        results = self.manager.legacy_results()
        self.assertEqual(
            set(results),
            {"warm", "fine", "backtest", "forecast", "df_forecasts", "df_outliers"},
        )

    def test_notebook_exposes_five_step_contract_and_no_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            "workflow.run_hpo_and_train(",
            "workflow.run_backtest(",
            "workflow.run_forecast(",
            "manager.visualise(",
            "results = manager.run_results()",
        ):
            self.assertIn(token, code)
        for private_token in (
            "manager._warmup_all(",
            "manager._finetune_all(",
            "manager._run_backtest(",
            "manager._run_forecast(",
        ):
            self.assertNotIn(private_token, code)
        self.assertTrue(
            all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        compile(code, str(NOTEBOOK), "exec")


if __name__ == "__main__":
    unittest.main()

class TestCheckpoint8GlobalMonitor(unittest.TestCase):
    def test_monitor_selects_one_winner_and_ensemble_row_per_series_date(self) -> None:
        from global_monitor import compare_global_runs

        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for architecture, offset in (("mlp", 0.0), ("rnn", 1.0)):
                root = Path(tmp) / f"GLOBAL_{architecture.upper()}"
                (root / "model").mkdir(parents=True)
                (root / "reports").mkdir(parents=True)
                (root / "model" / "manifest.json").write_text(
                    json.dumps({"architecture": architecture}), encoding="utf-8"
                )
                pl.DataFrame(
                    {
                        "partition": ["validation_seen", "validation_unseen"],
                        "macro_mae": [1.0 + offset, 2.0 + offset],
                    }
                ).write_parquet(root / "reports" / "evaluation_metrics.parquet")
                pl.DataFrame(
                    {
                        "serie": ["S1", "S2"],
                        "MAE": [1.0 + offset, 3.0 - offset],
                        "RMSE": [1.5 + offset, 3.5 - offset],
                        "WMAPE": [10.0 + offset, 30.0 - offset],
                        "EVS": [0.9 - offset * 0.1, 0.5 + offset * 0.1],
                        "R2": [0.8 - offset * 0.1, 0.4 + offset * 0.1],
                    }
                ).write_parquet(root / "reports" / "backtest_metrics_by_series.parquet")
                pl.DataFrame(
                    {
                        "date": [date(2026, 2, 1), date(2026, 2, 1)],
                        "cross_key_id": ["S1", "S2"],
                        "account_currency_id": ["A1", "A2"],
                        "tipo_serie": ["saldo", "saldo"],
                        "pred_orig": [100.0 + offset, 200.0 + offset],
                        "lower_ci": [90.0, 190.0],
                        "upper_ci": [110.0, 210.0],
                        "outlier_level": [0, 0],
                    }
                ).write_parquet(root / "reports" / "future_forecast_mc_by_series.parquet")
                roots.append(str(root))

            result = compare_global_runs(roots)
            self.assertEqual(result.winners_by_series.height, 2)
            self.assertEqual(result.ensemble_forecast.height, 2)
            self.assertTrue(
                (
                    result.ensemble_forecast["architecture"]
                    == result.ensemble_forecast["winner_architecture"]
                ).all()
            )
