from __future__ import annotations

import csv
import json
import math
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import polars as pl
import torch

from global_data import ContextScaler, GlobalWindowDataset
from global_models import build_global_model
from global_notebook import GlobalNotebookDatasetFactory
from global_training import (
    GlobalCandidateConfig,
    GlobalDatasetBundle,
    GlobalEpochRecord,
    GlobalHPOConfig,
    GlobalHPOTrainer,
    GlobalTrainer,
    GlobalTrainingConfig,
    GlobalTrainingResult,
    GlobalValidationMetrics,
    suggest_global_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "code_03_GLOBAL_MLP_E_D.ipynb"


def make_global_long(series_count: int = 6, length: int = 14) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for number in range(series_count):
        account = f"A{number:02d}_MXN"
        series_type = "saldo" if number % 2 == 0 else "variacion"
        key = f"{account}_{series_type}"
        for offset in range(length):
            rows.append(
                {
                    "fecha": origin + timedelta(days=offset),
                    "account_currency_id": account,
                    "cross_key_id": key,
                    "tipo_serie": series_type,
                    "target": float(1_000 + number * 100 + offset),
                    "difficulty_score": 0.5,
                    "nivel_curriculum": 1 + number % 3,
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


def make_calendar(length: int = 20) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=offset) for offset in range(length)],
            "dia_habil": [float(offset % 7 < 5) for offset in range(length)],
        }
    )


def make_bundle(window_size: int = 3) -> GlobalDatasetBundle:
    frame = make_global_long()
    calendar = make_calendar()
    train_ids = frame["cross_key_id"].unique().sort().to_list()[:4]
    unseen_ids = frame["cross_key_id"].unique().sort().to_list()[4:]
    train = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(train_ids)),
        window_size=window_size,
        horizon=2,
        exogenous=calendar,
        exogenous_columns=("dia_habil",),
    )
    seen = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(train_ids)),
        window_size=window_size,
        horizon=2,
        exogenous=calendar,
        exogenous_columns=("dia_habil",),
    )
    unseen = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(unseen_ids)),
        window_size=window_size,
        horizon=2,
        exogenous=calendar,
        exogenous_columns=("dia_habil",),
    )
    return GlobalDatasetBundle(train, seen, unseen)


def validation_metrics(value: float = 10.0) -> GlobalValidationMetrics:
    return GlobalValidationMetrics(
        macro_mae=value,
        macro_rmse=value,
        micro_mae=value,
        raw_macro_mae=value,
        raw_macro_rmse=value,
        raw_macro_wmape=value,
        raw_macro_smape=value,
        num_series=1,
        num_points=1,
        per_series={"S": {"raw_smape": value, "robust_mase": value}},
        robust_macro_mase=value,
    )


class TestCheckpoint9OriginalMethodologyAlignment(unittest.TestCase):
    def test_context_scaler_is_agnostic_and_stable_for_large_levels(self) -> None:
        scaler = ContextScaler()
        values = np.asarray([1_000_000_000.0] * 10)
        parameters = scaler.fit(values, series_type="saldo")
        self.assertEqual(parameters.transform, "linear_context_scale")
        self.assertEqual(parameters.center, 0.0)
        self.assertEqual(parameters.scale, 1_000_000_000.0)
        future = np.asarray([1_001_000_000.0])
        transformed = scaler.transform(future, parameters)
        self.assertLessEqual(float(abs(transformed[0])), 1.01)
        np.testing.assert_allclose(
            scaler.inverse_transform(transformed, parameters),
            future,
        )

    def test_series_type_does_not_change_scaling_contract(self) -> None:
        scaler = ContextScaler()
        values = np.asarray([-20.0, -10.0, 100.0])
        first = scaler.fit(values, series_type="variacion")
        second = scaler.fit(values, series_type="flujo_contractual")
        self.assertEqual(first, second)
        self.assertEqual(scaler.contract()["series_type_dependency"], "none")
        self.assertEqual(scaler.contract()["fit_scope"], "y_context_only")

    def test_default_candidate_keeps_batch_and_loss_outside_hpo(self) -> None:
        fixed = optuna.trial.FixedTrial(
            {
                "window_size": 5,
                "latent_dim": 32,
                "dropout_rate": 0.1,
                "activation": "gelu",
                "mlp_hidden_size": 64,
                "mlp_num_layers": 1,
                "learning_rate": 1e-3,
            }
        )
        base = GlobalTrainingConfig(batch_size=128, loss="huber")
        candidate = suggest_global_candidate(fixed, "mlp", base)
        self.assertEqual(candidate.training_config.batch_size, 128)
        self.assertEqual(candidate.training_config.loss, "huber")
        self.assertNotIn("batch_size", fixed.params)
        self.assertNotIn("loss", fixed.params)

    def test_hpo_uses_hyperband_cache_proxy_and_medium_fidelity_selection(self) -> None:
        calls = {"factory": 0, "fit": 0}
        bundle = make_bundle()
        base = GlobalTrainingConfig(
            epochs=20,
            batch_size=8,
            loss="huber",
            samples_per_epoch=None,
            device="cpu",
        )

        def factory(window_size: int) -> GlobalDatasetBundle:
            calls["factory"] += 1
            self.assertEqual(window_size, 3)
            return bundle

        def candidate_factory(trial, architecture, base_config):
            trial.suggest_categorical("window_size", [3])
            trial.suggest_categorical("learning_rate", [1e-3])
            return GlobalCandidateConfig(
                window_size=3,
                model_config={
                    "latent_dim": 4,
                    "enc_hidden_size": 8,
                    "enc_num_layers": 1,
                    "dec_hidden_size": 8,
                    "dec_num_layers": 1,
                    "dropout_rate": 0.0,
                    "activation": "gelu",
                },
                training_config=base_config,
            )

        def fake_fit(self, datasets, *, epoch_callback=None, **kwargs):
            calls["fit"] += 1
            metrics = {
                "validation_seen": validation_metrics(10.0),
                "validation_unseen": validation_metrics(20.0),
            }
            record = GlobalEpochRecord(
                epoch=1,
                train_loss=1.0,
                validation_objective=15.0,
                learning_rate=1e-3,
                validation=metrics,
            )
            if epoch_callback is not None:
                epoch_callback(record)
            model = build_global_model(
                "mlp",
                self.model_config,
                window_size=3,
                horizon=2,
                exogenous_dim=1,
            )
            return GlobalTrainingResult(
                architecture="mlp",
                model=model,
                model_config=self.model_config,
                training_config=self.training_config,
                history=(record,),
                best_epoch=1,
                best_score=15.0,
                validation=metrics,
                stopped_early=False,
            )

        trainer = GlobalHPOTrainer(
            "mlp",
            base_training_config=base,
            hpo_config=GlobalHPOConfig(
                epochs=1,
                windows_per_series_per_epoch=4,
                validation_windows_per_series=3,
                reduction_factor=2,
                finalists=2,
                fidelity_epochs=2,
                fidelity_windows_per_series_per_epoch=6,
            ),
            candidate_factory=candidate_factory,
            seed=7,
        )
        with patch.object(GlobalTrainer, "fit", new=fake_fit):
            result = trainer.search_and_fit(factory, n_trials=2)

        self.assertEqual(type(result.study.pruner).__name__, "HyperbandPruner")
        self.assertEqual(calls["factory"], 1)  # cache por window_size
        self.assertEqual(calls["fit"], 4)  # dos proxy + dos finalistas a fidelidad media
        self.assertEqual(
            result.study.best_trial.user_attrs["proxy_samples_per_epoch"],
            len(bundle.train.series_ids) * 4,
        )
        self.assertEqual(
            result.study.best_trial.user_attrs["validation_windows_per_series"],
            3,
        )
        self.assertEqual(
            result.study.best_trial.user_attrs["objective_metric"],
            "robust_macro_mase",
        )

    def test_eligibility_manifest_and_scaler_contract_are_explicit(self) -> None:
        factory = GlobalNotebookDatasetFactory(
            make_global_long(length=14),
            make_calendar(),
            exogenous_columns=("dia_habil",),
            horizon=2,
            seen_validation_size=4,
            validation_unseen_fraction=0.2,
            test_unseen_fraction=0.2,
            max_window_size=3,
        )
        self.assertTrue(
            {
                "row_count",
                "minimum_required_rows",
                "eligible",
                "status",
            }.issubset(factory.eligibility_manifest.columns)
        )
        self.assertEqual(factory.scaler_contract["fit_scope"], "y_context_only")

    def test_notebook_exposes_fast_hpo_outputs_and_remains_clean(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            "HPO_EPOCHS = 4",
            "HPO_WINDOWS_PER_SERIES = 6",
            "HPO_VALIDATION_WINDOWS_PER_SERIES = 4",
            "GlobalHPOConfig(",
            'SELECTION_METRIC = "robust_macro_mase"',
            "objective_metric=budget.selection_metric",
            'reports_dir / "hpo_trials.parquet"',
            'reports_dir / "scaler_contract.json"',
            'reports_dir / "eligibility_manifest.parquet"',
        ):
            self.assertIn(token, code)
        self.assertTrue(
            all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        compile(code, str(NOTEBOOK), "exec")

    def test_four_global_notebooks_requirement_is_registered_for_checkpoint_11(self) -> None:
        rows = list(csv.DictReader((ROOT / "GLOBAL_MODEL_CHECKPOINTS.csv").open()))
        checkpoint = next(row for row in rows if row["checkpoint"] == "11")
        self.assertEqual(checkpoint["status"], "completed")
        for name in (
            "code_03_GLOBAL_MLP_E_D.ipynb",
            "code_03_GLOBAL_MLP_VaE_D.ipynb",
            "code_03_GLOBAL_RNN_E_D.ipynb",
            "code_03_GLOBAL_RNNBi_E_D.ipynb",
        ):
            self.assertIn(name, checkpoint["modified_files"])
        for local_name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            self.assertTrue((ROOT / local_name).is_file())


if __name__ == "__main__":
    unittest.main()
