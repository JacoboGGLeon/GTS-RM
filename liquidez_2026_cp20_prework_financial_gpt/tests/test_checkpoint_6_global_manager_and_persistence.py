from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import optuna
import polars as pl
import torch

from global_curriculum import GlobalCurriculumConfig, state_dict_digest
from global_data import GlobalWindowDataset
from global_manager import (
    FORECAST_COLUMNS,
    GlobalManager,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
)
from global_training import (
    GlobalCandidateConfig,
    GlobalDatasetBundle,
    GlobalTrainingConfig,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG: Mapping[str, Any] = {
    "latent_dim": 4,
    "enc_hidden_size": 8,
    "enc_num_layers": 1,
    "dec_hidden_size": 8,
    "dec_num_layers": 1,
    "dropout_rate": 0.0,
    "activation": "gelu",
}


def make_rows(
    series_levels: Mapping[int, int],
    *,
    start_day: int,
    length: int,
) -> list[dict[str, object]]:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for series_number, level in series_levels.items():
        account_currency_id = f"ACC{series_number:02d}_MXN"
        series_type = "saldo" if series_number % 2 == 0 else "variacion"
        cross_key_id = f"{account_currency_id}_{series_type}"
        for offset in range(length):
            day_number = start_day + offset
            rows.append(
                {
                    "fecha": origin + timedelta(days=day_number),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(series_number * 10 + day_number + 1),
                    "difficulty_score": float(level) / 3.0,
                    "nivel_curriculum": level,
                    "grupo": "Grupo_2",
                }
            )
    return rows


def make_calendar(total_days: int = 50) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=day) for day in range(total_days)],
            "es_quincena": [float(day % 15 == 14) for day in range(total_days)],
            "dia_habil": [float(day % 7 < 5) for day in range(total_days)],
        }
    )


def make_bundle(window_size: int = 3, horizon: int = 2) -> GlobalDatasetBundle:
    levels = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
    unseen = {6: 1, 7: 2, 8: 3}
    calendar = make_calendar()
    exogenous_columns = ("es_quincena", "dia_habil")
    train = GlobalWindowDataset(
        pl.DataFrame(make_rows(levels, start_day=0, length=10)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_seen = GlobalWindowDataset(
        pl.DataFrame(make_rows(levels, start_day=14, length=8)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_unseen = GlobalWindowDataset(
        pl.DataFrame(make_rows(unseen, start_day=14, length=8)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    return GlobalDatasetBundle(train, validation_seen, validation_unseen)


def tiny_training_config() -> GlobalTrainingConfig:
    return GlobalTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        loss="mae",
        patience=2,
        scheduler_patience=1,
        samples_per_epoch=12,
        num_workers=0,
        seed=31,
        device="cpu",
    )


def tiny_curriculum_config() -> GlobalCurriculumConfig:
    return GlobalCurriculumConfig(
        warmup_epochs=1,
        fine_tune_epochs_per_level=1,
        consolidation_epochs=1,
        replay_fraction=0.25,
        fine_tune_lr_factor=0.2,
        consolidation_lr_factor=0.1,
    )


def fixed_candidate(
    trial: optuna.Trial,
    architecture: str,
    base: GlobalTrainingConfig,
) -> GlobalCandidateConfig:
    trial.suggest_categorical("fixed_window", [3])
    return GlobalCandidateConfig(
        window_size=3,
        model_config=MODEL_CONFIG,
        training_config=base,
    )


class TestCheckpoint6GlobalManagerAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_bundle()
        cls.manager = GlobalManager(
            "mlp",
            base_training_config=tiny_training_config(),
            curriculum_config=tiny_curriculum_config(),
            candidate_factory=fixed_candidate,
            seed=31,
        )
        cls.result = cls.manager.fit_global(
            make_bundle,
            n_trials=1,
            study_name="checkpoint_6_manager",
            split_manifest={
                "train_series": list(cls.bundle.train.series_ids),
                "validation_unseen_series": list(
                    cls.bundle.validation_unseen.series_ids
                ),
            },
            exogenous_columns=("es_quincena", "dia_habil"),
            run_metadata={"purpose": "unit-test"},
        )

    def test_manager_orchestrates_one_hpo_and_one_curriculum_model(self) -> None:
        self.assertTrue(self.manager.is_fitted)
        self.assertEqual(len(self.manager.hpo_result.study.trials), 1)
        self.assertFalse(isinstance(self.manager.model, dict))
        self.assertIs(self.manager.model, self.result.model)
        self.assertEqual(self.result.total_epochs, 4)
        self.assertEqual(
            state_dict_digest(self.manager.model.state_dict()),
            self.result.stages[-1].end_state_digest,
        )
        self.assertEqual(self.manager.dimensions.window_size, 3)
        self.assertEqual(self.manager.dimensions.exogenous_dim, 2)

    def test_seen_and_unseen_backtests_are_macro_by_series(self) -> None:
        seen = self.manager.backtest_seen(device="cpu")
        unseen = self.manager.backtest_unseen(device="cpu")
        self.assertEqual(seen.num_series, 6)
        self.assertEqual(unseen.num_series, 3)
        self.assertTrue(math.isfinite(seen.macro_mae))
        self.assertTrue(math.isfinite(unseen.raw_macro_rmse))

    def test_forecast_returns_auditable_long_output_in_original_scale(self) -> None:
        frame = self.manager.forecast(
            self.bundle.validation_unseen,
            device="cpu",
        )
        self.assertEqual(tuple(frame.columns), FORECAST_COLUMNS)
        self.assertEqual(
            frame.height,
            len(self.bundle.validation_unseen) * self.bundle.validation_unseen.horizon,
        )
        self.assertEqual(frame.get_column("cross_key_id").n_unique(), 3)
        self.assertTrue(frame.get_column("prediction").is_finite().all())
        expected = (
            frame.get_column("prediction_scaled") * frame.get_column("scale")
            + frame.get_column("center")
        ).alias("expected")
        checked = frame.with_columns(expected)
        self.assertTrue(
            (checked.get_column("prediction") - checked.get_column("expected")).abs().max() < 1e-6
        )

    def test_save_and_load_preserve_predictions_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self.manager.save_model(Path(tmp) / "global_run")
            expected_files = {
                "manifest.json",
                "model_state.pt",
                "metrics.json",
                "history.json",
                "hpo_summary.json",
                "split_manifest.json",
            }
            self.assertEqual({path.name for path in artifact.iterdir()}, expected_files)
            manifest = json.loads((artifact / MANIFEST_FILENAME).read_text())
            self.assertEqual(manifest["architecture"], "mlp")
            self.assertEqual(manifest["num_hpo_trials"], 1)
            self.assertEqual(manifest["run_metadata"]["purpose"], "unit-test")
            self.assertNotIn("models", manifest)

            before = self.manager.forecast(
                self.bundle.validation_unseen, device="cpu"
            )
            loaded = GlobalManager.load_model(artifact, map_location="cpu")
            after = loaded.forecast(self.bundle.validation_unseen, device="cpu")
            self.assertEqual(before.to_dicts(), after.to_dicts())
            self.assertEqual(
                state_dict_digest(self.manager.model.state_dict()),
                state_dict_digest(loaded.model.state_dict()),
            )
            self.assertEqual(loaded.split_manifest, self.manager.split_manifest)
            self.assertEqual(loaded.best_candidate["window_size"], 3)

    def test_checksum_detects_tampered_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self.manager.save_model(Path(tmp) / "tampered")
            with (artifact / MODEL_FILENAME).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                GlobalManager.load_model(artifact)

    def test_loaded_manager_requires_explicit_dataset_for_default_backtest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = GlobalManager.load_model(
                self.manager.save_model(Path(tmp) / "loaded"),
                map_location="cpu",
            )
            with self.assertRaisesRegex(RuntimeError, "explicit dataset"):
                loaded.backtest_seen()
            metrics = loaded.backtest_seen(
                self.bundle.validation_seen,
                device="cpu",
            )
            self.assertEqual(metrics.num_series, 6)

    def test_unfitted_manager_rejects_inference_and_persistence(self) -> None:
        manager = GlobalManager("mlp")
        with self.assertRaisesRegex(RuntimeError, "fitted or loaded"):
            _ = manager.model
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "fitted or loaded"):
                manager.save_model(Path(tmp) / "invalid")

    def test_checkpoint_does_not_modify_protected_pipeline_files(self) -> None:
        protected = (
            "code_01.ipynb",
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
            "engineer.py",
            "scientist.py",
            "manager.py",
            "models.py",
            "global_models.py",
            "global_training.py",
            "global_curriculum.py",
            "monitor_codigo_01.ipynb",
            "monitor_codigo_02.ipynb",
        )
        for name in protected:
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
