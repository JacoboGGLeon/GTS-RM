from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Mapping

import optuna
import polars as pl

from global_curriculum import GlobalCurriculumConfig
from global_manager import GlobalManager
from global_notebook import (
    GlobalNotebookConfig,
    GlobalNotebookDatasetFactory,
    find_latest_global_long_uri,
    load_global_inputs,
    prepare_calendar_frame,
    read_polars_uri,
)
from global_training import GlobalCandidateConfig, GlobalTrainingConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "code_03_GLOBAL_MLP_E_D.ipynb"


def make_global_long(*, num_series: int = 12, length: int = 24) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for number in range(num_series):
        account_currency_id = f"ACC{number:02d}_MXN"
        series_type = "saldo" if number % 2 == 0 else "variacion"
        cross_key_id = f"{account_currency_id}_{series_type}"
        level = 1 + number % 3
        for offset in range(length):
            rows.append(
                {
                    "fecha": origin + timedelta(days=offset),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(number * 10 + offset + 1),
                    "difficulty_score": float(level) / 3.0,
                    "nivel_curriculum": level,
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


def make_calendar(*, length: int = 30) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=offset) for offset in range(length)],
            "es_quincena": [offset % 15 == 14 for offset in range(length)],
            "dia_habil": [float(offset % 7 < 5) for offset in range(length)],
            "etiqueta": ["laboral" for _ in range(length)],
        }
    )


def tiny_candidate(
    trial: optuna.Trial,
    architecture: str,
    base: GlobalTrainingConfig,
) -> GlobalCandidateConfig:
    trial.suggest_categorical("window_size", [3])
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
        training_config=base,
    )


class TestCheckpoint7ParameterizedGlobalNotebook(unittest.TestCase):
    def test_canonical_global_notebook_preserves_parameter_cell_without_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        parameter_cells = [
            cell for cell in code_cells if "parameters" in cell.get("metadata", {}).get("tags", [])
        ]
        self.assertEqual(len(parameter_cells), 1)
        parameters = "".join(parameter_cells[0]["source"])
        self.assertIn('ARCHITECTURE = "mlp"', parameters)
        self.assertIn('GLOBAL_MODEL_LABEL = "GLOBAL_MLP_E_D"', parameters)
        self.assertIn('NOTEBOOK_FILENAME = "code_03_GLOBAL_MLP_E_D.ipynb"', parameters)
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))

        source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        self.assertEqual(source.count("GlobalManager("), 1)
        self.assertIn("manager.fit_global(", source)
        self.assertIn("manager.backtest_seen(", source)
        self.assertIn("manager.backtest_unseen(", source)
        self.assertIn("manager.save_model(", source)
        self.assertNotIn("Engineer(", source)
        self.assertNotIn("Scientist(", source)
        compile(source, str(NOTEBOOK), "exec")

    def test_configuration_rejects_invalid_execution_contracts(self) -> None:
        valid = GlobalNotebookConfig(
            architecture="mlp",
            global_long_uri="global.parquet",
            calendar_uri="calendar.csv",
            artifact_root="runs",
            horizon=2,
            seen_validation_size=4,
            n_trials=1,
            max_window_size=3,
        )
        valid.validate()
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            GlobalNotebookConfig(
                **{**valid.__dict__, "architecture": "transformer"}
            ).validate()
        with self.assertRaisesRegex(ValueError, "at least horizon"):
            GlobalNotebookConfig(
                **{**valid.__dict__, "seen_validation_size": 1}
            ).validate()

    def test_calendar_infers_only_numeric_and_boolean_features(self) -> None:
        calendar, columns = prepare_calendar_frame(make_calendar())
        self.assertEqual(columns, ("es_quincena", "dia_habil"))
        self.assertEqual(calendar.schema["es_quincena"], pl.Float64)
        self.assertNotIn("etiqueta", calendar.columns)
        self.assertEqual(calendar.get_column("fecha").n_unique(), calendar.height)

    def test_local_csv_parquet_loading_and_latest_run_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "20260101_20260101_20260101"
            new = root / "20260201_20260201_20260201"
            old.mkdir()
            new.mkdir()
            frame = make_global_long(num_series=3, length=5)
            frame.write_parquet(old / "global_series_long.parquet")
            frame.write_parquet(new / "global_series_long.parquet")
            frame.write_csv(root / "frame.csv")

            latest = find_latest_global_long_uri(root)
            self.assertEqual(Path(latest).parent.name, new.name)
            self.assertEqual(read_polars_uri(latest).shape, frame.shape)
            self.assertEqual(read_polars_uri(root / "frame.csv").shape, frame.shape)

    def test_factory_fixes_identity_split_and_temporal_holdout_across_windows(self) -> None:
        calendar, columns = prepare_calendar_frame(make_calendar())
        factory = GlobalNotebookDatasetFactory(
            make_global_long(),
            calendar,
            exogenous_columns=columns,
            horizon=2,
            seen_validation_size=4,
            validation_unseen_fraction=0.2,
            test_unseen_fraction=0.2,
            stride=1,
            seed=17,
            max_window_size=5,
        )
        bundle_3 = factory(3)
        bundle_5 = factory(5)

        self.assertEqual(bundle_3.train.series_ids, bundle_5.train.series_ids)
        self.assertEqual(
            bundle_3.validation_unseen.series_ids,
            bundle_5.validation_unseen.series_ids,
        )
        self.assertTrue(
            set(bundle_3.train.series_ids).isdisjoint(
                set(bundle_3.validation_unseen.series_ids)
            )
        )
        frames = factory.build_frames(3)
        for series_id, holdout_start in factory.seen_target_start_dates.items():
            train_max = (
                frames.train.filter(pl.col("cross_key_id") == series_id)
                .get_column("fecha")
                .max()
                .isoformat()
            )
            self.assertLess(train_max, holdout_start)
        first = bundle_3.train[0]
        self.assertEqual(
            tuple(first["model_inputs"]),
            ("y_context", "x_history", "x_future", "x_static"),
        )
        self.assertIn("cross_key_id", first["metadata"])
        self.assertNotIn("cross_key_id", first["model_inputs"])

    def test_load_inputs_validates_real_notebook_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_path = root / "global.parquet"
            calendar_path = root / "calendar.csv"
            make_global_long().write_parquet(global_path)
            make_calendar().write_csv(calendar_path)
            config = GlobalNotebookConfig(
                architecture="rnn_bi",
                global_long_uri=str(global_path),
                calendar_uri=str(calendar_path),
                artifact_root=str(root / "runs"),
                horizon=2,
                seen_validation_size=4,
                n_trials=1,
                max_window_size=3,
            )
            inputs = load_global_inputs(config)
            self.assertEqual(inputs.exogenous_columns, ("es_quincena", "dia_habil"))
            self.assertEqual(inputs.global_long.get_column("fecha").dtype, pl.Date)
            self.assertEqual(inputs.calendar.get_column("fecha").dtype, pl.Date)

    def test_support_factory_runs_through_global_manager(self) -> None:
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
            seed=23,
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
            seed=23,
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
        manager = GlobalManager(
            "mlp",
            base_training_config=training,
            curriculum_config=curriculum,
            candidate_factory=tiny_candidate,
            seed=23,
        )
        manager.fit_global(
            factory,
            n_trials=1,
            split_manifest=factory.split,
            exogenous_columns=columns,
        )
        test_dataset = factory.build_test_unseen(manager.dimensions.window_size)
        metrics = manager.evaluate(test_dataset, device="cpu")
        self.assertTrue(manager.is_fitted)
        self.assertFalse(isinstance(manager.model, dict))
        self.assertGreater(metrics.num_series, 0)
        self.assertEqual(manager.dimensions.exogenous_columns, columns)


if __name__ == "__main__":
    unittest.main()
