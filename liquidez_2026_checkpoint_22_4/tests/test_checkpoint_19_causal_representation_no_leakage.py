from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import unittest

import numpy as np
import polars as pl
import torch

from global_contracts import MODEL_INPUT_FIELDS
from global_data import ContextScaler, GlobalSeriesSplit, StaticFeatureEncoder
from global_manager import ARTIFACT_SCHEMA_VERSION
from global_models import build_global_model, list_global_models
from global_notebook import GlobalNotebookDatasetFactory, prepare_calendar_frame


ROOT = Path(__file__).resolve().parents[1]


def make_global_long(*, accounts: int = 8, length: int = 18) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    currencies = ("MXN", "USD", "EUR", "GBP")
    for account_number in range(accounts):
        currency = currencies[account_number % len(currencies)]
        account_currency_id = f"ACC{account_number:02d}{currency}"
        for series_type in ("saldo", "variacion"):
            cross_key_id = f"{account_currency_id}_{series_type}"
            for offset in range(length):
                if series_type == "saldo":
                    target = 1_000.0 * (account_number + 1) + 15.0 * offset
                else:
                    target = float(((-1) ** offset) * (account_number + 1) * (offset % 4))
                rows.append(
                    {
                        "fecha": origin + timedelta(days=offset),
                        "account_currency_id": account_currency_id,
                        "divisa": currency,
                        "cross_key_id": cross_key_id,
                        "tipo_serie": series_type,
                        "series_age_step": offset + 1,
                        "target": target,
                        "difficulty_score": 0.0,
                        "nivel_curriculum": 1,
                        "grupo": "Grupo_2",
                    }
                )
    return pl.DataFrame(rows)


def make_calendar(*, length: int = 24, future_spike: float = 1_000_000.0) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    continuous = [float(offset) for offset in range(length)]
    continuous[-1] = float(future_spike)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=offset) for offset in range(length)],
            "es_quincena": [1.0 if offset in {14, 15} else 0.0 for offset in range(length)],
            "monto_calendario": continuous,
        }
    )


def make_factory(global_long: pl.DataFrame, calendar: pl.DataFrame) -> GlobalNotebookDatasetFactory:
    prepared_calendar, columns = prepare_calendar_frame(calendar)
    return GlobalNotebookDatasetFactory(
        global_long,
        prepared_calendar,
        exogenous_columns=columns,
        horizon=2,
        seen_validation_size=4,
        validation_unseen_fraction=0.2,
        test_unseen_fraction=0.2,
        stride=1,
        seed=19,
        max_window_size=3,
    )


class TestCheckpoint19CausalRepresentationNoLeakage(unittest.TestCase):
    def test_target_scaler_is_linear_causal_and_exactly_reversible(self) -> None:
        scaler = ContextScaler()
        context = np.asarray([10.0, 10.0, 10.0, 1_000.0], dtype=np.float64)
        params = scaler.fit(context)
        expected_scale = max(
            float(np.mean(np.abs(context))),
            float(np.mean(np.abs(np.diff(context)))),
            1.0,
        )
        self.assertEqual(params.center, 0.0)
        self.assertAlmostEqual(params.scale, expected_scale)
        self.assertEqual(params.transform, "linear_context_scale")
        self.assertEqual(scaler.contract()["nonlinear_target_transform"], "none")

        future = np.asarray([-5_000.0, 0.0, 25_000.0], dtype=np.float64)
        transformed = scaler.transform(future, params)
        np.testing.assert_allclose(transformed, future / expected_scale)
        np.testing.assert_allclose(scaler.inverse_transform(transformed, params), future)

    def test_static_encoder_is_train_fitted_and_has_unknown_currency_bucket(self) -> None:
        train = make_global_long().filter(pl.col("divisa").is_in(["MXN", "USD"]))
        encoder = StaticFeatureEncoder.fit(train)
        self.assertEqual(encoder.currency_categories, ("MXN", "USD"))
        before_dimension = encoder.dimension
        encoded = encoder.encode(
            series_type="saldo",
            currency="JPY",
            scale=100.0,
            series_age=25,
        )
        self.assertEqual(encoded.shape, (before_dimension,))
        unknown_index = encoder.feature_names.index("divisa=__UNKNOWN__")
        self.assertEqual(float(encoded[unknown_index]), 1.0)
        self.assertEqual(encoder.dimension, before_dimension)
        self.assertTrue(np.all(np.isfinite(encoded)))

    def test_split_keeps_saldo_and_variacion_of_same_account_together(self) -> None:
        split = GlobalSeriesSplit.create(
            make_global_long(),
            validation_unseen_fraction=0.2,
            test_unseen_fraction=0.2,
            seed=19,
        )
        partitions = {
            "train": set(split.train_series),
            "validation_unseen": set(split.validation_unseen_series),
            "test_unseen": set(split.test_unseen_series),
        }
        for account_number in range(8):
            currency = ("MXN", "USD", "EUR", "GBP")[account_number % 4]
            account = f"ACC{account_number:02d}{currency}"
            pair = {f"{account}_saldo", f"{account}_variacion"}
            owners = [name for name, values in partitions.items() if pair & values]
            self.assertEqual(len(owners), 1, (account, owners))
            self.assertTrue(pair.issubset(partitions[owners[0]]))

    def test_validation_and_unseen_targets_do_not_change_train_difficulty(self) -> None:
        original = make_global_long()
        calendar = make_calendar()
        first = make_factory(original, calendar)

        seen_starts = {
            key: date.fromisoformat(value)
            for key, value in first.seen_target_start_dates.items()
        }
        train_ids = set(first.split.train_series)

        perturbed_rows: list[dict[str, object]] = []
        for row in original.iter_rows(named=True):
            row = dict(row)
            key = str(row["cross_key_id"])
            is_seen_holdout = key in train_ids and row["fecha"] >= seen_starts[key]
            is_unseen = key not in train_ids
            if is_seen_holdout or is_unseen:
                row["target"] = float(row["target"]) * 1_000_000.0 + 987_654_321.0
            perturbed_rows.append(row)
        second = make_factory(pl.DataFrame(perturbed_rows), calendar)

        self.assertEqual(first.split, second.split)
        first_manifest = first.difficulty_manifest.sort("cross_key_id")
        second_manifest = second.difficulty_manifest.sort("cross_key_id")
        self.assertEqual(first_manifest.columns, second_manifest.columns)
        for column in first_manifest.columns:
            left = first_manifest[column].to_numpy()
            right = second_manifest[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)
            else:
                np.testing.assert_array_equal(left, right)

    def test_exogenous_statistics_use_train_dates_only_and_binary_stays_identity(self) -> None:
        global_long = make_global_long(length=18)
        calendar = make_calendar(length=24, future_spike=9_999_999_999.0)
        factory = make_factory(global_long, calendar)
        contract = factory.exogenous_contract["features"]
        self.assertEqual(contract["es_quincena"]["mode"], "binary_identity")
        self.assertEqual(contract["es_quincena"]["mean_train"], 0.0)
        self.assertEqual(contract["es_quincena"]["std_train"], 1.0)

        train_frame = factory.build_frames(3).train
        train_dates = set(train_frame["fecha"].unique().to_list())
        expected_values = np.asarray(
            [
                row["monto_calendario"]
                for row in calendar.iter_rows(named=True)
                if row["fecha"] in train_dates
            ],
            dtype=np.float64,
        )
        self.assertAlmostEqual(
            contract["monto_calendario"]["mean_train"],
            float(expected_values.mean()),
        )
        self.assertAlmostEqual(
            contract["monto_calendario"]["std_train"],
            float(expected_values.std()),
        )
        self.assertLess(contract["monto_calendario"]["mean_train"], 100.0)

    def test_dataset_and_all_architectures_use_x_static_without_context_mask(self) -> None:
        factory = make_factory(make_global_long(), make_calendar())
        bundle = factory(3)
        sample = bundle.train[0]
        self.assertEqual(tuple(sample["model_inputs"]), MODEL_INPUT_FIELDS)
        self.assertNotIn("context_mask", sample["model_inputs"])
        self.assertEqual(
            tuple(sample["model_inputs"]["x_static"].shape),
            (bundle.train.static_dim,),
        )
        batch = {
            name: value.unsqueeze(0)
            for name, value in sample["model_inputs"].items()
        }
        base_cfg = {
            "latent_dim": 8,
            "enc_hidden_size": 12,
            "enc_num_layers": 1,
            "dec_hidden_size": 12,
            "dec_num_layers": 1,
            "rnn_hidden_size": 10,
            "rnn_num_layers": 1,
            "decoder_num_layers": 1,
            "dropout_rate": 0.0,
            "activation": "gelu",
            "beta_kl": 0.01,
            "beta_ae": 0.1,
            "ae_hidden_size": 8,
            "ae_num_layers": 1,
        }
        for architecture in list_global_models():
            for use_ae in (False, True):
                cfg = {**base_cfg, "use_auxiliary_autoencoder": use_ae}
                model = build_global_model(
                    architecture,
                    cfg,
                    window_size=3,
                    horizon=2,
                    exogenous_dim=len(factory.exogenous_columns),
                    static_dim=bundle.train.static_dim,
                ).eval()
                with torch.no_grad():
                    output = model(**batch)
                self.assertEqual(tuple(output["y_pred"].shape), (1, 2, 1))
                self.assertTrue(torch.isfinite(output["y_pred"]).all())
                self.assertEqual("context_reconstruction" in output, use_ae)

    def test_code_01_exports_all_eligible_global_series_not_easy_filter_only(self) -> None:
        notebook = json.loads((ROOT / "code_01.ipynb").read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        start = code.index("global_series_long_saldo_source")
        global_section = code[start:]
        self.assertIn("base\n    .select", global_section)
        self.assertNotIn("base_selected\n    .select", global_section)
        self.assertIn("pl.lit(0.0)", global_section)
        self.assertIn("build_and_validate_global_long", global_section)

    def test_artifact_and_notebook_contract_are_checkpoint_19(self) -> None:
        self.assertEqual(ARTIFACT_SCHEMA_VERSION, "1.5")
        notebooks = sorted(ROOT.glob("code_03_GLOBAL_*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertIn('"representation_checkpoint": 19', code)
            self.assertIn('"context_mask_is_model_input": False', code)
            self.assertIn("static_feature_contract.json", code)
            self.assertIn("exogenous_contract.json", code)
            self.assertIn("difficulty_train_only.parquet", code)
            self.assertTrue(
                all(
                    cell.get("execution_count") is None and not cell.get("outputs")
                    for cell in notebook["cells"]
                    if cell.get("cell_type") == "code"
                )
            )


if __name__ == "__main__":
    unittest.main()
