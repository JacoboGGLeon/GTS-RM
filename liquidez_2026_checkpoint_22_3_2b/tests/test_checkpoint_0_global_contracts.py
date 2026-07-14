from __future__ import annotations

import json
import unittest
from pathlib import Path

from global_contracts import (
    CROSS_KEY_COLUMN,
    DEFAULT_GLOBAL_CONTRACT,
    GLOBAL_LONG_REQUIRED_COLUMNS,
    MODEL_INPUT_FIELDS,
    SUPPORTED_ARCHITECTURES,
    canonical_cross_key,
    validate_global_long_columns,
    validate_model_input_fields,
)


ROOT = Path(__file__).resolve().parents[1]


class TestCheckpoint0GlobalContracts(unittest.TestCase):
    def test_supported_architectures_match_current_bundle(self) -> None:
        self.assertEqual(
            SUPPORTED_ARCHITECTURES,
            ("mlp", "mlp_vae", "rnn", "rnn_bi"),
        )

    def test_cross_key_is_account_currency_plus_series_type(self) -> None:
        self.assertEqual(canonical_cross_key("110203MXP", "saldo"), "110203MXP_saldo")
        self.assertEqual(
            canonical_cross_key("110203MXP", "variacion"),
            "110203MXP_variacion",
        )
        self.assertEqual(
            canonical_cross_key("CONTRATO_42", "flujo contractual"),
            "CONTRATO_42_flujo_contractual",
        )

    def test_cross_key_never_enters_model_inputs(self) -> None:
        self.assertNotIn(CROSS_KEY_COLUMN, MODEL_INPUT_FIELDS)
        validate_model_input_fields(MODEL_INPUT_FIELDS)
        with self.assertRaises(ValueError):
            validate_model_input_fields((*MODEL_INPUT_FIELDS, CROSS_KEY_COLUMN))

    def test_default_contract_is_valid(self) -> None:
        DEFAULT_GLOBAL_CONTRACT.validate()
        self.assertTrue(set(DEFAULT_GLOBAL_CONTRACT.model_inputs).isdisjoint(
            DEFAULT_GLOBAL_CONTRACT.metadata_fields
        ))

    def test_canonical_long_schema_is_complete(self) -> None:
        self.assertEqual(
            validate_global_long_columns(GLOBAL_LONG_REQUIRED_COLUMNS),
            GLOBAL_LONG_REQUIRED_COLUMNS,
        )

    def test_code_01_already_exports_long_baseline(self) -> None:
        notebook = json.loads((ROOT / "code_01.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("series_long.csv", source)
        self.assertIn("series_long.parquet", source)
        self.assertIn("tipo_serie", source)
        self.assertIn("difficulty_score", source)

    def test_local_pipeline_is_untouched_in_checkpoint_0(self) -> None:
        scientist_source = (ROOT / "scientist.py").read_text(encoding="utf-8")
        manager_source = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertIn("self.models", scientist_source)
        self.assertIn("for serie", manager_source)


if __name__ == "__main__":
    unittest.main()
