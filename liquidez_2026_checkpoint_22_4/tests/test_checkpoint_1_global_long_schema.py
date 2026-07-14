from __future__ import annotations

import json
import unittest
from pathlib import Path

import polars as pl

from global_contracts import GLOBAL_LONG_REQUIRED_COLUMNS
from global_long_schema import (
    build_and_validate_global_long,
    build_global_long,
    validate_global_long,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_legacy_long() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fecha": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "cross_key_id": ["110203MXN"] * 4,
            "tipo_serie": ["saldo", "saldo", "variacion", "variacion"],
            "total_amount": [100.0, 120.0, 10.0, 20.0],
            "total_amount_robust": [0.0, 1.0, -1.0, 1.0],
            "difficulty_score": [0.75] * 4,
            "curriculum_bucket": [15] * 4,
            "grupo": ["Grupo_2"] * 4,
        }
    )


class TestCheckpoint1GlobalLongSchema(unittest.TestCase):
    def test_builds_exact_canonical_schema(self) -> None:
        result = build_global_long(sample_legacy_long())
        self.assertEqual(tuple(result.columns), GLOBAL_LONG_REQUIRED_COLUMNS)
        self.assertEqual(result.height, 4)

    def test_cross_key_is_account_currency_plus_series_type(self) -> None:
        result = build_global_long(sample_legacy_long())
        keys = set(result.get_column("cross_key_id").to_list())
        self.assertEqual(keys, {"110203MXN_saldo", "110203MXN_variacion"})
        self.assertEqual(
            set(result.get_column("account_currency_id").to_list()),
            {"110203MXN"},
        )

    def test_uses_original_target_not_legacy_robust_target(self) -> None:
        result = build_global_long(sample_legacy_long())
        saldo = result.filter(pl.col("cross_key_id") == "110203MXN_saldo")
        self.assertEqual(saldo.get_column("target").to_list(), [100.0, 120.0])

    def test_returns_validation_evidence(self) -> None:
        _, report = build_and_validate_global_long(sample_legacy_long())
        self.assertEqual(report.row_count, 4)
        self.assertEqual(report.series_count, 2)
        self.assertEqual(report.account_currency_count, 1)
        self.assertEqual(report.series_types, ("saldo", "variacion"))
        self.assertEqual(report.min_curriculum_level, 15)
        self.assertEqual(report.max_curriculum_level, 15)

    def test_rejects_duplicated_series_date(self) -> None:
        canonical = build_global_long(sample_legacy_long())
        duplicated = pl.concat([canonical, canonical.head(1)])
        with self.assertRaisesRegex(ValueError, "duplicated cross_key_id/date"):
            validate_global_long(duplicated)

    def test_accepts_arbitrary_non_empty_series_type(self) -> None:
        generic = sample_legacy_long().with_columns(
            pl.when(pl.arange(0, pl.len()) < 2)
            .then(pl.lit("flujo contractual"))
            .otherwise(pl.col("tipo_serie"))
            .alias("tipo_serie")
        )
        result = build_global_long(generic)
        self.assertIn("flujo_contractual", result.get_column("tipo_serie").to_list())
        self.assertIn(
            "110203MXN_flujo_contractual",
            result.get_column("cross_key_id").to_list(),
        )

    def test_rejects_missing_target(self) -> None:
        invalid = sample_legacy_long().with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("total_amount"))
            .alias("total_amount")
        )
        with self.assertRaisesRegex(ValueError, "contains nulls"):
            build_global_long(invalid)

    def test_code_01_exports_canonical_csv_and_parquet(self) -> None:
        notebook = json.loads((ROOT / "code_01.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("build_and_validate_global_long(\n    global_series_long_source\n)", source)
        self.assertIn("global_series_long.csv", source)
        self.assertIn("global_series_long.parquet", source)
        self.assertIn("PATH_GLOBAL_SERIES_LONG_PARQUET", source)

    def test_local_training_pipeline_remains_unchanged(self) -> None:
        for notebook_name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            source = (ROOT / notebook_name).read_text(encoding="utf-8")
            self.assertNotIn("global_long_schema", source)
            self.assertNotIn("GlobalWindowDataset", source)

        scientist_source = (ROOT / "scientist.py").read_text(encoding="utf-8")
        manager_source = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertIn("self.models", scientist_source)
        self.assertIn("for serie", manager_source)


if __name__ == "__main__":
    unittest.main()
