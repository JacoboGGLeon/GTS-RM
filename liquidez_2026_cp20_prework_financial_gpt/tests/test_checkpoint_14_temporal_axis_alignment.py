from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import unittest

import pandas as pd
import polars as pl
import torch

from global_data import StaticFeatureEncoder
from global_models import GLOBAL_OUTPUT_FIELD, GlobalForecastModel, GlobalModelDimensions
from global_monitoring import MCDropoutConfig, forecast_future_mc
from temporal_axis import (
    ForecastRequest,
    InsufficientFutureContextError,
    TemporalAxis,
    TemporalWindowAligner,
)


ROOT = Path(__file__).resolve().parents[1]


class ZeroModel(GlobalForecastModel):
    def __init__(self, window_size: int, horizon: int, exogenous_dim: int, static_dim: int) -> None:
        super().__init__(GlobalModelDimensions(window_size, horizon, exogenous_dim, static_dim))
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, y_context, x_history, x_future, x_static):
        self._validate_and_prepare(y_context, x_history, x_future, x_static)
        shape = (y_context.shape[0], self.dimensions.horizon, 1)
        return {GLOBAL_OUTPUT_FIELD: torch.zeros(shape, device=y_context.device) + self.anchor}


def global_long(dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame({
        "fecha": dates,
        "account_currency_id": ["ACC_MXN"] * len(dates),
        "cross_key_id": ["ACC_MXN_saldo"] * len(dates),
        "tipo_serie": ["saldo"] * len(dates),
        "target": [float(i + 1) for i in range(len(dates))],
        "difficulty_score": [0.5] * len(dates),
        "nivel_curriculum": [1] * len(dates),
        "grupo": ["Grupo_2"] * len(dates),
    })


class TestCheckpoint14TemporalAxisAlignment(unittest.TestCase):
    def test_axis_uses_provider_rows_without_frequency_assumptions(self) -> None:
        calendar = pl.DataFrame({
            "fecha": [date(2026, 1, 2), date(2026, 1, 5), date(2026, 2, 2)],
            "feature": [1.0, 2.0, 3.0],
        })
        axis = TemporalAxis.from_frame(calendar, feature_columns=("feature",))
        selected = axis.after(date(2026, 1, 2), n_steps=2)
        self.assertEqual(
            list(selected),
            [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")],
        )
        ranged = axis.resolve(
            ForecastRequest(start="2026-01-03", end="2026-02-02"),
            anchor="2026-01-02",
        )
        self.assertEqual(list(ranged), list(selected))

    def test_alignment_is_non_destructive_and_auditable(self) -> None:
        source = global_long([
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 3),
            date(2026, 1, 5),
        ])
        calendar = pl.DataFrame({
            "fecha": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 5)],
            "feature": [0.0, 1.0, 2.0],
        })
        axis = TemporalAxis.from_frame(calendar, feature_columns=("feature",))
        aligned, report = TemporalWindowAligner(axis).align_global_long(source)
        self.assertEqual(source.height, 4)
        self.assertEqual(aligned.height, 3)
        row = report.row(0, named=True)
        self.assertEqual(row["excluded_rows"], 1)
        self.assertAlmostEqual(row["coverage_ratio"], 0.75)
        self.assertIn("feature", aligned.columns)

    def test_future_forecast_uses_next_axis_steps_not_calendar_days(self) -> None:
        observed = global_long([date(2026, 5, 28), date(2026, 5, 29)])
        calendar = pl.DataFrame({
            "fecha": [
                date(2026, 5, 28),
                date(2026, 5, 29),
                date(2026, 6, 1),
                date(2026, 6, 2),
            ],
            "feature": [0.0, 1.0, 2.0, 3.0],
        })
        static_encoder = StaticFeatureEncoder(("saldo", "variacion"), ("MXN",))
        model = ZeroModel(window_size=2, horizon=2, exogenous_dim=1, static_dim=static_encoder.dimension)
        results, consolidated = forecast_future_mc(
            model,
            observed,
            calendar,
            window_size=2,
            horizon=2,
            exogenous_columns=("feature",),
            static_feature_encoder=static_encoder,
            n_steps=2,
            config=MCDropoutConfig(n_mc=2, batch_size=2, device="cpu"),
        )
        self.assertIn("ACC_MXN_saldo", results)
        self.assertEqual(
            list(pd.to_datetime(consolidated["date"])),
            [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")],
        )

    def test_insufficient_future_context_has_explicit_error(self) -> None:
        calendar = pl.DataFrame({
            "fecha": [date(2026, 1, 1), date(2026, 1, 2)],
            "feature": [0.0, 1.0],
        })
        axis = TemporalAxis.from_frame(calendar, feature_columns=("feature",))
        with self.assertRaisesRegex(InsufficientFutureContextError, "requested_steps=2"):
            axis.after(date(2026, 1, 1), n_steps=2)

    def test_four_global_notebooks_use_temporal_steps_and_reports(self) -> None:
        notebooks = sorted(ROOT.glob("code_03_GLOBAL_*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for notebook_path in notebooks:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertIn("n_steps=HORIZON", code)
            self.assertIn("temporal_alignment_report.parquet", code)
            self.assertIn("backtest_run_report.json", code)
            self.assertNotIn("fc_start + pd.Timedelta(days=HORIZON - 1)", code)
            self.assertTrue(all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ))


if __name__ == "__main__":
    unittest.main()
