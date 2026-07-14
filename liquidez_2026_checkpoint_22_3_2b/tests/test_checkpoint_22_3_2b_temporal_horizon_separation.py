from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import torch

from global_data import StaticFeatureEncoder
from global_models import GLOBAL_OUTPUT_FIELD, GlobalForecastModel, GlobalModelDimensions
from global_monitoring import MCDropoutConfig, forecast_future_mc
from global_notebook import GlobalNotebookConfig
from global_pipeline import ForecastRequest
from global_surface_config import TemporalForecastConfig
from gtrm_config import GTRMModelConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted(ROOT.glob("code_03_GLOBAL_*.ipynb")))


class CountingZeroModel(GlobalForecastModel):
    def __init__(self, *, window_size: int, chunk_size: int, exogenous_dim: int, static_dim: int) -> None:
        super().__init__(
            GlobalModelDimensions(window_size, chunk_size, exogenous_dim, static_dim)
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def forward(self, y_context, x_history, x_future, x_static):
        self._validate_and_prepare(y_context, x_history, x_future, x_static)
        self.forward_calls += 1
        shape = (y_context.shape[0], self.dimensions.horizon, 1)
        return {
            GLOBAL_OUTPUT_FIELD: torch.zeros(shape, device=y_context.device)
            + self.anchor
        }


def _long_frame(observed_dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fecha": observed_dates,
            "account_currency_id": ["ACC_MXN"] * len(observed_dates),
            "cross_key_id": ["ACC_MXN_saldo"] * len(observed_dates),
            "tipo_serie": ["saldo"] * len(observed_dates),
            "target": [float(index + 1) for index in range(len(observed_dates))],
            "difficulty_score": [0.5] * len(observed_dates),
            "nivel_curriculum": [1] * len(observed_dates),
            "grupo": ["Grupo_2"] * len(observed_dates),
        }
    )


def test_temporal_contract_separates_total_horizon_chunk_and_stride() -> None:
    contract = TemporalForecastConfig(
        forecast_horizon=25,
        rollout_chunk_size=3,
        training_stride=1,
    )
    assert contract.forecast_horizon == 25
    assert contract.rollout_chunk_size == 3
    assert contract.training_stride == 1
    assert contract.rollout_blocks == 9

    with pytest.raises(ValueError, match="cannot exceed"):
        TemporalForecastConfig(
            forecast_horizon=2,
            rollout_chunk_size=3,
            training_stride=1,
        )


def test_notebook_config_persists_both_temporal_quantities() -> None:
    config = GlobalNotebookConfig(
        architecture="mlp",
        global_long_uri="s3://bucket/global.parquet",
        calendar_uri="s3://bucket/calendar.csv",
        artifact_root="./run",
        horizon=3,
        forecast_horizon=25,
        seen_validation_size=50,
        stride=1,
        model_config=GTRMModelConfig(architecture="mlp"),
    )
    config.validate()
    payload = config.to_dict()
    assert config.rollout_chunk_size == 3
    assert payload["horizon"] == 3
    assert payload["rollout_chunk_size"] == 3
    assert payload["forecast_horizon"] == 25



def test_forecast_request_enforces_total_horizon_cap() -> None:
    request = ForecastRequest(n_steps=25, max_steps=25)
    assert request.n_steps == 25
    assert request.max_steps == 25

    with pytest.raises(ValueError, match="cannot exceed"):
        ForecastRequest(n_steps=26, max_steps=25)

def test_forecast_repeats_three_step_chunks_until_total_horizon() -> None:
    observed_dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(5)]
    future_dates = [date(2026, 1, 6) + timedelta(days=i) for i in range(7)]
    calendar_dates = observed_dates + future_dates + [
        date(2026, 1, 13),
        date(2026, 1, 14),
    ]
    calendar = pl.DataFrame(
        {
            "fecha": calendar_dates,
            "feature": [float(i) for i in range(len(calendar_dates))],
        }
    )
    static_encoder = StaticFeatureEncoder(("saldo", "variacion"), ("MXN",))
    model = CountingZeroModel(
        window_size=3,
        chunk_size=3,
        exogenous_dim=1,
        static_dim=static_encoder.dimension,
    )

    results, consolidated = forecast_future_mc(
        model,
        _long_frame(observed_dates),
        calendar,
        window_size=3,
        horizon=3,
        exogenous_columns=("feature",),
        static_feature_encoder=static_encoder,
        n_steps=7,
        config=MCDropoutConfig(n_mc=1, batch_size=8, device="cpu"),
    )

    assert model.forward_calls == 3
    assert len(results["ACC_MXN_saldo"]) == 7
    assert len(consolidated) == 7
    assert list(pd.to_datetime(consolidated["date"])) == list(
        pd.to_datetime(future_dates)
    )



def test_explicit_date_range_cannot_exceed_total_horizon_cap() -> None:
    observed_dates = [date(2026, 2, 1) + timedelta(days=i) for i in range(4)]
    future_dates = [date(2026, 2, 5) + timedelta(days=i) for i in range(5)]
    calendar_dates = observed_dates + future_dates
    calendar = pl.DataFrame(
        {
            "fecha": calendar_dates,
            "feature": [float(i) for i in range(len(calendar_dates))],
        }
    )
    static_encoder = StaticFeatureEncoder(("saldo", "variacion"), ("MXN",))
    model = CountingZeroModel(
        window_size=3,
        chunk_size=2,
        exogenous_dim=1,
        static_dim=static_encoder.dimension,
    )

    with pytest.raises(ValueError, match="exceeds max_steps"):
        forecast_future_mc(
            model,
            _long_frame(observed_dates),
            calendar,
            window_size=3,
            horizon=2,
            exogenous_columns=("feature",),
            static_feature_encoder=static_encoder,
            start_date="2026-02-05",
            end_date="2026-02-09",
            max_steps=3,
            config=MCDropoutConfig(n_mc=1, batch_size=8, device="cpu"),
        )

def test_four_notebooks_explain_and_connect_temporal_controls() -> None:
    assert len(NOTEBOOKS) == 4
    for notebook_path in NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "FORECAST_HORIZON = 25" in code
        assert "ROLLOUT_CHUNK_SIZE = 3" in code
        assert "TRAINING_STRIDE = 1" in code
        assert "horizon=temporal_config.rollout_chunk_size" in code
        assert "forecast_horizon=temporal_config.forecast_horizon" in code
        assert "n_steps=temporal_config.forecast_horizon" in code
        assert "rollout_chunk_size" in markdown
        assert "forecast_horizon" in markdown
