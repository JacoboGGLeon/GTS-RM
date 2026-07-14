from __future__ import annotations

from datetime import date, timedelta

import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader

from global_contracts import HISTORY_EMBEDDING_FIELD
from gtrm_embedding_diagnostics import (
    HistoryEmbeddingDiagnosticsCriteria,
    build_history_embedding_diagnostics,
    embedding_columns,
    validate_history_embedding_frame,
    write_history_embedding_artifacts,
)
from gtrm_representation import collect_history_embeddings


def _model_cfg() -> dict[str, object]:
    return {
        "latent_dim": 4,
        "enc_hidden_size": 8,
        "enc_num_layers": 1,
        "dec_hidden_size": 8,
        "dec_num_layers": 1,
        "rnn_hidden_size": 6,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "use_auxiliary_autoencoder": False,
    }


def _embeddings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cross_key_id": ["A", "A", "B", "B"],
            "cutoff": ["2026-01-02", "2026-01-03", "2026-01-02", "2026-01-03"],
            "tipo_serie": ["saldo", "saldo", "variacion", "variacion"],
            "divisa": ["MXN", "MXN", "USD", "USD"],
            f"{HISTORY_EMBEDDING_FIELD}_000": [0.0, 1.0, 0.0, 2.0],
            f"{HISTORY_EMBEDDING_FIELD}_001": [1.0, 1.0, 2.0, 2.0],
        }
    )


def test_embedding_frame_validation_detects_columns_and_rejects_nonfinite() -> None:
    frame = _embeddings_frame()
    assert embedding_columns(frame) == (
        f"{HISTORY_EMBEDDING_FIELD}_000",
        f"{HISTORY_EMBEDDING_FIELD}_001",
    )
    assert validate_history_embedding_frame(frame) == embedding_columns(frame)
    bad = frame.copy()
    bad.loc[0, f"{HISTORY_EMBEDDING_FIELD}_000"] = float("nan")
    try:
        validate_history_embedding_frame(bad)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite embeddings must be rejected")


def test_build_history_embedding_diagnostics_reports_series_drift_and_cohorts() -> None:
    report = build_history_embedding_diagnostics(
        _embeddings_frame(),
        cohort_columns=("tipo_serie", "divisa"),
    )
    assert report.summary["num_embeddings"] == 4
    assert report.summary["embedding_dim"] == 2
    assert set(report.by_series["cross_key_id"]) == {"A", "B"}
    assert "tipo_serie" in report.by_cohort
    assert "divisa" in report.by_cohort
    assert set(report.dimension_report["embedding_column"]) == set(embedding_columns(_embeddings_frame()))


def test_write_history_embedding_artifacts_is_portable_without_parquet_engine(tmp_path) -> None:
    frame = _embeddings_frame()
    destination = write_history_embedding_artifacts(
        frame,
        tmp_path,
        cohort_columns=("tipo_serie",),
        write_parquet=False,
    )
    assert (destination / "history_embeddings.csv").exists()
    assert (destination / "history_embeddings_schema.json").exists()
    assert (destination / "history_embedding_diagnostics_summary.json").exists()
    assert (destination / "history_embedding_dimension_report.csv").exists()
    assert (destination / "history_embedding_by_series_summary.csv").exists()
    assert (destination / "history_embedding_by_tipo_serie.csv").exists()


@torch.no_grad()
def test_collect_then_diagnose_real_model_history_embeddings() -> None:
    pl = pytest.importorskip("polars")
    from global_data import GlobalWindowDataset
    from global_models import build_global_model

    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for number in range(3):
        currency = "MXN" if number % 2 == 0 else "USD"
        series_type = "saldo" if number % 2 == 0 else "variacion"
        account = f"ACC{number:02d}{currency}"
        for offset in range(8):
            rows.append({
                "fecha": origin + timedelta(days=offset),
                "account_currency_id": account,
                "divisa": currency,
                "cross_key_id": f"{account}_{series_type}",
                "tipo_serie": series_type,
                "series_age_step": offset + 1,
                "target": float(5 + number * 2 + offset),
                "difficulty_score": float(number % 3) / 3.0,
                "nivel_curriculum": 1 + (number % 2),
                "grupo": "Grupo_2" if number != 2 else "Grupo_3",
            })
    dataset = GlobalWindowDataset(pl.DataFrame(rows), window_size=3, horizon=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = build_global_model(
        "mlp",
        _model_cfg(),
        window_size=dataset.window_size,
        horizon=dataset.horizon,
        exogenous_dim=len(dataset.exogenous_columns),
        static_dim=dataset.static_dim,
    )
    embeddings = collect_history_embeddings(model, loader, max_batches=2)
    validate_history_embedding_frame(
        embeddings,
        criteria=HistoryEmbeddingDiagnosticsCriteria(min_embedding_dim=4),
    )
    diagnostics = build_history_embedding_diagnostics(
        embeddings,
        cohort_columns=("tipo_serie", "divisa", "grupo"),
    )
    assert diagnostics.summary["embedding_dim"] == 4
    assert diagnostics.summary["num_embeddings"] == len(embeddings)
    assert not diagnostics.by_series.empty
