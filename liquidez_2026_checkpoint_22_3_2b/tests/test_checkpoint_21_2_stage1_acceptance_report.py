from __future__ import annotations

import json

import pandas as pd

from gtrm_acceptance import (
    Stage1AcceptanceCriteria,
    build_stage1_acceptance_report,
)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cross_key_id": "s1", "candidate_id": "GLOBAL_RNN_E_D", "family": "global", "architecture": "rnn", "MASE": 0.80, "WMAPE": 0.10, "tipo_serie": "saldo"},
            {"cross_key_id": "s1", "candidate_id": "NAIVE_LAST_VALUE", "family": "baseline", "architecture": "last", "MASE": 1.00, "WMAPE": 0.11, "tipo_serie": "saldo"},
            {"cross_key_id": "s2", "candidate_id": "GLOBAL_RNN_E_D", "family": "global", "architecture": "rnn", "MASE": 1.10, "WMAPE": 0.20, "tipo_serie": "saldo"},
            {"cross_key_id": "s2", "candidate_id": "NAIVE_LAST_VALUE", "family": "baseline", "architecture": "last", "MASE": 1.20, "WMAPE": 0.21, "tipo_serie": "saldo"},
            {"cross_key_id": "s3", "candidate_id": "GLOBAL_RNN_E_D", "family": "global", "architecture": "rnn", "MASE": 0.90, "WMAPE": 0.30, "tipo_serie": "variacion"},
            {"cross_key_id": "s3", "candidate_id": "NAIVE_ZERO", "family": "baseline", "architecture": "zero", "MASE": 1.10, "WMAPE": 0.31, "tipo_serie": "variacion"},
            {"cross_key_id": "s4", "candidate_id": "GLOBAL_RNN_E_D", "family": "global", "architecture": "rnn", "MASE": 1.05, "WMAPE": 0.36, "tipo_serie": "variacion"},
            {"cross_key_id": "s4", "candidate_id": "NAIVE_ZERO", "family": "baseline", "architecture": "zero", "MASE": 1.00, "WMAPE": 0.35, "tipo_serie": "variacion"},
        ]
    )


def test_stage1_acceptance_report_prioritizes_percent_series_improved() -> None:
    report = build_stage1_acceptance_report(
        _fixture(),
        criteria=Stage1AcceptanceCriteria(min_percent_series_improved=55.0),
        cohort_columns=("tipo_serie",),
    )
    assert report.summary.accepted is True
    assert report.summary.num_series == 4
    assert report.summary.percent_series_improved == 75.0
    assert report.per_series["improved"].tolist().count(True) == 3
    assert "tipo_serie" in report.by_cohort


def test_stage1_acceptance_report_fails_when_improvement_is_too_low() -> None:
    report = build_stage1_acceptance_report(
        _fixture(),
        criteria=Stage1AcceptanceCriteria(min_percent_series_improved=80.0),
    )
    assert report.summary.accepted is False
    assert "percent_series_improved" in report.summary.reason


def test_stage1_acceptance_report_writes_auditable_files(tmp_path) -> None:
    report = build_stage1_acceptance_report(_fixture(), cohort_columns=("tipo_serie",))
    output_dir = report.write(tmp_path)
    summary = json.loads((output_dir / "stage1_acceptance_summary.json").read_text())
    assert summary["num_series"] == 4
    assert (output_dir / "stage1_acceptance_by_series.csv").exists()
    assert (output_dir / "stage1_acceptance_by_tipo_serie.csv").exists()


def test_stage1_acceptance_report_requires_baselines() -> None:
    frame = _fixture()
    frame = frame[frame["family"] != "baseline"]
    try:
        build_stage1_acceptance_report(frame)
    except ValueError as exc:
        assert "baseline" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing baselines must fail")
