from __future__ import annotations

import json

import pandas as pd

from gtrm_static_ablation import (
    StaticContextAblationCriteria,
    build_static_context_ablation_report,
)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cross_key_id": "s1", "candidate_id": "GTRM_A_RNN", "family": "global", "architecture": "rnn", "use_static_context": False, "MASE": 1.00, "WMAPE": 0.20, "tipo_serie": "saldo"},
            {"cross_key_id": "s1", "candidate_id": "GTRM_B_RNN", "family": "global", "architecture": "rnn", "use_static_context": True, "MASE": 0.80, "WMAPE": 0.18, "tipo_serie": "saldo"},
            {"cross_key_id": "s2", "candidate_id": "GTRM_A_RNN", "family": "global", "architecture": "rnn", "use_static_context": False, "MASE": 1.20, "WMAPE": 0.22, "tipo_serie": "saldo"},
            {"cross_key_id": "s2", "candidate_id": "GTRM_B_RNN", "family": "global", "architecture": "rnn", "use_static_context": True, "MASE": 1.10, "WMAPE": 0.21, "tipo_serie": "saldo"},
            {"cross_key_id": "s3", "candidate_id": "GTRM_A_RNN", "family": "global", "architecture": "rnn", "use_static_context": False, "MASE": 0.95, "WMAPE": 0.30, "tipo_serie": "variacion"},
            {"cross_key_id": "s3", "candidate_id": "GTRM_B_RNN", "family": "global", "architecture": "rnn", "use_static_context": True, "MASE": 0.90, "WMAPE": 0.29, "tipo_serie": "variacion"},
            {"cross_key_id": "s4", "candidate_id": "GTRM_A_RNN", "family": "global", "architecture": "rnn", "use_static_context": False, "MASE": 1.00, "WMAPE": 0.34, "tipo_serie": "variacion"},
            {"cross_key_id": "s4", "candidate_id": "GTRM_B_RNN", "family": "global", "architecture": "rnn", "use_static_context": True, "MASE": 1.05, "WMAPE": 0.34, "tipo_serie": "variacion"},
            # Baselines are allowed in the same monitor export but ignored here.
            {"cross_key_id": "s1", "candidate_id": "NAIVE_LAST", "family": "baseline", "use_static_context": False, "MASE": 1.10, "WMAPE": 0.21, "tipo_serie": "saldo"},
        ]
    )


def test_static_context_ablation_accepts_when_static_improves_majority() -> None:
    report = build_static_context_ablation_report(
        _fixture(),
        criteria=StaticContextAblationCriteria(min_percent_series_improved=50.0),
        cohort_columns=("tipo_serie",),
    )
    assert report.summary.accepted is True
    assert report.summary.percent_series_improved_by_static == 75.0
    assert report.summary.recommendation == "keep_use_static_context_true_as_stage1_default"
    assert report.per_series["improved_by_static"].tolist().count(True) == 3
    assert "tipo_serie" in report.by_cohort


def test_static_context_ablation_fails_when_threshold_is_too_high() -> None:
    report = build_static_context_ablation_report(
        _fixture(),
        criteria=StaticContextAblationCriteria(min_percent_series_improved=90.0),
    )
    assert report.summary.accepted is False
    assert "percent_series_improved_by_static" in report.summary.reason


def test_static_context_ablation_writes_auditable_files(tmp_path) -> None:
    report = build_static_context_ablation_report(_fixture(), cohort_columns=("tipo_serie",))
    output_dir = report.write(tmp_path)
    summary = json.loads((output_dir / "static_context_ablation_summary.json").read_text())
    assert summary["num_series"] == 4
    assert (output_dir / "static_context_ablation_criteria.json").exists()
    assert (output_dir / "static_context_ablation_by_series.csv").exists()
    assert (output_dir / "static_context_ablation_by_tipo_serie.csv").exists()


def test_static_context_ablation_accepts_string_flags() -> None:
    frame = _fixture().copy()
    frame["use_static_context"] = frame["use_static_context"].map({True: "GTRM-B", False: "GTRM-A"})
    report = build_static_context_ablation_report(frame)
    assert report.summary.num_series == 4
    assert report.summary.percent_series_improved_by_static == 75.0


def test_static_context_ablation_requires_both_variants() -> None:
    frame = _fixture()
    frame = frame[frame["use_static_context"] == True]
    try:
        build_static_context_ablation_report(frame)
    except ValueError as exc:
        assert "False" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing no-static variant must fail")
