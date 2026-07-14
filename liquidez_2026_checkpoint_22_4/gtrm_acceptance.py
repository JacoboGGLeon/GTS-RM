"""Stage 1 acceptance report for GTRM.

Checkpoint 21.2 does not train or change model behavior. It turns the monitor
outputs into a deterministic acceptance gate for the Global Representation Base:

- compare global candidates against baseline candidates by series;
- prioritize individual-series accuracy instead of only aggregate WMAPE;
- report macro MASE, WMAPE, P90 error and percent of improved series;
- optionally slice the same decision by cohorts such as tipo_serie, divisa,
  grupo or nivel_curriculum.

The module is deliberately pandas/stdlib based so it can run in lightweight CI
environments where Polars is not installed. It accepts either a pandas DataFrame
or a list of dictionaries, making it compatible with ``financial_gpt_monitor``
exports and small unit-test fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from global_contracts import CROSS_KEY_COLUMN


DEFAULT_STAGE1_THRESHOLDS: Mapping[str, float] = {
    # Must improve a reasonable majority of individual series vs the best naive
    # or seasonal baseline available for that series.
    "min_percent_series_improved": 55.0,
    # Accept a tiny relative tolerance to avoid declaring regressions due only
    # to numeric noise in repeated monitor runs.
    "max_macro_mase_relative_regression": 0.0,
    "max_p90_relative_regression": 0.0,
    # WMAPE is secondary: it must not materially degrade while per-series macro
    # metrics improve. A small default tolerance avoids rejecting useful models
    # because a few high-volume series shift marginally.
    "max_wmape_relative_regression": 0.02,
}

_ACCEPTED_MODEL_FAMILIES = {"model", "global", "gtrm", "financial_gpt"}
_ACCEPTED_BASELINE_FAMILIES = {"baseline", "naive"}


@dataclass(frozen=True)
class Stage1AcceptanceCriteria:
    """Business gate for the Stage 1 Global Representation Base.

    All relative regressions are measured vs the best baseline per series using
    the same primary metric. Values are proportions, so ``0.02`` means 2%.
    """

    primary_metric: str = "MASE"
    wmape_metric: str = "WMAPE"
    min_percent_series_improved: float = 55.0
    max_macro_mase_relative_regression: float = 0.0
    max_p90_relative_regression: float = 0.0
    max_wmape_relative_regression: float = 0.02
    min_series: int = 1

    def validate(self) -> None:
        if not str(self.primary_metric).strip():
            raise ValueError("primary_metric must not be empty")
        if not str(self.wmape_metric).strip():
            raise ValueError("wmape_metric must not be empty")
        if not 0.0 <= float(self.min_percent_series_improved) <= 100.0:
            raise ValueError("min_percent_series_improved must be between 0 and 100")
        for name in (
            "max_macro_mase_relative_regression",
            "max_p90_relative_regression",
            "max_wmape_relative_regression",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite value")
        if isinstance(self.min_series, bool) or int(self.min_series) <= 0:
            raise ValueError("min_series must be a positive integer")

    @classmethod
    def default(cls) -> "Stage1AcceptanceCriteria":
        return cls(**DEFAULT_STAGE1_THRESHOLDS)


@dataclass(frozen=True)
class Stage1AcceptanceSummary:
    """Top-level acceptance decision."""

    accepted: bool
    reason: str
    num_series: int
    percent_series_improved: float
    macro_model_metric: float
    macro_baseline_metric: float
    macro_relative_delta: float
    p90_model_metric: float
    p90_baseline_metric: float
    p90_relative_delta: float
    wmape_model: float
    wmape_baseline: float
    wmape_relative_delta: float
    primary_metric: str = "MASE"
    wmape_metric: str = "WMAPE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1AcceptanceReport:
    """Full report with per-series and optional cohort diagnostics."""

    summary: Stage1AcceptanceSummary
    criteria: Stage1AcceptanceCriteria
    per_series: pd.DataFrame
    by_cohort: Mapping[str, pd.DataFrame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "criteria": asdict(self.criteria),
            "by_cohort": {
                name: frame.to_dict(orient="records")
                for name, frame in self.by_cohort.items()
            },
        }

    def write(self, output_directory: str | Path) -> Path:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "stage1_acceptance_summary.json").write_text(
            json.dumps(self.summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (destination / "stage1_acceptance_criteria.json").write_text(
            json.dumps(asdict(self.criteria), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.per_series.to_csv(destination / "stage1_acceptance_by_series.csv", index=False)
        for name, frame in self.by_cohort.items():
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
            frame.to_csv(destination / f"stage1_acceptance_by_{safe_name}.csv", index=False)
        return destination


def build_stage1_acceptance_report(
    metrics_by_series: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    criteria: Stage1AcceptanceCriteria | None = None,
    cohort_columns: Sequence[str] = (),
) -> Stage1AcceptanceReport:
    """Build an acceptance report from model/baseline metrics by series.

    Expected columns include:

    - ``cross_key_id`` or ``serie``;
    - ``candidate_id``;
    - ``family`` with values like ``model``/``global`` and ``baseline``;
    - the primary metric, by default ``MASE``;
    - ``WMAPE`` when available.

    The function selects the best model and best baseline per series using the
    primary metric, then compares them series by series.
    """

    selected_criteria = criteria or Stage1AcceptanceCriteria.default()
    selected_criteria.validate()

    frame = _normalize_metrics_frame(metrics_by_series)
    primary_metric = _resolve_metric_column(frame, selected_criteria.primary_metric)
    wmape_metric = _resolve_metric_column(frame, selected_criteria.wmape_metric, required=False)
    _validate_acceptance_input(frame, primary_metric=primary_metric)

    model_frame = frame[frame["family_normalized"].isin(_ACCEPTED_MODEL_FAMILIES)].copy()
    baseline_frame = frame[frame["family_normalized"].isin(_ACCEPTED_BASELINE_FAMILIES)].copy()
    if model_frame.empty:
        raise ValueError("Acceptance report requires at least one model/global candidate")
    if baseline_frame.empty:
        raise ValueError("Acceptance report requires at least one baseline candidate")

    best_model = _best_by_series(model_frame, primary_metric, prefix="model")
    best_baseline = _best_by_series(baseline_frame, primary_metric, prefix="baseline")
    comparison = best_model.merge(best_baseline, on=CROSS_KEY_COLUMN, how="inner")
    if len(comparison) < int(selected_criteria.min_series):
        raise ValueError(
            "Not enough series with both model and baseline metrics: "
            f"{len(comparison)} < {selected_criteria.min_series}"
        )

    comparison["improved"] = comparison[f"model_{primary_metric}"] < comparison[f"baseline_{primary_metric}"]
    comparison["metric_delta"] = comparison[f"model_{primary_metric}"] - comparison[f"baseline_{primary_metric}"]
    comparison["metric_relative_delta"] = _relative_delta(
        comparison[f"model_{primary_metric}"],
        comparison[f"baseline_{primary_metric}"],
    )

    if wmape_metric is not None:
        comparison["wmape_relative_delta"] = _relative_delta(
            comparison[f"model_{wmape_metric}"],
            comparison[f"baseline_{wmape_metric}"],
        )
    else:
        comparison["wmape_relative_delta"] = np.nan

    summary = _summarize_acceptance(
        comparison,
        criteria=selected_criteria,
        primary_metric=primary_metric,
        wmape_metric=wmape_metric,
    )
    by_cohort = _cohort_reports(
        frame,
        comparison,
        cohort_columns=cohort_columns,
        primary_metric=primary_metric,
        wmape_metric=wmape_metric,
    )
    return Stage1AcceptanceReport(
        summary=summary,
        criteria=selected_criteria,
        per_series=comparison.sort_values(CROSS_KEY_COLUMN).reset_index(drop=True),
        by_cohort=by_cohort,
    )


def _normalize_metrics_frame(
    metrics_by_series: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frame = (
        metrics_by_series.copy()
        if isinstance(metrics_by_series, pd.DataFrame)
        else pd.DataFrame(list(metrics_by_series))
    )
    if frame.empty:
        raise ValueError("metrics_by_series must not be empty")
    if CROSS_KEY_COLUMN not in frame.columns:
        if "serie" not in frame.columns:
            raise ValueError(f"metrics must contain {CROSS_KEY_COLUMN!r} or 'serie'")
        frame[CROSS_KEY_COLUMN] = frame["serie"].astype(str)
    if "candidate_id" not in frame.columns:
        raise ValueError("metrics must contain candidate_id")
    if "family" not in frame.columns:
        raise ValueError("metrics must contain family")
    frame[CROSS_KEY_COLUMN] = frame[CROSS_KEY_COLUMN].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["family"] = frame["family"].astype(str)
    frame["family_normalized"] = frame["family"].str.lower().str.strip()
    return frame


def _resolve_metric_column(frame: pd.DataFrame, requested: str, *, required: bool = True) -> str | None:
    aliases = {
        requested,
        requested.upper(),
        requested.lower(),
        requested.capitalize(),
    }
    if requested.upper() == "MASE":
        aliases.update({"MASE", "robust_mase", "robust_macro_mase"})
    if requested.upper() == "WMAPE":
        aliases.update({"WMAPE", "raw_wmape", "raw_macro_wmape"})
    for candidate in aliases:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"metrics are missing required metric column {requested!r}")
    return None


def _validate_acceptance_input(frame: pd.DataFrame, *, primary_metric: str) -> None:
    values = pd.to_numeric(frame[primary_metric], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Metric {primary_metric!r} must be finite for all candidates")
    if (values < 0).any():
        raise ValueError(f"Metric {primary_metric!r} must be non-negative")
    frame[primary_metric] = values.astype(float)
    for optional in ("WMAPE", "raw_wmape", "raw_macro_wmape"):
        if optional in frame.columns:
            frame[optional] = pd.to_numeric(frame[optional], errors="coerce")


def _best_by_series(frame: pd.DataFrame, primary_metric: str, *, prefix: str) -> pd.DataFrame:
    sorted_frame = frame.sort_values([CROSS_KEY_COLUMN, primary_metric, "candidate_id"])
    keep_columns = [CROSS_KEY_COLUMN, "candidate_id", "family", primary_metric]
    for optional in ("architecture", "WMAPE", "raw_wmape", "raw_macro_wmape"):
        if optional in sorted_frame.columns and optional not in keep_columns:
            keep_columns.append(optional)
    best = sorted_frame.groupby(CROSS_KEY_COLUMN, as_index=False).first()[keep_columns]
    return best.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in keep_columns
            if column != CROSS_KEY_COLUMN
        }
    )


def _relative_delta(model: pd.Series, baseline: pd.Series) -> pd.Series:
    baseline_safe = baseline.abs().clip(lower=1e-12)
    return (model - baseline) / baseline_safe


def _summarize_acceptance(
    comparison: pd.DataFrame,
    *,
    criteria: Stage1AcceptanceCriteria,
    primary_metric: str,
    wmape_metric: str | None,
) -> Stage1AcceptanceSummary:
    num_series = int(len(comparison))
    percent_improved = float(100.0 * comparison["improved"].mean())
    model_values = comparison[f"model_{primary_metric}"].to_numpy(dtype=float)
    baseline_values = comparison[f"baseline_{primary_metric}"].to_numpy(dtype=float)
    macro_model = float(np.mean(model_values))
    macro_baseline = float(np.mean(baseline_values))
    macro_relative_delta = float((macro_model - macro_baseline) / max(abs(macro_baseline), 1e-12))
    p90_model = float(np.percentile(model_values, 90))
    p90_baseline = float(np.percentile(baseline_values, 90))
    p90_relative_delta = float((p90_model - p90_baseline) / max(abs(p90_baseline), 1e-12))
    if wmape_metric is None:
        wmape_model = math.nan
        wmape_baseline = math.nan
        wmape_relative_delta = math.nan
        wmape_pass = True
    else:
        wmape_model_values = comparison[f"model_{wmape_metric}"].to_numpy(dtype=float)
        wmape_baseline_values = comparison[f"baseline_{wmape_metric}"].to_numpy(dtype=float)
        wmape_model = float(np.nanmean(wmape_model_values))
        wmape_baseline = float(np.nanmean(wmape_baseline_values))
        wmape_relative_delta = float(
            (wmape_model - wmape_baseline) / max(abs(wmape_baseline), 1e-12)
        )
        wmape_pass = wmape_relative_delta <= float(criteria.max_wmape_relative_regression)

    checks = {
        "percent_series_improved": percent_improved >= float(criteria.min_percent_series_improved),
        "macro_mase": macro_relative_delta <= float(criteria.max_macro_mase_relative_regression),
        "p90": p90_relative_delta <= float(criteria.max_p90_relative_regression),
        "wmape": wmape_pass,
    }
    accepted = bool(all(checks.values()))
    reason = "accepted" if accepted else "; ".join(
        key for key, passed in checks.items() if not passed
    )
    return Stage1AcceptanceSummary(
        accepted=accepted,
        reason=reason,
        num_series=num_series,
        percent_series_improved=percent_improved,
        macro_model_metric=macro_model,
        macro_baseline_metric=macro_baseline,
        macro_relative_delta=macro_relative_delta,
        p90_model_metric=p90_model,
        p90_baseline_metric=p90_baseline,
        p90_relative_delta=p90_relative_delta,
        wmape_model=wmape_model,
        wmape_baseline=wmape_baseline,
        wmape_relative_delta=wmape_relative_delta,
        primary_metric=primary_metric,
        wmape_metric=wmape_metric or "",
    )


def _cohort_reports(
    source_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    cohort_columns: Sequence[str],
    primary_metric: str,
    wmape_metric: str | None,
) -> Mapping[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    if not cohort_columns:
        return reports
    available = [column for column in cohort_columns if column in source_frame.columns]
    if not available:
        return reports
    metadata = (
        source_frame[[CROSS_KEY_COLUMN, *available]]
        .drop_duplicates(subset=[CROSS_KEY_COLUMN])
        .copy()
    )
    enriched = comparison.merge(metadata, on=CROSS_KEY_COLUMN, how="left")
    for column in available:
        grouped_rows: list[dict[str, Any]] = []
        for value, group in enriched.groupby(column, dropna=False):
            model_values = group[f"model_{primary_metric}"].to_numpy(dtype=float)
            baseline_values = group[f"baseline_{primary_metric}"].to_numpy(dtype=float)
            row = {
                column: value,
                "num_series": int(len(group)),
                "percent_series_improved": float(100.0 * group["improved"].mean()),
                "macro_model_metric": float(np.mean(model_values)),
                "macro_baseline_metric": float(np.mean(baseline_values)),
                "p90_model_metric": float(np.percentile(model_values, 90)),
                "p90_baseline_metric": float(np.percentile(baseline_values, 90)),
            }
            if wmape_metric is not None:
                row["wmape_model"] = float(np.nanmean(group[f"model_{wmape_metric}"].to_numpy(dtype=float)))
                row["wmape_baseline"] = float(np.nanmean(group[f"baseline_{wmape_metric}"].to_numpy(dtype=float)))
            grouped_rows.append(row)
        reports[column] = pd.DataFrame(grouped_rows).sort_values(
            ["percent_series_improved", "num_series"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return reports
