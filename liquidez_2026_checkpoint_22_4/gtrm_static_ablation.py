"""Static-context ablation report for GTRM Stage 1.

Checkpoint 21.4 formaliza la comparación entre dos variantes del Global
Representation Base:

- GTRM-A: ``use_static_context=False``;
- GTRM-B: ``use_static_context=True``.

El módulo no entrena modelos. Consume métricas por serie emitidas por el monitor
para dos corridas comparables y produce una decisión auditable. La pregunta no
es si ``x_static`` aumenta la capacidad del modelo, sino si mejora la precisión
por serie sin inducir deterioros macro/P90. Por eso el gate compara mejor
candidato con static context vs mejor candidato sin static context por
``cross_key_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from global_contracts import CROSS_KEY_COLUMN


_MODEL_FAMILIES = {"model", "global", "gtrm", "financial_gpt"}
_STATIC_TRUE_VALUES = {True, 1, "1", "true", "t", "yes", "y", "static", "with_static", "gtrm-b", "gtrm_b"}
_STATIC_FALSE_VALUES = {False, 0, "0", "false", "f", "no", "n", "no_static", "without_static", "gtrm-a", "gtrm_a"}


@dataclass(frozen=True)
class StaticContextAblationCriteria:
    """Gate para decidir si ``use_static_context=True`` se queda como default."""

    primary_metric: str = "MASE"
    wmape_metric: str = "WMAPE"
    min_percent_series_improved: float = 50.0
    max_macro_relative_regression: float = 0.0
    max_p90_relative_regression: float = 0.0
    max_wmape_relative_regression: float = 0.01
    min_series: int = 1

    def validate(self) -> None:
        if not str(self.primary_metric).strip():
            raise ValueError("primary_metric must not be empty")
        if not str(self.wmape_metric).strip():
            raise ValueError("wmape_metric must not be empty")
        if not 0.0 <= float(self.min_percent_series_improved) <= 100.0:
            raise ValueError("min_percent_series_improved must be between 0 and 100")
        for name in (
            "max_macro_relative_regression",
            "max_p90_relative_regression",
            "max_wmape_relative_regression",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite value")
        if isinstance(self.min_series, bool) or int(self.min_series) <= 0:
            raise ValueError("min_series must be a positive integer")


@dataclass(frozen=True)
class StaticContextAblationSummary:
    """Resultado top-level del experimento GTRM-A vs GTRM-B."""

    accepted: bool
    reason: str
    recommendation: str
    num_series: int
    percent_series_improved_by_static: float
    macro_static_metric: float
    macro_no_static_metric: float
    macro_relative_delta: float
    p90_static_metric: float
    p90_no_static_metric: float
    p90_relative_delta: float
    wmape_static: float
    wmape_no_static: float
    wmape_relative_delta: float
    primary_metric: str = "MASE"
    wmape_metric: str = "WMAPE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaticContextAblationReport:
    """Reporte completo con comparación por serie y por cohorte."""

    summary: StaticContextAblationSummary
    criteria: StaticContextAblationCriteria
    per_series: pd.DataFrame
    by_cohort: Mapping[str, pd.DataFrame]

    def write(self, output_directory: str | Path) -> Path:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "static_context_ablation_summary.json").write_text(
            json.dumps(self.summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (destination / "static_context_ablation_criteria.json").write_text(
            json.dumps(asdict(self.criteria), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.per_series.to_csv(destination / "static_context_ablation_by_series.csv", index=False)
        for name, frame in self.by_cohort.items():
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
            frame.to_csv(destination / f"static_context_ablation_by_{safe_name}.csv", index=False)
        return destination

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "criteria": asdict(self.criteria),
            "by_cohort": {
                name: frame.to_dict(orient="records") for name, frame in self.by_cohort.items()
            },
        }


def build_static_context_ablation_report(
    metrics_by_series: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    criteria: StaticContextAblationCriteria | None = None,
    cohort_columns: Sequence[str] = (),
    static_flag_column: str = "use_static_context",
) -> StaticContextAblationReport:
    """Compara GTRM-A vs GTRM-B con métricas por serie.

    Columnas esperadas:

    - ``cross_key_id`` o ``serie``;
    - ``candidate_id``;
    - ``family`` con valores tipo ``global``/``model``;
    - ``use_static_context`` indicando la variante;
    - métrica primaria, por defecto ``MASE``;
    - ``WMAPE`` opcional.

    Si hay varias arquitecturas por variante, se selecciona el mejor candidato
    por serie y por valor de ``use_static_context`` usando la métrica primaria.
    """

    selected_criteria = criteria or StaticContextAblationCriteria()
    selected_criteria.validate()

    frame = _normalize_ablation_frame(metrics_by_series, static_flag_column=static_flag_column)
    primary_metric = _resolve_metric_column(frame, selected_criteria.primary_metric)
    wmape_metric = _resolve_metric_column(frame, selected_criteria.wmape_metric, required=False)
    _validate_metric_values(frame, primary_metric=primary_metric, wmape_metric=wmape_metric)

    model_frame = frame[frame["family_normalized"].isin(_MODEL_FAMILIES)].copy()
    if model_frame.empty:
        raise ValueError("Static context ablation requires model/global candidates")

    no_static = model_frame[model_frame["use_static_context_normalized"] == False].copy()  # noqa: E712
    static = model_frame[model_frame["use_static_context_normalized"] == True].copy()  # noqa: E712
    if no_static.empty:
        raise ValueError("Static context ablation requires use_static_context=False metrics")
    if static.empty:
        raise ValueError("Static context ablation requires use_static_context=True metrics")

    best_no_static = _best_by_series(no_static, primary_metric, prefix="no_static")
    best_static = _best_by_series(static, primary_metric, prefix="static")
    comparison = best_static.merge(best_no_static, on=CROSS_KEY_COLUMN, how="inner")
    if len(comparison) < int(selected_criteria.min_series):
        raise ValueError(
            "Not enough series with both static and no-static metrics: "
            f"{len(comparison)} < {selected_criteria.min_series}"
        )

    comparison["improved_by_static"] = (
        comparison[f"static_{primary_metric}"] < comparison[f"no_static_{primary_metric}"]
    )
    comparison["metric_delta"] = (
        comparison[f"static_{primary_metric}"] - comparison[f"no_static_{primary_metric}"]
    )
    comparison["metric_relative_delta"] = _relative_delta(
        comparison[f"static_{primary_metric}"],
        comparison[f"no_static_{primary_metric}"],
    )

    if wmape_metric is not None:
        comparison["wmape_relative_delta"] = _relative_delta(
            comparison[f"static_{wmape_metric}"],
            comparison[f"no_static_{wmape_metric}"],
        )
    else:
        comparison["wmape_relative_delta"] = np.nan

    summary = _summarize_ablation(
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
    return StaticContextAblationReport(
        summary=summary,
        criteria=selected_criteria,
        per_series=comparison.sort_values(CROSS_KEY_COLUMN).reset_index(drop=True),
        by_cohort=by_cohort,
    )


def _normalize_ablation_frame(
    metrics_by_series: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    static_flag_column: str,
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
    for required in ("candidate_id", "family", static_flag_column):
        if required not in frame.columns:
            raise ValueError(f"metrics must contain {required}")
    frame[CROSS_KEY_COLUMN] = frame[CROSS_KEY_COLUMN].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["family"] = frame["family"].astype(str)
    frame["family_normalized"] = frame["family"].str.lower().str.strip()
    frame["use_static_context_normalized"] = frame[static_flag_column].map(_parse_static_flag)
    if frame["use_static_context_normalized"].isna().any():
        invalid = frame.loc[frame["use_static_context_normalized"].isna(), static_flag_column].unique().tolist()
        raise ValueError(f"Invalid static context flag values: {invalid}")
    frame["use_static_context_normalized"] = frame["use_static_context_normalized"].astype(bool)
    return frame


def _parse_static_flag(value: object) -> bool | None:
    normalized = value
    if isinstance(value, str):
        normalized = value.strip().lower()
    if normalized in _STATIC_TRUE_VALUES:
        return True
    if normalized in _STATIC_FALSE_VALUES:
        return False
    return None


def _resolve_metric_column(frame: pd.DataFrame, requested: str, *, required: bool = True) -> str | None:
    aliases = {requested, requested.upper(), requested.lower(), requested.capitalize()}
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


def _validate_metric_values(frame: pd.DataFrame, *, primary_metric: str, wmape_metric: str | None) -> None:
    values = pd.to_numeric(frame[primary_metric], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Metric {primary_metric!r} must be finite for all candidates")
    if (values < 0).any():
        raise ValueError(f"Metric {primary_metric!r} must be non-negative")
    frame[primary_metric] = values.astype(float)
    if wmape_metric is not None:
        frame[wmape_metric] = pd.to_numeric(frame[wmape_metric], errors="coerce")


def _best_by_series(frame: pd.DataFrame, primary_metric: str, *, prefix: str) -> pd.DataFrame:
    sorted_frame = frame.sort_values([CROSS_KEY_COLUMN, primary_metric, "candidate_id"])
    keep_columns = [CROSS_KEY_COLUMN, "candidate_id", "family", primary_metric, "use_static_context_normalized"]
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


def _summarize_ablation(
    comparison: pd.DataFrame,
    *,
    criteria: StaticContextAblationCriteria,
    primary_metric: str,
    wmape_metric: str | None,
) -> StaticContextAblationSummary:
    num_series = int(len(comparison))
    percent_improved = float(100.0 * comparison["improved_by_static"].mean())
    static_values = comparison[f"static_{primary_metric}"].to_numpy(dtype=float)
    no_static_values = comparison[f"no_static_{primary_metric}"].to_numpy(dtype=float)
    macro_static = float(np.mean(static_values))
    macro_no_static = float(np.mean(no_static_values))
    macro_relative_delta = float((macro_static - macro_no_static) / max(abs(macro_no_static), 1e-12))
    p90_static = float(np.percentile(static_values, 90))
    p90_no_static = float(np.percentile(no_static_values, 90))
    p90_relative_delta = float((p90_static - p90_no_static) / max(abs(p90_no_static), 1e-12))

    if wmape_metric is None:
        wmape_static = math.nan
        wmape_no_static = math.nan
        wmape_relative_delta = math.nan
        wmape_pass = True
    else:
        wmape_static = float(np.nanmean(comparison[f"static_{wmape_metric}"].to_numpy(dtype=float)))
        wmape_no_static = float(np.nanmean(comparison[f"no_static_{wmape_metric}"].to_numpy(dtype=float)))
        wmape_relative_delta = float((wmape_static - wmape_no_static) / max(abs(wmape_no_static), 1e-12))
        wmape_pass = wmape_relative_delta <= float(criteria.max_wmape_relative_regression)

    checks = {
        "percent_series_improved_by_static": percent_improved >= float(criteria.min_percent_series_improved),
        "macro_metric": macro_relative_delta <= float(criteria.max_macro_relative_regression),
        "p90_metric": p90_relative_delta <= float(criteria.max_p90_relative_regression),
        "wmape": wmape_pass,
    }
    accepted = bool(all(checks.values()))
    reason = "accepted" if accepted else "; ".join(key for key, passed in checks.items() if not passed)
    recommendation = (
        "keep_use_static_context_true_as_stage1_default"
        if accepted
        else "do_not_promote_static_context_without_review"
    )
    return StaticContextAblationSummary(
        accepted=accepted,
        reason=reason,
        recommendation=recommendation,
        num_series=num_series,
        percent_series_improved_by_static=percent_improved,
        macro_static_metric=macro_static,
        macro_no_static_metric=macro_no_static,
        macro_relative_delta=macro_relative_delta,
        p90_static_metric=p90_static,
        p90_no_static_metric=p90_no_static,
        p90_relative_delta=p90_relative_delta,
        wmape_static=wmape_static,
        wmape_no_static=wmape_no_static,
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
    metadata = source_frame[[CROSS_KEY_COLUMN, *available]].drop_duplicates(subset=[CROSS_KEY_COLUMN]).copy()
    enriched = comparison.merge(metadata, on=CROSS_KEY_COLUMN, how="left")
    for column in available:
        rows: list[dict[str, Any]] = []
        for value, group in enriched.groupby(column, dropna=False):
            static_values = group[f"static_{primary_metric}"].to_numpy(dtype=float)
            no_static_values = group[f"no_static_{primary_metric}"].to_numpy(dtype=float)
            row: dict[str, Any] = {
                column: value,
                "num_series": int(len(group)),
                "percent_series_improved_by_static": float(100.0 * group["improved_by_static"].mean()),
                "macro_static_metric": float(np.mean(static_values)),
                "macro_no_static_metric": float(np.mean(no_static_values)),
                "p90_static_metric": float(np.percentile(static_values, 90)),
                "p90_no_static_metric": float(np.percentile(no_static_values, 90)),
            }
            if wmape_metric is not None:
                row["wmape_static"] = float(np.nanmean(group[f"static_{wmape_metric}"].to_numpy(dtype=float)))
                row["wmape_no_static"] = float(np.nanmean(group[f"no_static_{wmape_metric}"].to_numpy(dtype=float)))
            rows.append(row)
        reports[column] = pd.DataFrame(rows).sort_values(
            ["percent_series_improved_by_static", "num_series"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return reports
