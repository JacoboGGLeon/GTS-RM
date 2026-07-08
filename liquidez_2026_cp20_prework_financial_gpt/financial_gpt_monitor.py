"""Comparadores reproducibles para los monitores locales y Financial-GPT.

``monitor_codigo_02.ipynb`` compara exclusivamente los cuatro modelos locales
por serie y ``monitor_codigo_03_FINANCIAL_GPT.ipynb`` compara exclusivamente las
cuatro arquitecturas globales. Ambos monitores comparten los mismos baselines:
``NAIVE_LAST_VALUE``, ``NAIVE_ZERO`` sólo para variación y
``SEASONAL_NAIVE_FINANCIAL``. Las métricas siempre se recalculan sobre fechas de
test comunes por ``cross_key_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import polars as pl

from global_contracts import CROSS_KEY_COLUMN
from global_s3 import DEFAULT_FINANCIAL_GPT_S3_ROOT, resolve_latest_run_uri
from tools import Tools


LOCAL_ALGORITHMS: Tuple[str, ...] = (
    "MLP_E_D",
    "MLP_VaE_D",
    "RNN_E_D",
    "RNNBi_E_D",
)
GLOBAL_ARCHITECTURES: Mapping[str, str] = {
    "GLOBAL_MLP_E_D": "mlp",
    "GLOBAL_MLP_VaE_D": "mlp_vae",
    "GLOBAL_RNN_E_D": "rnn",
    "GLOBAL_RNNBi_E_D": "rnn_bi",
}
DEFAULT_LOCAL_EXECUTION_ROOT = (
    "s3://ada-us-east-1-sbx-live-mx-m6hn-data/"
    "data/sandboxes/m6hn/data/coe_liquidez_diaria/execution"
)
DEFAULT_WINNER_METRICS: Tuple[str, ...] = ("MASE",)
PRIMARY_WINNER_METRIC = "MASE"
QUALITY_METRICS = frozenset({"EVS", "R2"})
NAIVE_LAST_VALUE_ID = "NAIVE_LAST_VALUE"
NAIVE_ZERO_ID = "NAIVE_ZERO"
SEASONAL_NAIVE_ID = "SEASONAL_NAIVE_FINANCIAL"
# Alias histórico para consumidores existentes.
NAIVE_CANDIDATE_ID = NAIVE_LAST_VALUE_ID
BASELINE_CANDIDATE_IDS: Tuple[str, ...] = (
    NAIVE_LAST_VALUE_ID,
    NAIVE_ZERO_ID,
    SEASONAL_NAIVE_ID,
)
DEFAULT_SEASONAL_PERIOD_DAYS = 7


@dataclass(frozen=True)
class CandidateRun:
    candidate_id: str
    family: str
    architecture: str
    run_uri: str
    backtest: pd.DataFrame
    forecast: pd.DataFrame


@dataclass(frozen=True)
class FinancialGPTMonitorResult:
    run_inventory: pl.DataFrame
    comparison_coverage: pl.DataFrame
    metrics_by_series: pl.DataFrame
    winners_by_series: pl.DataFrame
    winner_counts: pl.DataFrame
    ensemble_forecast: pl.DataFrame

    def to_dict(self) -> Mapping[str, Any]:
        """Return one portable JSON document with every monitor table."""

        return {
            "schema_version": "1.1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "candidates": int(self.run_inventory.height),
                "series_compared": int(self.winners_by_series.height),
                "forecast_rows": int(self.ensemble_forecast.height),
            },
            "run_inventory": _frame_records(self.run_inventory),
            "comparison_coverage": _frame_records(self.comparison_coverage),
            "metrics_by_series": _frame_records(self.metrics_by_series),
            "winners_by_series": _frame_records(self.winners_by_series),
            "winner_counts": _frame_records(self.winner_counts),
            "ensemble_forecast": _frame_records(self.ensemble_forecast),
        }

    def write(self, output_directory: str | Path) -> Path:
        """Write exactly one non-ZIP monitor artifact: a JSON document."""

        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / "financial_gpt_monitor.json"
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path


def _frame_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_compatible(value) for key, value in row.items()}
        for row in frame.to_dicts()
    ]


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return str(value)


def discover_latest_local_runs(
    execution_root: str = DEFAULT_LOCAL_EXECUTION_ROOT,
    *,
    algorithms: Sequence[str] = LOCAL_ALGORITHMS,
    s3_client=None,
) -> Mapping[str, str]:
    """Descubre el último run disponible por algoritmo local."""

    root = str(execution_root).strip().rstrip("/")
    if not root:
        raise ValueError("execution_root must not be empty")
    if root.startswith("s3://"):
        return _discover_latest_local_runs_s3(root, algorithms, s3_client=s3_client)
    return _discover_latest_local_runs_path(root, algorithms)


def discover_latest_global_runs(
    s3_root: str = DEFAULT_FINANCIAL_GPT_S3_ROOT,
    *,
    architectures: Mapping[str, str] = GLOBAL_ARCHITECTURES,
    s3_client=None,
) -> Mapping[str, str]:
    """Resuelve ``latest.json`` de las cuatro arquitecturas globales."""

    result: Dict[str, str] = {}
    for candidate_id, architecture in architectures.items():
        result[candidate_id] = resolve_latest_run_uri(
            s3_root,
            architecture,
            client=s3_client,
        )
    return result


def load_local_candidate(
    algorithm: str,
    run_uri: str,
    *,
    s3_client=None,
) -> CandidateRun:
    normalized = str(algorithm).strip()
    if normalized not in LOCAL_ALGORITHMS:
        raise ValueError(f"Unsupported local algorithm: {algorithm!r}")
    base = str(run_uri).strip().rstrip("/")
    backtest = _read_dataframe(
        _join_uri(base, "data/backtest_timeseries.csv"),
        s3_client=s3_client,
    )
    forecast = _read_dataframe(
        _join_uri(base, "data/forecast_timeseries.csv"),
        s3_client=s3_client,
    )
    return CandidateRun(
        candidate_id=f"LOCAL_{normalized}",
        family="local",
        architecture=normalized,
        run_uri=base,
        backtest=_normalize_backtest(backtest),
        forecast=_normalize_forecast(forecast),
    )


def load_global_candidate(
    candidate_id: str,
    run_uri: str,
    *,
    s3_client=None,
) -> CandidateRun:
    expected_architecture = GLOBAL_ARCHITECTURES.get(str(candidate_id))
    if expected_architecture is None:
        raise ValueError(f"Unsupported global candidate: {candidate_id!r}")
    base = str(run_uri).strip().rstrip("/")
    manifest = _read_json(
        _join_uri(base, "model/manifest.json"),
        s3_client=s3_client,
    )
    architecture = str(manifest.get("architecture", "")).strip().lower()
    if architecture != expected_architecture:
        raise ValueError(
            f"{candidate_id} expected architecture={expected_architecture!r}, "
            f"got {architecture!r}"
        )
    backtest = _read_dataframe(
        _join_uri(base, "reports/backtest_mc_by_series.parquet"),
        s3_client=s3_client,
    )
    forecast = _read_dataframe(
        _join_uri(base, "reports/future_forecast_mc_by_series.parquet"),
        s3_client=s3_client,
    )
    return CandidateRun(
        candidate_id=str(candidate_id),
        family="global",
        architecture=architecture,
        run_uri=base,
        backtest=_normalize_backtest(backtest),
        forecast=_normalize_forecast(forecast),
    )


def compare_local_financial_runs(
    *,
    local_run_uris: Mapping[str, str],
    metrics: Sequence[str] = DEFAULT_WINNER_METRICS,
    include_baselines: bool = True,
    seasonal_period_days: int = DEFAULT_SEASONAL_PERIOD_DAYS,
    include_naive: bool | None = None,
    s3_client=None,
) -> FinancialGPTMonitorResult:
    """Compara exactamente cuatro modelos locales y los baselines comunes."""

    missing = [value for value in LOCAL_ALGORITHMS if value not in local_run_uris]
    unexpected = [value for value in local_run_uris if value not in LOCAL_ALGORITHMS]
    if missing or unexpected:
        raise ValueError(
            "Local monitor requires exactly four local runs; "
            f"missing={missing}, unexpected={unexpected}"
        )
    candidates = [
        load_local_candidate(
            algorithm,
            local_run_uris[algorithm],
            s3_client=s3_client,
        )
        for algorithm in LOCAL_ALGORITHMS
    ]
    enabled = _resolve_baseline_flag(include_baselines, include_naive)
    return _compare_candidate_runs(
        candidates,
        metrics=metrics,
        include_baselines=enabled,
        seasonal_period_days=seasonal_period_days,
    )


def compare_financial_gpt_runs(
    *,
    local_run_uris: Mapping[str, str],
    global_run_uris: Mapping[str, str],
    metrics: Sequence[str] = DEFAULT_WINNER_METRICS,
    include_baselines: bool = True,
    seasonal_period_days: int = DEFAULT_SEASONAL_PERIOD_DAYS,
    include_naive: bool | None = None,
    s3_client=None,
) -> FinancialGPTMonitorResult:
    """Comparador histórico de ocho modelos y los tres baselines comunes."""

    _validate_expected_runs(local_run_uris, global_run_uris)
    candidates: list[CandidateRun] = []
    for algorithm in LOCAL_ALGORITHMS:
        candidates.append(
            load_local_candidate(
                algorithm,
                local_run_uris[algorithm],
                s3_client=s3_client,
            )
        )
    for candidate_id in GLOBAL_ARCHITECTURES:
        candidates.append(
            load_global_candidate(
                candidate_id,
                global_run_uris[candidate_id],
                s3_client=s3_client,
            )
        )
    enabled = _resolve_baseline_flag(include_baselines, include_naive)
    return _compare_candidate_runs(
        candidates,
        metrics=metrics,
        include_baselines=enabled,
        seasonal_period_days=seasonal_period_days,
    )


def compare_global_financial_gpt_runs(
    *,
    global_run_uris: Mapping[str, str],
    metrics: Sequence[str] = DEFAULT_WINNER_METRICS,
    include_baselines: bool = True,
    seasonal_period_days: int = DEFAULT_SEASONAL_PERIOD_DAYS,
    include_naive: bool | None = None,
    s3_client=None,
) -> FinancialGPTMonitorResult:
    """Compara cuatro modelos globales y los tres baselines comunes."""

    _validate_expected_global_runs(global_run_uris)
    candidates = [
        load_global_candidate(
            candidate_id,
            global_run_uris[candidate_id],
            s3_client=s3_client,
        )
        for candidate_id in GLOBAL_ARCHITECTURES
    ]
    enabled = _resolve_baseline_flag(include_baselines, include_naive)
    return _compare_candidate_runs(
        candidates,
        metrics=metrics,
        include_baselines=enabled,
        seasonal_period_days=seasonal_period_days,
    )


def _compare_candidate_runs(
    candidates: Sequence[CandidateRun],
    *,
    metrics: Sequence[str],
    include_baselines: bool,
    seasonal_period_days: int,
) -> FinancialGPTMonitorResult:
    if not candidates:
        raise ValueError("At least one candidate run is required")
    if int(seasonal_period_days) < 1:
        raise ValueError("seasonal_period_days must be >= 1")

    # ``metrics`` is retained as a compatibility-only diagnostic argument.
    # Checkpoint 20 always selects with one non-redundant primary metric.
    requested_metrics = tuple(str(value) for value in metrics)
    known_metrics = set(Tools.metrics_regression([1.0, 2.0], [1.0, 2.0]))
    unknown = [value for value in requested_metrics if value not in known_metrics]
    if unknown:
        raise ValueError(f"Unsupported diagnostic metrics: {unknown}")
    selected_metrics = (PRIMARY_WINNER_METRIC,)

    metric_rows, coverage_rows, baseline_forecasts, baseline_series = _evaluate_candidates(
        candidates,
        include_baselines=include_baselines,
        seasonal_period_days=int(seasonal_period_days),
    )
    if not metric_rows:
        raise RuntimeError("No common test observations were found")
    metrics_frame = pl.DataFrame(metric_rows)
    ranked = _rank_candidates(metrics_frame, selected_metrics)
    winners = (
        ranked.sort(
            [CROSS_KEY_COLUMN, "selection_rank", "sMAPE", "candidate_id"]
        )
        .group_by(CROSS_KEY_COLUMN, maintain_order=True)
        .first()
        .select(
            CROSS_KEY_COLUMN,
            pl.col("candidate_id").alias("winner_candidate"),
            pl.col("family").alias("winner_family"),
            pl.col("architecture").alias("winner_architecture"),
            "selection_metric",
            "selection_score",
            "selection_rank",
            "comparison_points",
            *selected_metrics,
        )
        .sort(CROSS_KEY_COLUMN)
    )
    ensemble = _build_ensemble_forecast(
        candidates,
        winners,
        baseline_forecasts=baseline_forecasts,
    )
    inventory_rows = [
        {
            "candidate_id": item.candidate_id,
            "family": item.family,
            "architecture": item.architecture,
            "run_uri": item.run_uri,
            "backtest_series": int(item.backtest[CROSS_KEY_COLUMN].nunique()),
            "forecast_series": int(item.forecast[CROSS_KEY_COLUMN].nunique()),
        }
        for item in candidates
    ]
    if include_baselines:
        baseline_meta = {
            NAIVE_LAST_VALUE_ID: "last_value",
            NAIVE_ZERO_ID: "zero_variacion",
            SEASONAL_NAIVE_ID: f"seasonal_{int(seasonal_period_days)}d",
        }
        for candidate_id in BASELINE_CANDIDATE_IDS:
            count = len(baseline_series.get(candidate_id, set()))
            inventory_rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": "baseline",
                    "architecture": baseline_meta[candidate_id],
                    "run_uri": "derived_from_common_backtest_actuals",
                    "backtest_series": count,
                    "forecast_series": count,
                }
            )
    winner_counts = (
        winners.group_by(["winner_candidate", "winner_family", "winner_architecture"])
        .len(name="num_series")
        .sort("num_series", descending=True)
    )
    return FinancialGPTMonitorResult(
        run_inventory=pl.DataFrame(inventory_rows).sort("candidate_id"),
        comparison_coverage=pl.DataFrame(coverage_rows).sort(CROSS_KEY_COLUMN),
        metrics_by_series=ranked.sort(
            [CROSS_KEY_COLUMN, "selection_rank", "candidate_id"]
        ),
        winners_by_series=winners,
        winner_counts=winner_counts,
        ensemble_forecast=ensemble,
    )


def _evaluate_candidates(
    candidates: Sequence[CandidateRun],
    *,
    include_baselines: bool,
    seasonal_period_days: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Mapping[str, Mapping[str, pd.DataFrame]],
    Mapping[str, set[str]],
]:
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    baseline_forecasts: Dict[str, Dict[str, pd.DataFrame]] = {
        candidate_id: {} for candidate_id in BASELINE_CANDIDATE_IDS
    }
    baseline_series: Dict[str, set[str]] = {
        candidate_id: set() for candidate_id in BASELINE_CANDIDATE_IDS
    }
    all_series = sorted(
        set().union(
            *(set(item.backtest[CROSS_KEY_COLUMN].astype(str)) for item in candidates)
        )
    )
    for series_id in all_series:
        frames: Dict[str, pd.DataFrame] = {}
        candidate_meta: Dict[str, CandidateRun] = {}
        for candidate in candidates:
            frame = candidate.backtest[
                candidate.backtest[CROSS_KEY_COLUMN].astype(str) == series_id
            ].copy()
            test = frame.loc[~frame["isTrain"]].sort_values("date")
            if not test.empty:
                frames[candidate.candidate_id] = test
                candidate_meta[candidate.candidate_id] = candidate
        if len(frames) < 2:
            continue
        common_dates = set.intersection(
            *(set(pd.to_datetime(frame["date"])) for frame in frames.values())
        )
        if not common_dates:
            continue
        ordered_dates = pd.DatetimeIndex(sorted(common_dates))
        actual_by_candidate: list[np.ndarray] = []
        for frame in frames.values():
            indexed = frame.drop_duplicates("date").set_index("date")
            actual_by_candidate.append(
                indexed.loc[ordered_dates, "actual_orig"].to_numpy(dtype=float)
            )
        reference_actual = actual_by_candidate[0]
        for values in actual_by_candidate[1:]:
            if not np.allclose(reference_actual, values, rtol=1e-7, atol=1e-6, equal_nan=True):
                raise ValueError(f"Actual values differ across candidates for {series_id!r}")
        reference_full = next(iter(candidate_meta.values())).backtest
        mase_scale = _causal_mase_scale_from_backtest(
            reference_full,
            series_id=series_id,
            comparison_start=ordered_dates.min(),
        )
        for candidate_id, frame in frames.items():
            candidate = candidate_meta[candidate_id]
            indexed = frame.drop_duplicates("date").set_index("date")
            prediction = indexed.loc[ordered_dates, "pred_orig"].to_numpy(dtype=float)
            metric_rows.append(
                {
                    CROSS_KEY_COLUMN: series_id,
                    "candidate_id": candidate_id,
                    "family": candidate.family,
                    "architecture": candidate.architecture,
                    "comparison_start": ordered_dates.min(),
                    "comparison_end": ordered_dates.max(),
                    "comparison_points": len(ordered_dates),
                    **_metrics_with_causal_mase(
                        reference_actual, prediction, mase_scale=mase_scale
                    ),
                }
            )

        applicable_baselines: list[str] = []
        if include_baselines:
            baseline_predictions: Dict[str, np.ndarray] = {
                NAIVE_LAST_VALUE_ID: _last_value_prediction(
                    reference_full, series_id=series_id, target_dates=ordered_dates
                ),
                SEASONAL_NAIVE_ID: _seasonal_prediction(
                    reference_full,
                    series_id=series_id,
                    target_dates=ordered_dates,
                    period_days=seasonal_period_days,
                ),
            }
            if _is_variation_series(reference_full, series_id):
                baseline_predictions[NAIVE_ZERO_ID] = np.zeros(len(ordered_dates), dtype=float)

            architecture = {
                NAIVE_LAST_VALUE_ID: "last_value",
                NAIVE_ZERO_ID: "zero_variacion",
                SEASONAL_NAIVE_ID: f"seasonal_{seasonal_period_days}d",
            }
            for candidate_id, prediction in baseline_predictions.items():
                if len(prediction) != len(ordered_dates):
                    continue
                metric_rows.append(
                    {
                        CROSS_KEY_COLUMN: series_id,
                        "candidate_id": candidate_id,
                        "family": "baseline",
                        "architecture": architecture[candidate_id],
                        "comparison_start": ordered_dates.min(),
                        "comparison_end": ordered_dates.max(),
                        "comparison_points": len(ordered_dates),
                        **_metrics_with_causal_mase(
                            reference_actual, prediction, mase_scale=mase_scale
                        ),
                    }
                )
                applicable_baselines.append(candidate_id)
                baseline_series[candidate_id].add(series_id)

        coverage_rows.append(
            {
                CROSS_KEY_COLUMN: series_id,
                "comparison_start": ordered_dates.min(),
                "comparison_end": ordered_dates.max(),
                "comparison_points": len(ordered_dates),
                "model_candidates": len(frames),
                "baseline_candidates": len(applicable_baselines),
                "baseline_ids": ",".join(applicable_baselines),
                "mase_scale": float(mase_scale),
            }
        )

        if not include_baselines:
            continue
        future_dates = sorted(
            set().union(
                *(
                    set(pd.to_datetime(candidate.forecast.loc[
                        candidate.forecast[CROSS_KEY_COLUMN].astype(str) == series_id,
                        "date",
                    ]))
                    for candidate in candidates
                )
            )
        )
        if not future_dates:
            continue
        future_index = pd.DatetimeIndex(future_dates)
        last_value = _last_observed_value(reference_full, series_id)
        if last_value is None:
            continue
        future_predictions: Dict[str, np.ndarray] = {
            NAIVE_LAST_VALUE_ID: np.full(len(future_index), last_value, dtype=float),
            SEASONAL_NAIVE_ID: _seasonal_prediction(
                reference_full,
                series_id=series_id,
                target_dates=future_index,
                period_days=seasonal_period_days,
            ),
        }
        if _is_variation_series(reference_full, series_id):
            future_predictions[NAIVE_ZERO_ID] = np.zeros(len(future_index), dtype=float)
        architecture = {
            NAIVE_LAST_VALUE_ID: "last_value",
            NAIVE_ZERO_ID: "zero_variacion",
            SEASONAL_NAIVE_ID: f"seasonal_{seasonal_period_days}d",
        }
        for candidate_id, prediction in future_predictions.items():
            if len(prediction) != len(future_index):
                continue
            baseline_forecasts[candidate_id][series_id] = pd.DataFrame(
                {
                    "date": future_index,
                    CROSS_KEY_COLUMN: series_id,
                    "pred_orig": prediction,
                    "lower_ci": prediction,
                    "upper_ci": prediction,
                    "candidate_id": candidate_id,
                    "family": "baseline",
                    "architecture": architecture[candidate_id],
                }
            )
    return metric_rows, coverage_rows, baseline_forecasts, baseline_series


def _last_value_prediction(
    backtest: pd.DataFrame,
    *,
    series_id: str,
    target_dates: pd.DatetimeIndex,
) -> np.ndarray:
    frame = _actual_history(backtest, series_id)
    if frame.empty:
        return np.asarray([], dtype=float)
    history = frame["actual_orig"].astype(float)
    shifted = history.shift(1)
    values = shifted.reindex(target_dates)
    if values.isna().any():
        fallback = history.loc[history.index < target_dates.min()]
        if fallback.empty:
            return np.asarray([], dtype=float)
        values = values.fillna(float(fallback.iloc[-1]))
    return values.to_numpy(dtype=float)


def _seasonal_prediction(
    backtest: pd.DataFrame,
    *,
    series_id: str,
    target_dates: pd.DatetimeIndex,
    period_days: int,
) -> np.ndarray:
    frame = _actual_history(backtest, series_id)
    if frame.empty:
        return np.asarray([], dtype=float)
    history = frame["actual_orig"].astype(float)
    generated: Dict[pd.Timestamp, float] = {}
    predictions: list[float] = []
    delta = pd.Timedelta(days=int(period_days))
    for raw_date in pd.DatetimeIndex(target_dates).sort_values():
        target = pd.Timestamp(raw_date).normalize()
        source = target - delta
        if source in history.index:
            value = float(history.loc[source])
        elif source in generated:
            value = float(generated[source])
        else:
            prior_history = history.loc[history.index < target]
            prior_generated = [
                (when, value) for when, value in generated.items() if when < target
            ]
            if prior_generated:
                generated_date, generated_value = max(prior_generated, key=lambda item: item[0])
            else:
                generated_date, generated_value = pd.Timestamp.min, np.nan
            if not prior_history.empty and prior_history.index[-1] >= generated_date:
                value = float(prior_history.iloc[-1])
            elif np.isfinite(generated_value):
                value = float(generated_value)
            else:
                return np.asarray([], dtype=float)
        generated[target] = value
        predictions.append(value)
    return np.asarray(predictions, dtype=float)


def _actual_history(backtest: pd.DataFrame, series_id: str) -> pd.DataFrame:
    frame = backtest[backtest[CROSS_KEY_COLUMN].astype(str) == str(series_id)].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return (
        frame.sort_values(["date", "isTrain"], ascending=[True, False])
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )


def _last_observed_value(backtest: pd.DataFrame, series_id: str) -> float | None:
    frame = _actual_history(backtest, series_id)
    if frame.empty:
        return None
    value = float(frame["actual_orig"].iloc[-1])
    return value if np.isfinite(value) else None


def _is_variation_series(backtest: pd.DataFrame, series_id: str) -> bool:
    frame = backtest[backtest[CROSS_KEY_COLUMN].astype(str) == str(series_id)]
    if "tipo_serie" in frame.columns:
        values = frame["tipo_serie"].dropna().astype(str).str.lower().str.strip()
        if not values.empty:
            return bool((values == "variacion").all() or (values == "variación").all())
    normalized = re.sub(r"[^a-z0-9]+", "_", str(series_id).lower()).strip("_")
    return normalized.endswith("variacion") or normalized.endswith("variation")


def _resolve_baseline_flag(include_baselines: bool, include_naive: bool | None) -> bool:
    return bool(include_baselines if include_naive is None else include_naive)


def _rank_candidates(frame: pl.DataFrame, metrics: Sequence[str]) -> pl.DataFrame:
    if tuple(metrics) != (PRIMARY_WINNER_METRIC,):
        raise ValueError("Exactly one non-redundant primary metric is required")
    metric = PRIMARY_WINNER_METRIC
    safe_metric = (
        pl.when(pl.col(metric).is_finite())
        .then(pl.col(metric))
        .otherwise(pl.lit(float("inf")))
    )
    return frame.with_columns(
        pl.lit(metric).alias("selection_metric"),
        safe_metric.alias("selection_score"),
        safe_metric.rank(method="average", descending=False)
        .over(CROSS_KEY_COLUMN)
        .alias("selection_rank"),
    )


def _metrics_with_causal_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mase_scale: float,
) -> Mapping[str, float]:
    metrics = dict(Tools.metrics_regression(y_true, y_pred))
    if not np.isfinite(mase_scale) or float(mase_scale) <= 0.0:
        raise ValueError(f"Invalid causal MASE scale: {mase_scale}")
    metrics["MASE"] = float(metrics["MAE"] / float(mase_scale))
    metrics["MASE_SCALE"] = float(mase_scale)
    return metrics


def _causal_mase_scale_from_backtest(
    backtest: pd.DataFrame,
    *,
    series_id: str,
    comparison_start: pd.Timestamp,
) -> float:
    frame = backtest[
        backtest[CROSS_KEY_COLUMN].astype(str) == str(series_id)
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    cutoff = pd.Timestamp(comparison_start).normalize()
    history = frame.loc[
        (frame["date"] < cutoff) & frame["isTrain"].map(_coerce_bool),
        ["date", "actual_orig"],
    ]
    if history.empty:
        history = frame.loc[
            frame["date"] < cutoff, ["date", "actual_orig"]
        ]
    values = (
        history.sort_values("date")
        .drop_duplicates("date")["actual_orig"]
        .to_numpy(dtype=float)
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(
            f"No causal reference history is available for {series_id!r}"
        )
    naive_mae = (
        float(np.mean(np.abs(np.diff(values)))) if values.size > 1 else 0.0
    )
    mean_abs_level = float(np.mean(np.abs(values)))
    return float(max(naive_mae, 0.01 * max(mean_abs_level, 1.0), 1e-6))


def _build_ensemble_forecast(
    candidates: Sequence[CandidateRun],
    winners: pl.DataFrame,
    *,
    baseline_forecasts: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pl.DataFrame:
    frames: list[pd.DataFrame] = []
    by_id = {item.candidate_id: item for item in candidates}
    for row in winners.to_dicts():
        series_id = str(row[CROSS_KEY_COLUMN])
        winner = str(row["winner_candidate"])
        if winner in baseline_forecasts:
            selected = baseline_forecasts[winner].get(series_id, pd.DataFrame()).copy()
        else:
            candidate = by_id[winner]
            selected = candidate.forecast[
                candidate.forecast[CROSS_KEY_COLUMN].astype(str) == series_id
            ].copy()
            if not selected.empty:
                selected["candidate_id"] = candidate.candidate_id
                selected["family"] = candidate.family
                selected["architecture"] = candidate.architecture
        if selected.empty:
            continue
        selected["winner_candidate"] = winner
        frames.append(selected)
    if not frames:
        return pl.DataFrame(
            schema={
                "date": pl.Datetime,
                CROSS_KEY_COLUMN: pl.String,
                "pred_orig": pl.Float64,
                "lower_ci": pl.Float64,
                "upper_ci": pl.Float64,
                "candidate_id": pl.String,
                "family": pl.String,
                "architecture": pl.String,
                "winner_candidate": pl.String,
            }
        )
    consolidated = pd.concat(frames, ignore_index=True)
    keep = [
        "date",
        CROSS_KEY_COLUMN,
        "pred_orig",
        "lower_ci",
        "upper_ci",
        "candidate_id",
        "family",
        "architecture",
        "winner_candidate",
    ]
    return pl.DataFrame(consolidated.loc[:, keep].to_dict("records")).sort(
        [CROSS_KEY_COLUMN, "date"]
    )


def _normalize_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    df = _drop_unnamed(frame.copy())
    if CROSS_KEY_COLUMN not in df.columns:
        if "serie" not in df.columns:
            raise ValueError("Backtest must contain 'serie' or 'cross_key_id'")
        df[CROSS_KEY_COLUMN] = df["serie"].astype(str)
    if "serie" not in df.columns:
        df["serie"] = df[CROSS_KEY_COLUMN].astype(str)
    rename = {}
    for canonical, candidates in {
        "date": ("date", "fecha"),
        "actual_orig": ("actual_orig", "actual", "y_true"),
        "pred_orig": ("pred_orig", "prediction", "y_pred"),
    }.items():
        if canonical not in df.columns:
            match = next((value for value in candidates if value in df.columns), None)
            if match is None:
                raise ValueError(f"Backtest is missing {canonical!r}")
            rename[match] = canonical
    df = df.rename(columns=rename)
    if "isTrain" not in df.columns:
        df["isTrain"] = False
    df["isTrain"] = df["isTrain"].map(_coerce_bool)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df[CROSS_KEY_COLUMN] = df[CROSS_KEY_COLUMN].astype(str).str.strip()
    df["actual_orig"] = pd.to_numeric(df["actual_orig"], errors="coerce")
    df["pred_orig"] = pd.to_numeric(df["pred_orig"], errors="coerce")
    df = df.dropna(subset=["date", CROSS_KEY_COLUMN, "actual_orig", "pred_orig"])
    return (
        df.groupby(["date", CROSS_KEY_COLUMN, "isTrain"], as_index=False)
        .agg(actual_orig=("actual_orig", "mean"), pred_orig=("pred_orig", "mean"))
        .sort_values([CROSS_KEY_COLUMN, "date", "isTrain"])
        .reset_index(drop=True)
    )


def _normalize_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    df = _drop_unnamed(frame.copy())
    if CROSS_KEY_COLUMN not in df.columns:
        if "serie" not in df.columns:
            raise ValueError("Forecast must contain 'serie' or 'cross_key_id'")
        df[CROSS_KEY_COLUMN] = df["serie"].astype(str)
    rename = {}
    for canonical, candidates in {
        "date": ("date", "fecha"),
        "pred_orig": ("pred_orig", "prediction", "forecast"),
    }.items():
        if canonical not in df.columns:
            match = next((value for value in candidates if value in df.columns), None)
            if match is None:
                raise ValueError(f"Forecast is missing {canonical!r}")
            rename[match] = canonical
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df[CROSS_KEY_COLUMN] = df[CROSS_KEY_COLUMN].astype(str).str.strip()
    df["pred_orig"] = pd.to_numeric(df["pred_orig"], errors="coerce")
    for column in ("lower_ci", "upper_ci"):
        if column not in df.columns:
            df[column] = df["pred_orig"]
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", CROSS_KEY_COLUMN, "pred_orig"])
    return (
        df.groupby(["date", CROSS_KEY_COLUMN], as_index=False)
        .agg(
            pred_orig=("pred_orig", "mean"),
            lower_ci=("lower_ci", "mean"),
            upper_ci=("upper_ci", "mean"),
        )
        .sort_values([CROSS_KEY_COLUMN, "date"])
        .reset_index(drop=True)
    )


def _validate_expected_global_runs(
    global_run_uris: Mapping[str, str],
) -> None:
    missing_global = [
        value for value in GLOBAL_ARCHITECTURES if value not in global_run_uris
    ]
    unexpected = [
        value for value in global_run_uris if value not in GLOBAL_ARCHITECTURES
    ]
    if missing_global or unexpected:
        raise ValueError(
            "Financial-GPT monitor requires exactly four global runs; "
            f"missing_global={missing_global}, unexpected={unexpected}"
        )


def _validate_expected_runs(
    local_run_uris: Mapping[str, str],
    global_run_uris: Mapping[str, str],
) -> None:
    missing_local = [value for value in LOCAL_ALGORITHMS if value not in local_run_uris]
    missing_global = [
        value for value in GLOBAL_ARCHITECTURES if value not in global_run_uris
    ]
    if missing_local or missing_global:
        raise ValueError(
            f"Monitor requires four local and four global runs; "
            f"missing_local={missing_local}, missing_global={missing_global}"
        )


def _discover_latest_local_runs_path(
    root: str,
    algorithms: Sequence[str],
) -> Mapping[str, str]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    result: Dict[str, str] = {}
    for algorithm in algorithms:
        matches = sorted(
            path
            for path in base.rglob(f"{algorithm}_*")
            if path.is_dir() and (path / "data/backtest_timeseries.csv").is_file()
        )
        if not matches:
            raise FileNotFoundError(f"No local run found for {algorithm}")
        result[str(algorithm)] = str(matches[-1])
    return result


def _discover_latest_local_runs_s3(
    root: str,
    algorithms: Sequence[str],
    *,
    s3_client=None,
) -> Mapping[str, str]:
    parsed = urlparse(root)
    api = s3_client or _default_s3_client()
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    dates = _list_s3_prefixes(api, parsed.netloc, prefix)
    date_prefixes = sorted(
        value
        for value in dates
        if re.fullmatch(r"\d{8}/", value[len(prefix) :])
    )
    if not date_prefixes:
        raise FileNotFoundError(f"No dated execution folders under {root}")
    result: Dict[str, str] = {}
    for algorithm in algorithms:
        candidates: list[str] = []
        for date_prefix in reversed(date_prefixes):
            runs = _list_s3_prefixes(
                api,
                parsed.netloc,
                date_prefix,
            )
            matching = [
                value
                for value in runs
                if Path(value.rstrip("/")).name.startswith(f"{algorithm}_")
            ]
            if matching:
                candidates.extend(matching)
                break
        if not candidates:
            raise FileNotFoundError(f"No local run found for {algorithm}")
        selected = sorted(candidates)[-1].rstrip("/")
        result[str(algorithm)] = f"s3://{parsed.netloc}/{selected}"
    return result


def _list_s3_prefixes(client, bucket: str, prefix: str) -> list[str]:
    values: list[str] = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "Delimiter": "/",
        }
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        values.extend(item["Prefix"] for item in response.get("CommonPrefixes", []))
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return values


def _read_dataframe(uri: str, *, s3_client=None) -> pd.DataFrame:
    suffix = Path(urlparse(uri).path).suffix.lower()
    payload = _read_bytes(uri, s3_client=s3_client)
    if suffix == ".csv":
        return pd.read_csv(BytesIO(payload), low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.DataFrame(pl.read_parquet(BytesIO(payload)).to_dicts())
    raise ValueError(f"Unsupported dataframe format: {uri}")


def _read_json(uri: str, *, s3_client=None) -> Mapping[str, Any]:
    return json.loads(_read_bytes(uri, s3_client=s3_client).decode("utf-8"))


def _read_bytes(uri: str, *, s3_client=None) -> bytes:
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        api = s3_client or _default_s3_client()
        return api.get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )["Body"].read()
    return Path(uri).expanduser().read_bytes()


def _join_uri(base: str, relative: str) -> str:
    if str(base).startswith("s3://"):
        return str(base).rstrip("/") + "/" + str(relative).lstrip("/")
    return str(Path(base).expanduser() / relative)


def _drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[value for value in frame.columns if str(value).startswith("Unnamed")],
        errors="ignore",
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _default_s3_client():
    import boto3

    return boto3.client("s3")
