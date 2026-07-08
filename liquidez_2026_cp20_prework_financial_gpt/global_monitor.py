"""Monitor final para comparar las cuatro variantes globales Financial-GPT.

El monitor no reentrena. Lee artefactos ya producidos, compara métricas por
``cross_key_id`` y selecciona una arquitectura ganadora por serie. El ensemble
final conserva el forecast de la arquitectura ganadora de cada identidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import polars as pl

from global_contracts import CROSS_KEY_COLUMN, SUPPORTED_ARCHITECTURES
from global_notebook import read_polars_uri


ERROR_METRICS: Tuple[str, ...] = ("MAE", "RMSE", "MAPE", "sMAPE", "WMAPE", "MASE", "MedAE", "MedAPE")
QUALITY_METRICS: Tuple[str, ...] = ("EVS", "R2")
REQUIRED_REPORT_FILES: Tuple[str, ...] = (
    "evaluation_metrics.parquet",
    "backtest_metrics_by_series.parquet",
    "future_forecast_mc_by_series.parquet",
)


@dataclass(frozen=True)
class GlobalRunReport:
    architecture: str
    run_uri: str
    evaluation_metrics: pl.DataFrame
    backtest_metrics: pl.DataFrame
    future_forecast: pl.DataFrame


@dataclass(frozen=True)
class GlobalMonitoringResult:
    run_summary: pl.DataFrame
    metrics_by_series: pl.DataFrame
    winners_by_series: pl.DataFrame
    ensemble_forecast: pl.DataFrame

    def write(self, output_directory: str | Path) -> Path:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self.run_summary.write_parquet(destination / "global_run_summary.parquet")
        self.metrics_by_series.write_parquet(destination / "global_metrics_by_series.parquet")
        self.winners_by_series.write_parquet(destination / "global_winners_by_series.parquet")
        self.ensemble_forecast.write_parquet(destination / "financial_gpt_ensemble_forecast.parquet")
        return destination


def load_global_run_report(run_uri: str) -> GlobalRunReport:
    base = str(run_uri).strip().rstrip("/")
    if not base:
        raise ValueError("run_uri must not be empty")
    manifest = _read_json(_join_uri(base, "model/manifest.json"))
    architecture = str(manifest.get("architecture", "")).strip().lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Invalid or missing architecture in {base!r}: {architecture!r}")

    evaluation = read_polars_uri(_join_uri(base, "reports/evaluation_metrics.parquet"))
    backtest = read_polars_uri(_join_uri(base, "reports/backtest_metrics_by_series.parquet"))
    forecast = read_polars_uri(_join_uri(base, "reports/future_forecast_mc_by_series.parquet"))
    if "serie" not in backtest.columns:
        raise ValueError(f"Backtest metrics for {architecture} are missing 'serie'")
    if CROSS_KEY_COLUMN not in forecast.columns:
        raise ValueError(f"Forecast for {architecture} is missing {CROSS_KEY_COLUMN!r}")
    return GlobalRunReport(
        architecture=architecture,
        run_uri=base,
        evaluation_metrics=evaluation.with_columns(pl.lit(architecture).alias("architecture")),
        backtest_metrics=backtest.with_columns(
            pl.col("serie").cast(pl.String).alias(CROSS_KEY_COLUMN),
            pl.lit(architecture).alias("architecture"),
        ),
        future_forecast=forecast.with_columns(pl.lit(architecture).alias("architecture")),
    )


def compare_global_runs(
    run_uris: Sequence[str],
    *,
    metrics: Sequence[str] = ("MAE", "RMSE", "WMAPE", "EVS", "R2"),
) -> GlobalMonitoringResult:
    if len(run_uris) < 2:
        raise ValueError("At least two global run URIs are required")
    reports = [load_global_run_report(uri) for uri in run_uris]
    architectures = [report.architecture for report in reports]
    if len(set(architectures)) != len(architectures):
        raise ValueError("Each architecture may appear only once")

    evaluation = pl.concat([report.evaluation_metrics for report in reports], how="diagonal_relaxed")
    backtest = pl.concat([report.backtest_metrics for report in reports], how="diagonal_relaxed")
    forecast = pl.concat([report.future_forecast for report in reports], how="diagonal_relaxed")

    selected_metrics = tuple(str(metric) for metric in metrics)
    missing = [metric for metric in selected_metrics if metric not in backtest.columns]
    if missing:
        raise ValueError(f"Backtest metrics are missing winner metrics: {missing}")
    ranked = _rank_architectures(backtest, selected_metrics)
    winners = (
        ranked.sort([CROSS_KEY_COLUMN, "rank_score", "architecture"])
        .group_by(CROSS_KEY_COLUMN, maintain_order=True)
        .first()
        .select(
            CROSS_KEY_COLUMN,
            pl.col("architecture").alias("winner_architecture"),
            "rank_score",
            *selected_metrics,
        )
        .sort(CROSS_KEY_COLUMN)
    )
    ensemble = (
        forecast.join(winners.select(CROSS_KEY_COLUMN, "winner_architecture"), on=CROSS_KEY_COLUMN, how="inner")
        .filter(pl.col("architecture") == pl.col("winner_architecture"))
        .sort([CROSS_KEY_COLUMN, "date"])
    )
    run_summary = (
        evaluation.group_by(["architecture", "partition"])
        .agg([pl.col(column).mean().alias(column) for column in evaluation.columns if column not in {"architecture", "partition"} and evaluation.schema[column].is_numeric()])
        .sort(["partition", "architecture"])
    )
    return GlobalMonitoringResult(
        run_summary=run_summary,
        metrics_by_series=ranked.sort([CROSS_KEY_COLUMN, "rank_score", "architecture"]),
        winners_by_series=winners,
        ensemble_forecast=ensemble,
    )


def _rank_architectures(backtest: pl.DataFrame, metrics: Sequence[str]) -> pl.DataFrame:
    ranked = backtest
    rank_columns: list[str] = []
    for metric in metrics:
        rank_column = f"rank_{metric}"
        descending = metric in QUALITY_METRICS
        ranked = ranked.with_columns(
            pl.col(metric)
            .rank(method="average", descending=descending)
            .over(CROSS_KEY_COLUMN)
            .alias(rank_column)
        )
        rank_columns.append(rank_column)
    return ranked.with_columns(
        pl.sum_horizontal([pl.col(column) for column in rank_columns]).alias("rank_score")
    )


def _join_uri(base: str, relative: str) -> str:
    if base.startswith("s3://"):
        return base.rstrip("/") + "/" + relative.lstrip("/")
    return str(Path(base).expanduser() / relative)


def _read_json(uri: str) -> Mapping[str, Any]:
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        import boto3
        payload = boto3.client("s3").get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )["Body"].read()
        return json.loads(payload.decode("utf-8"))
    return json.loads(Path(uri).expanduser().read_text(encoding="utf-8"))
