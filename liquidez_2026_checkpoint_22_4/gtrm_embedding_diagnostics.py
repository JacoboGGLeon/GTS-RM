"""Export y diagnóstico de ``history_embedding`` para GTRM Stage 1.

Checkpoint 21.3 convierte el embedding histórico en un artefacto auditable:

- exporta embeddings por ventana con metadata sólo de trazabilidad;
- valida finitud, dimensión estable y columnas latentes;
- calcula diagnósticos de norma, colapso de dimensiones y drift temporal;
- resume comportamiento por serie y por cohortes opcionales.

El módulo no modifica el entrenamiento ni introduce residual local, cuantiles,
patching o SSL. Su única responsabilidad es observar la representación que ya
produce el Global Representation Base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from global_contracts import CROSS_KEY_COLUMN, HISTORY_EMBEDDING_FIELD


DEFAULT_EMBEDDING_VARIANCE_EPSILON: float = 1e-12
DEFAULT_NEAR_DUPLICATE_DECIMALS: int = 8


@dataclass(frozen=True)
class HistoryEmbeddingDiagnosticsCriteria:
    """Umbrales deterministas para diagnosticar la representación Stage 1."""

    variance_epsilon: float = DEFAULT_EMBEDDING_VARIANCE_EPSILON
    near_duplicate_decimals: int = DEFAULT_NEAR_DUPLICATE_DECIMALS
    min_embedding_dim: int = 1

    def validate(self) -> None:
        if not np.isfinite(self.variance_epsilon) or float(self.variance_epsilon) < 0.0:
            raise ValueError("variance_epsilon must be a non-negative finite value")
        if isinstance(self.near_duplicate_decimals, bool) or int(self.near_duplicate_decimals) < 0:
            raise ValueError("near_duplicate_decimals must be a non-negative integer")
        if isinstance(self.min_embedding_dim, bool) or int(self.min_embedding_dim) <= 0:
            raise ValueError("min_embedding_dim must be a positive integer")


@dataclass(frozen=True)
class HistoryEmbeddingDiagnostics:
    """Colección de artefactos diagnósticos de embeddings GTRM."""

    summary: Mapping[str, Any]
    dimension_report: pd.DataFrame
    by_series: pd.DataFrame
    by_cohort: Mapping[str, pd.DataFrame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "dimension_report": self.dimension_report.to_dict(orient="records"),
            "by_series": self.by_series.to_dict(orient="records"),
            "by_cohort": {
                name: frame.to_dict(orient="records")
                for name, frame in self.by_cohort.items()
            },
        }

    def write(self, output_directory: str | Path) -> Path:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "history_embedding_diagnostics_summary.json").write_text(
            json.dumps(dict(self.summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.dimension_report.to_csv(
            destination / "history_embedding_dimension_report.csv", index=False
        )
        self.by_series.to_csv(
            destination / "history_embedding_by_series_summary.csv", index=False
        )
        for name, frame in self.by_cohort.items():
            safe_name = _safe_artifact_name(name)
            frame.to_csv(destination / f"history_embedding_by_{safe_name}.csv", index=False)
        return destination


def embedding_columns(
    frame: pd.DataFrame,
    *,
    prefix: str = HISTORY_EMBEDDING_FIELD,
) -> tuple[str, ...]:
    """Devuelve columnas latentes en orden numérico estable."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("history embedding frame must be a non-empty pandas DataFrame")
    marker = f"{prefix}_"
    columns = [str(column) for column in frame.columns if str(column).startswith(marker)]
    if not columns:
        raise ValueError(f"No embedding columns found with prefix {marker!r}")

    def _sort_key(name: str) -> tuple[int, str]:
        suffix = name[len(marker):]
        return (int(suffix) if suffix.isdigit() else 10**9, name)

    return tuple(sorted(columns, key=_sort_key))


def validate_history_embedding_frame(
    frame: pd.DataFrame,
    *,
    criteria: HistoryEmbeddingDiagnosticsCriteria | None = None,
) -> tuple[str, ...]:
    """Valida que el frame tenga embeddings finitos y dimensión estable."""

    selected = criteria or HistoryEmbeddingDiagnosticsCriteria()
    selected.validate()
    latent_columns = embedding_columns(frame)
    if len(latent_columns) < int(selected.min_embedding_dim):
        raise ValueError(
            "Embedding dimension is below the required minimum: "
            f"{len(latent_columns)} < {selected.min_embedding_dim}"
        )
    values = frame.loc[:, latent_columns].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[0] != len(frame):
        raise ValueError("Embedding matrix must have shape [rows, embedding_dim]")
    if not np.all(np.isfinite(values)):
        raise ValueError("History embedding frame contains non-finite latent values")
    if "embedding_dim" in frame.columns:
        dims = set(pd.to_numeric(frame["embedding_dim"], errors="raise").astype(int).tolist())
        if dims != {len(latent_columns)}:
            raise ValueError(
                "embedding_dim column is inconsistent with latent columns: "
                f"{sorted(dims)} vs {len(latent_columns)}"
            )
    return latent_columns


def build_history_embedding_diagnostics(
    embeddings: pd.DataFrame,
    *,
    criteria: HistoryEmbeddingDiagnosticsCriteria | None = None,
    cohort_columns: Sequence[str] = (),
) -> HistoryEmbeddingDiagnostics:
    """Construye resumen, diagnóstico dimensional y drift por serie/cohorte."""

    selected = criteria or HistoryEmbeddingDiagnosticsCriteria()
    selected.validate()
    frame = embeddings.copy()
    latent_columns = validate_history_embedding_frame(frame, criteria=selected)
    matrix = frame.loc[:, latent_columns].to_numpy(dtype=np.float64, copy=True)
    norms = np.linalg.norm(matrix, axis=1)
    frame["embedding_norm"] = norms

    dimension_report = _dimension_report(
        matrix,
        latent_columns,
        variance_epsilon=float(selected.variance_epsilon),
    )
    by_series = _by_series_report(frame, latent_columns)
    by_cohort = _by_cohort_reports(frame, cohort_columns=cohort_columns)

    rounded = np.round(matrix, decimals=int(selected.near_duplicate_decimals))
    unique_rounded = np.unique(rounded, axis=0).shape[0]
    collapsed_dimensions = int(dimension_report["is_collapsed"].sum())
    summary: dict[str, Any] = {
        "num_embeddings": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "num_unique_embeddings_rounded": int(unique_rounded),
        "near_duplicate_embedding_fraction": float(1.0 - unique_rounded / max(matrix.shape[0], 1)),
        "collapsed_dimensions": collapsed_dimensions,
        "collapsed_dimension_fraction": float(collapsed_dimensions / max(matrix.shape[1], 1)),
        "embedding_norm_mean": float(np.mean(norms)),
        "embedding_norm_std": float(np.std(norms)),
        "embedding_norm_min": float(np.min(norms)),
        "embedding_norm_p50": float(np.percentile(norms, 50)),
        "embedding_norm_p90": float(np.percentile(norms, 90)),
        "embedding_norm_max": float(np.max(norms)),
        "criteria": asdict(selected),
    }
    if CROSS_KEY_COLUMN in frame.columns:
        summary["num_series"] = int(frame[CROSS_KEY_COLUMN].nunique(dropna=False))
    if "cutoff" in frame.columns:
        summary["num_cutoffs"] = int(frame["cutoff"].nunique(dropna=False))
    return HistoryEmbeddingDiagnostics(
        summary=summary,
        dimension_report=dimension_report,
        by_series=by_series,
        by_cohort=by_cohort,
    )


def write_history_embedding_artifacts(
    embeddings: pd.DataFrame,
    output_directory: str | Path,
    *,
    diagnostics: HistoryEmbeddingDiagnostics | None = None,
    criteria: HistoryEmbeddingDiagnosticsCriteria | None = None,
    cohort_columns: Sequence[str] = (),
    write_parquet: bool = True,
) -> Path:
    """Escribe embeddings y reportes diagnósticos en un directorio.

    Siempre escribe CSV para portabilidad. Si ``write_parquet`` está activo,
    intenta escribir Parquet; si el entorno no tiene engine de Parquet, registra
    el motivo en ``history_embeddings_parquet_status.json`` sin fallar el gate.
    """

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    latent_columns = validate_history_embedding_frame(embeddings, criteria=criteria)
    embeddings.to_csv(destination / "history_embeddings.csv", index=False)
    parquet_status: dict[str, Any] = {"requested": bool(write_parquet), "written": False}
    if write_parquet:
        try:
            embeddings.to_parquet(destination / "history_embeddings.parquet", index=False)
        except Exception as exc:  # pragma: no cover - depends on optional parquet engine
            parquet_status["error"] = f"{type(exc).__name__}: {exc}"
        else:
            parquet_status["written"] = True
    (destination / "history_embeddings_schema.json").write_text(
        json.dumps(
            {
                "latent_field": HISTORY_EMBEDDING_FIELD,
                "embedding_columns": list(latent_columns),
                "embedding_dim": len(latent_columns),
                "metadata_columns": [
                    column for column in embeddings.columns if column not in latent_columns
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (destination / "history_embeddings_parquet_status.json").write_text(
        json.dumps(parquet_status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    selected_diagnostics = diagnostics or build_history_embedding_diagnostics(
        embeddings,
        criteria=criteria,
        cohort_columns=cohort_columns,
    )
    selected_diagnostics.write(destination)
    return destination


def _dimension_report(
    matrix: np.ndarray,
    columns: Sequence[str],
    *,
    variance_epsilon: float,
) -> pd.DataFrame:
    variances = np.var(matrix, axis=0)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    return pd.DataFrame(
        {
            "embedding_column": list(columns),
            "dimension_index": list(range(len(columns))),
            "mean": means,
            "std": stds,
            "variance": variances,
            "min": mins,
            "max": maxs,
            "is_collapsed": variances <= float(variance_epsilon),
        }
    )


def _by_series_report(frame: pd.DataFrame, latent_columns: Sequence[str]) -> pd.DataFrame:
    if CROSS_KEY_COLUMN not in frame.columns:
        result = pd.DataFrame(
            {
                "window_count": [int(len(frame))],
                "embedding_norm_mean": [float(frame["embedding_norm"].mean())],
                "embedding_norm_std": [float(frame["embedding_norm"].std(ddof=0))],
                "embedding_drift_mean": [float("nan")],
                "embedding_drift_p90": [float("nan")],
            }
        )
        return result
    rows: list[dict[str, Any]] = []
    sort_columns = [CROSS_KEY_COLUMN]
    if "cutoff" in frame.columns:
        sort_columns.append("cutoff")
    for series_id, group in frame.sort_values(sort_columns).groupby(CROSS_KEY_COLUMN, dropna=False):
        vectors = group.loc[:, latent_columns].to_numpy(dtype=np.float64, copy=True)
        drift = _successive_distances(vectors)
        record: dict[str, Any] = {
            CROSS_KEY_COLUMN: series_id,
            "window_count": int(len(group)),
            "embedding_dim": int(vectors.shape[1]),
            "embedding_norm_mean": float(group["embedding_norm"].mean()),
            "embedding_norm_std": float(group["embedding_norm"].std(ddof=0)),
            "embedding_drift_mean": _safe_mean(drift),
            "embedding_drift_p90": _safe_percentile(drift, 90),
        }
        for metadata_column in ("account_currency_id", "divisa", "tipo_serie", "grupo", "nivel_curriculum"):
            if metadata_column in group.columns:
                values = group[metadata_column].drop_duplicates().tolist()
                record[metadata_column] = values[0] if len(values) == 1 else "__MIXED__"
        rows.append(record)
    return pd.DataFrame(rows)


def _by_cohort_reports(
    frame: pd.DataFrame,
    *,
    cohort_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    for column in cohort_columns:
        if column not in frame.columns:
            continue
        rows: list[dict[str, Any]] = []
        for value, group in frame.groupby(column, dropna=False):
            record: dict[str, Any] = {
                column: value,
                "window_count": int(len(group)),
                "embedding_norm_mean": float(group["embedding_norm"].mean()),
                "embedding_norm_std": float(group["embedding_norm"].std(ddof=0)),
            }
            if CROSS_KEY_COLUMN in group.columns:
                record["series_count"] = int(group[CROSS_KEY_COLUMN].nunique(dropna=False))
            rows.append(record)
        reports[str(column)] = pd.DataFrame(rows)
    return reports


def _successive_distances(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(np.diff(vectors, axis=0), axis=1)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _safe_artifact_name(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)
    return safe or "cohort"
