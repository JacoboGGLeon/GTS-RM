"""Eje temporal agnóstico para Financial-GPT.

El modelo no interpreta fines de semana, festivos ni frecuencias. Cada fila del
proveedor temporal representa un paso válido y contiene las covariables que el
modelo puede utilizar. Las series aportan targets; ``TemporalAxis`` aporta los
timestamps válidos y ``TemporalWindowAligner`` construye la intersección de
forma auditable sin modificar el frame fuente.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from global_contracts import CROSS_KEY_COLUMN, DATE_COLUMN


class TemporalAlignmentError(ValueError):
    """Error base para contratos temporales inválidos."""


class InsufficientFutureContextError(TemporalAlignmentError):
    """El eje no contiene suficientes pasos futuros para la petición."""


@dataclass(frozen=True)
class ForecastRequest:
    """Petición de forecast por pasos del eje o por rango de timestamps.

    Se debe utilizar exactamente uno de los dos modos:

    - ``n_steps``: próximas N filas válidas del eje después del ancla;
    - ``start``/``end``: filas del eje dentro del rango y posteriores al ancla.
    """

    n_steps: int | None = None
    start: Any | None = None
    end: Any | None = None

    def validate(self) -> None:
        uses_steps = self.n_steps is not None
        uses_range = self.start is not None or self.end is not None
        if uses_steps == uses_range:
            raise ValueError("ForecastRequest requires exactly one mode: n_steps or start/end")
        if uses_steps:
            if isinstance(self.n_steps, bool) or int(self.n_steps) <= 0:
                raise ValueError("n_steps must be a positive integer")
            return
        if self.start is None or self.end is None:
            raise ValueError("Both start and end are required for range mode")
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError("end must be greater than or equal to start")


@dataclass(frozen=True)
class TemporalAlignmentSummary:
    cross_key_id: str
    original_rows: int
    aligned_rows: int
    excluded_rows: int
    coverage_ratio: float
    first_original_timestamp: str | None
    last_original_timestamp: str | None
    first_aligned_timestamp: str | None
    last_aligned_timestamp: str | None

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class TemporalAxis:
    """Índice ordenado de pasos temporales válidos y sus covariables."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = DATE_COLUMN,
        feature_columns: Sequence[str] = (),
    ) -> None:
        timestamp_column = str(timestamp_column)
        features = tuple(str(value) for value in feature_columns)
        if timestamp_column not in frame.columns:
            raise ValueError(f"Temporal frame is missing {timestamp_column!r}")
        missing = sorted(set(features).difference(frame.columns))
        if missing:
            raise ValueError(f"Temporal frame is missing feature columns: {missing}")
        if timestamp_column in features:
            raise ValueError("timestamp column cannot be a temporal feature")
        if len(set(features)) != len(features):
            raise ValueError("feature_columns must be unique")

        prepared = frame.loc[:, [timestamp_column, *features]].copy()
        prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
        if prepared[timestamp_column].isna().any():
            raise ValueError("Temporal axis contains invalid timestamps")
        if prepared[timestamp_column].duplicated().any():
            raise ValueError("Temporal axis must contain one row per timestamp")
        prepared = prepared.sort_values(timestamp_column).set_index(timestamp_column)
        if not prepared.index.is_monotonic_increasing:
            raise RuntimeError("Temporal axis timestamps are not ordered")
        for column in features:
            numeric = pd.to_numeric(prepared[column], errors="coerce")
            if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError(f"Temporal feature {column!r} must be finite and numeric")
            prepared[column] = numeric.astype(float)

        self._frame = prepared
        self.timestamp_column = timestamp_column
        self.feature_columns: Tuple[str, ...] = features

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame | pd.DataFrame,
        *,
        timestamp_column: str = DATE_COLUMN,
        feature_columns: Sequence[str] = (),
    ) -> "TemporalAxis":
        if isinstance(frame, pl.DataFrame):
            pandas_frame = pd.DataFrame(frame.to_dicts())
        elif isinstance(frame, pd.DataFrame):
            pandas_frame = frame
        else:
            raise TypeError("frame must be a polars or pandas DataFrame")
        return cls(
            pandas_frame,
            timestamp_column=timestamp_column,
            feature_columns=feature_columns,
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._frame.index.copy()

    @property
    def first_timestamp(self) -> pd.Timestamp:
        if self._frame.empty:
            raise ValueError("Temporal axis is empty")
        return pd.Timestamp(self._frame.index[0])

    @property
    def last_timestamp(self) -> pd.Timestamp:
        if self._frame.empty:
            raise ValueError("Temporal axis is empty")
        return pd.Timestamp(self._frame.index[-1])

    def contains(self, timestamps: Sequence[Any] | pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        return index.isin(self._frame.index)

    def features_for(self, timestamps: Sequence[Any] | pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        missing = index.difference(self._frame.index)
        if len(missing):
            raise TemporalAlignmentError(
                f"Temporal axis is missing {len(missing)} requested timestamps; "
                f"first={missing[0]}"
            )
        if not self.feature_columns:
            return np.empty((len(index), 0), dtype=np.float32)
        return self._frame.loc[index, list(self.feature_columns)].to_numpy(dtype=np.float32)

    def after(self, anchor: Any, *, n_steps: int) -> pd.DatetimeIndex:
        if isinstance(n_steps, bool) or int(n_steps) <= 0:
            raise ValueError("n_steps must be a positive integer")
        anchor_ts = pd.Timestamp(anchor)
        candidates = self._frame.index[self._frame.index > anchor_ts]
        if len(candidates) < int(n_steps):
            raise InsufficientFutureContextError(
                "Temporal axis has insufficient future context: "
                f"requested_steps={int(n_steps)}, available_steps={len(candidates)}, "
                f"anchor={anchor_ts}, last_axis_timestamp={self.last_timestamp}"
            )
        return pd.DatetimeIndex(candidates[: int(n_steps)])

    def between(self, start: Any, end: Any, *, after: Any | None = None) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be greater than or equal to start")
        mask = (self._frame.index >= start_ts) & (self._frame.index <= end_ts)
        if after is not None:
            mask &= self._frame.index > pd.Timestamp(after)
        return pd.DatetimeIndex(self._frame.index[mask])

    def resolve(self, request: ForecastRequest, *, anchor: Any) -> pd.DatetimeIndex:
        request.validate()
        if request.n_steps is not None:
            return self.after(anchor, n_steps=int(request.n_steps))
        selected = self.between(request.start, request.end, after=anchor)
        if selected.empty:
            raise InsufficientFutureContextError(
                "Temporal axis has no valid timestamps for the requested range: "
                f"anchor={pd.Timestamp(anchor)}, start={pd.Timestamp(request.start)}, "
                f"end={pd.Timestamp(request.end)}"
            )
        return selected

    def to_polars(self) -> pl.DataFrame:
        frame = self._frame.reset_index()
        output = pl.from_pandas(frame)
        # El contrato canónico actual trabaja con fechas diarias; si el proveedor
        # usa timestamps intradía, Polars conserva Datetime.
        return output.rename({self.timestamp_column: DATE_COLUMN}) if self.timestamp_column != DATE_COLUMN else output


class TemporalWindowAligner:
    """Alinea targets con un ``TemporalAxis`` y reporta la cobertura."""

    def __init__(self, axis: TemporalAxis) -> None:
        if not isinstance(axis, TemporalAxis):
            raise TypeError("axis must be a TemporalAxis")
        self.axis = axis

    def align_global_long(
        self,
        global_long: pl.DataFrame,
        *,
        cross_key_column: str = CROSS_KEY_COLUMN,
        timestamp_column: str = DATE_COLUMN,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if not isinstance(global_long, pl.DataFrame):
            raise TypeError("global_long must be a polars DataFrame")
        required = {cross_key_column, timestamp_column}
        missing = sorted(required.difference(global_long.columns))
        if missing:
            raise ValueError(f"global_long is missing columns: {missing}")

        axis_frame = self.axis.to_polars()
        if timestamp_column != DATE_COLUMN:
            axis_frame = axis_frame.rename({DATE_COLUMN: timestamp_column})
        target_dtype = global_long.schema[timestamp_column]
        axis_frame = axis_frame.with_columns(
            pl.col(timestamp_column).cast(target_dtype, strict=False)
        ).drop_nulls([timestamp_column])

        aligned = global_long.join(axis_frame, on=timestamp_column, how="inner")
        reports: list[dict[str, Any]] = []
        for raw_key, source in global_long.partition_by(
            cross_key_column, as_dict=True, maintain_order=False
        ).items():
            key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
            series_id = str(key)
            target = aligned.filter(pl.col(cross_key_column) == key)
            summary = self._summary(series_id, source, target, timestamp_column)
            reports.append(dict(summary.to_dict()))
        report = pl.DataFrame(reports).sort(cross_key_column)
        return aligned.sort([cross_key_column, timestamp_column]), report

    @staticmethod
    def _summary(
        series_id: str,
        source: pl.DataFrame,
        aligned: pl.DataFrame,
        timestamp_column: str,
    ) -> TemporalAlignmentSummary:
        original_rows = int(source.height)
        aligned_rows = int(aligned.height)
        def iso(value: Any | None) -> str | None:
            return None if value is None else pd.Timestamp(value).isoformat()
        return TemporalAlignmentSummary(
            cross_key_id=series_id,
            original_rows=original_rows,
            aligned_rows=aligned_rows,
            excluded_rows=original_rows - aligned_rows,
            coverage_ratio=(aligned_rows / original_rows) if original_rows else 0.0,
            first_original_timestamp=iso(source.get_column(timestamp_column).min()),
            last_original_timestamp=iso(source.get_column(timestamp_column).max()),
            first_aligned_timestamp=iso(
                aligned.get_column(timestamp_column).min() if aligned_rows else None
            ),
            last_aligned_timestamp=iso(
                aligned.get_column(timestamp_column).max() if aligned_rows else None
            ),
        )
