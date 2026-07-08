"""Soporte del notebook global parametrizable de Financial-GFM.

Checkpoint 19 concentra aquí la construcción causal de la representación:

- lectura local o S3 de ``global_series_long`` y calendario;
- alineación sobre el eje temporal suministrado por el proveedor;
- split por ``account_currency_id`` antes de ajustar categorías o estadísticas;
- holdout temporal para ``validation_seen`` sin targets compartidos con train;
- dificultad curricular calculada por serie final usando sólo targets de train;
- exógenas temporales estandarizadas sólo con fechas de train;
- ``x_static`` train-only con tipo, divisa, escala contextual y edad causal;
- reconstrucción determinista de datasets para cada ``window_size``.

Los identificadores contables se usan únicamente para particionar y auditar. No
se incorporan a ``model_inputs``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Final, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import polars as pl

from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    CURRICULUM_COLUMN,
    DATE_COLUMN,
    DIFFICULTY_COLUMN,
    GLOBAL_LONG_REQUIRED_COLUMNS,
    GROUP_COLUMN,
    SERIES_AGE_COLUMN,
    SERIES_TYPE_COLUMN,
    SUPPORTED_ARCHITECTURES,
    TARGET_COLUMN,
)
from global_data import (
    ContextScaler,
    GlobalSeriesSplit,
    GlobalWindowDataset,
    StaticFeatureEncoder,
    robust_mase_scale,
)
from global_long_schema import upgrade_global_long_checkpoint19, validate_global_long
from temporal_axis import TemporalAxis, TemporalWindowAligner
from global_training import GlobalDatasetBundle


SUPPORTED_INPUT_SUFFIXES: Final[Tuple[str, ...]] = (".parquet", ".csv")
DEFAULT_MAX_WINDOW_SIZE: Final[int] = 25


@dataclass(frozen=True)
class GlobalNotebookConfig:
    """Parámetros serializables de una ejecución del notebook global."""

    architecture: str
    global_long_uri: str
    calendar_uri: str
    artifact_root: str
    horizon: int = 25
    seen_validation_size: int = 50
    validation_unseen_fraction: float = 0.15
    test_unseen_fraction: float = 0.15
    stride: int = 1
    exogenous_columns: Tuple[str, ...] = ()
    calendar_date_column: str = DATE_COLUMN
    n_trials: int = 15
    hpo_timeout_seconds: float | None = None
    seed: int = 42
    max_window_size: int = DEFAULT_MAX_WINDOW_SIZE
    artifact_s3_uri: str = ""

    def validate(self) -> None:
        architecture = str(self.architecture).strip().lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture={self.architecture!r}; "
                f"expected {SUPPORTED_ARCHITECTURES}"
            )
        for label, value in (
            ("horizon", self.horizon),
            ("seen_validation_size", self.seen_validation_size),
            ("stride", self.stride),
            ("n_trials", self.n_trials),
            ("max_window_size", self.max_window_size),
        ):
            _positive_int(value, label)
        if self.seen_validation_size < self.horizon:
            raise ValueError("seen_validation_size must be at least horizon")
        if self.max_window_size < 3:
            raise ValueError("max_window_size must be at least 3")
        for label, value in (
            ("validation_unseen_fraction", self.validation_unseen_fraction),
            ("test_unseen_fraction", self.test_unseen_fraction),
        ):
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError(f"{label} must be in the open interval (0, 1)")
        if self.validation_unseen_fraction + self.test_unseen_fraction >= 1.0:
            raise ValueError("unseen fractions must sum to less than 1")
        if self.hpo_timeout_seconds is not None:
            if not math.isfinite(float(self.hpo_timeout_seconds)) or float(
                self.hpo_timeout_seconds
            ) <= 0:
                raise ValueError("hpo_timeout_seconds must be positive when provided")
        if not str(self.global_long_uri).strip():
            raise ValueError("global_long_uri must not be empty")
        if not str(self.calendar_uri).strip():
            raise ValueError("calendar_uri must not be empty")
        if not str(self.artifact_root).strip():
            raise ValueError("artifact_root must not be empty")
        if not str(self.calendar_date_column).strip():
            raise ValueError("calendar_date_column must not be empty")
        normalized_columns = tuple(str(value).strip() for value in self.exogenous_columns)
        if any(not value for value in normalized_columns):
            raise ValueError("exogenous_columns must not contain empty names")
        if len(set(normalized_columns)) != len(normalized_columns):
            raise ValueError("exogenous_columns must not contain duplicates")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalInputFrames:
    """Inputs canónicos cargados por el notebook."""

    global_long: pl.DataFrame
    calendar: pl.DataFrame
    exogenous_columns: Tuple[str, ...]
    global_long_uri: str
    calendar_uri: str


@dataclass(frozen=True)
class GlobalPreparedFrames:
    """Frames temporales usados para construir datasets de una ventana."""

    train: pl.DataFrame
    validation_seen: pl.DataFrame
    validation_unseen: pl.DataFrame
    test_unseen: pl.DataFrame


@dataclass(frozen=True)
class ExogenousFeatureScaler:
    """Estandarizador ajustado sólo con fechas presentes en train.

    Columnas binarias se preservan exactamente en 0/1. Las demás columnas
    numéricas usan media y desviación estándar de train.
    """

    columns: Tuple[str, ...]
    modes: Mapping[str, str]
    means: Mapping[str, float]
    stds: Mapping[str, float]

    @classmethod
    def fit(
        cls,
        calendar: pl.DataFrame,
        *,
        columns: Sequence[str],
        train_dates: Sequence[object],
    ) -> "ExogenousFeatureScaler":
        selected = tuple(str(value) for value in columns)
        if not selected:
            raise ValueError("ExogenousFeatureScaler requires at least one column")
        train_calendar = calendar.filter(pl.col(DATE_COLUMN).is_in(list(train_dates)))
        if train_calendar.is_empty():
            raise ValueError("No calendar rows overlap the training dates")
        modes: Dict[str, str] = {}
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for column in selected:
            values = train_calendar[column].to_numpy().astype(np.float64, copy=False)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"Exogenous column {column!r} has invalid train values")
            unique = set(np.unique(values).tolist())
            if unique.issubset({0.0, 1.0}):
                modes[column] = "binary_identity"
                means[column] = 0.0
                stds[column] = 1.0
            else:
                mean = float(np.mean(values))
                std = float(np.std(values))
                modes[column] = "standardize"
                means[column] = mean
                stds[column] = std if np.isfinite(std) and std > 1e-12 else 1.0
        return cls(selected, modes, means, stds)

    def transform(self, calendar: pl.DataFrame) -> pl.DataFrame:
        expressions: list[pl.Expr] = [pl.col(DATE_COLUMN)]
        for column in self.columns:
            expression = pl.col(column).cast(pl.Float64)
            if self.modes[column] == "standardize":
                expression = (expression - self.means[column]) / self.stds[column]
            expressions.append(expression.alias(column))
        result = calendar.select(expressions).sort(DATE_COLUMN)
        for column in self.columns:
            if not result[column].is_finite().all():
                raise ValueError(f"Scaled exogenous column {column!r} is not finite")
        return result

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "fit_scope": "calendar_rows_used_by_training_targets_only",
            "columns": list(self.columns),
            "features": {
                column: {
                    "mode": self.modes[column],
                    "mean_train": self.means[column],
                    "std_train": self.stds[column],
                    "known_in_future": True,
                }
                for column in self.columns
            },
        }


class GlobalNotebookDatasetFactory:
    """Factoría determinista de datasets para HPO y entrenamiento productivo.

    El split de identidades se calcula una sola vez en ``__init__``. Cada llamada
    cambia únicamente ``window_size`` y reconstruye ventanas sobre las mismas
    identidades y sobre el mismo holdout temporal.
    """

    def __init__(
        self,
        global_long: pl.DataFrame,
        calendar: pl.DataFrame,
        *,
        exogenous_columns: Sequence[str],
        horizon: int,
        seen_validation_size: int,
        validation_unseen_fraction: float,
        test_unseen_fraction: float,
        stride: int = 1,
        seed: int = 42,
        max_window_size: int = DEFAULT_MAX_WINDOW_SIZE,
    ) -> None:
        global_long = upgrade_global_long_checkpoint19(global_long)
        global_long = global_long.select(GLOBAL_LONG_REQUIRED_COLUMNS)
        validate_global_long(global_long)
        _positive_int(horizon, "horizon")
        _positive_int(seen_validation_size, "seen_validation_size")
        _positive_int(stride, "stride")
        _positive_int(max_window_size, "max_window_size")
        if seen_validation_size < horizon:
            raise ValueError("seen_validation_size must be at least horizon")
        if max_window_size < 3:
            raise ValueError("max_window_size must be at least 3")

        self.horizon = int(horizon)
        self.seen_validation_size = int(seen_validation_size)
        self.stride = int(stride)
        self.seed = int(seed)
        self.max_window_size = int(max_window_size)
        self.exogenous_columns = tuple(str(value) for value in exogenous_columns)

        raw_calendar = prepare_calendar_frame(
            calendar,
            exogenous_columns=self.exogenous_columns,
            date_column=DATE_COLUMN,
        )[0]
        raw_axis = TemporalAxis.from_frame(
            raw_calendar,
            timestamp_column=DATE_COLUMN,
            feature_columns=self.exogenous_columns,
        )
        aligned_global_long, self.temporal_alignment_report = TemporalWindowAligner(
            raw_axis
        ).align_global_long(global_long)

        minimum_rows = self.seen_validation_size + self.max_window_size + self.horizon
        self.minimum_required_rows = int(minimum_rows)
        source_metadata = global_long.group_by(CROSS_KEY_COLUMN).agg(
            pl.first(ACCOUNT_CURRENCY_ID_COLUMN).alias(ACCOUNT_CURRENCY_ID_COLUMN),
            pl.first(CURRENCY_COLUMN).alias(CURRENCY_COLUMN),
            pl.first(SERIES_TYPE_COLUMN).alias(SERIES_TYPE_COLUMN),
            pl.first(GROUP_COLUMN).alias(GROUP_COLUMN),
        )
        aligned_stats = aligned_global_long.group_by(CROSS_KEY_COLUMN).agg(
            pl.len().alias("row_count"),
            pl.min(DATE_COLUMN).alias("first_date"),
            pl.max(DATE_COLUMN).alias("last_date"),
        )
        manifest = (
            source_metadata.join(aligned_stats, on=CROSS_KEY_COLUMN, how="left")
            .with_columns(pl.col("row_count").fill_null(0).cast(pl.UInt32))
            .with_columns(
                pl.lit(minimum_rows).alias("minimum_required_rows"),
                (pl.col("row_count") >= minimum_rows).alias("eligible"),
                pl.when(pl.col("row_count") == 0)
                .then(pl.lit("no_temporal_coverage"))
                .when(pl.col("row_count") >= minimum_rows)
                .then(pl.lit("ready"))
                .otherwise(pl.lit("insufficient_aligned_history"))
                .alias("status"),
            )
            .sort(CROSS_KEY_COLUMN)
        )
        self.eligibility_manifest = manifest
        eligible_ids = manifest.filter(pl.col("eligible"))[CROSS_KEY_COLUMN]
        if eligible_ids.len() < 3:
            raise ValueError(
                "At least three series must have >= "
                f"{minimum_rows} rows for the configured HPO window range"
            )
        eligible_frame = manifest.filter(pl.col("eligible")).select(CROSS_KEY_COLUMN)
        self.global_long = (
            aligned_global_long.join(eligible_frame, on=CROSS_KEY_COLUMN, how="semi")
            .select(GLOBAL_LONG_REQUIRED_COLUMNS)
            .sort([CROSS_KEY_COLUMN, DATE_COLUMN])
        )
        validate_global_long(self.global_long)

        # Split por cuenta-divisa antes de ajustar cualquier representación,
        # categoría, estadístico exógeno o dificultad curricular.
        self.split = GlobalSeriesSplit.create(
            self.global_long,
            validation_unseen_fraction=validation_unseen_fraction,
            test_unseen_fraction=test_unseen_fraction,
            seed=self.seed,
        )
        self._seen_target_start_dates = self._calculate_seen_target_start_dates()
        train_reference = self._training_reference_frame()
        self.mase_scale_manifest = _build_causal_mase_scale_manifest(
            self.global_long,
            split=self.split,
            holdout_size=self.seen_validation_size,
        )
        self._mase_scale_by_series = {
            str(row[CROSS_KEY_COLUMN]): float(row["mase_scale"])
            for row in self.mase_scale_manifest.to_dicts()
        }

        difficulty_table = _build_train_only_difficulty(train_reference)
        self.difficulty_manifest = difficulty_table
        self.global_long = (
            self.global_long.drop([DIFFICULTY_COLUMN, CURRICULUM_COLUMN])
            .join(difficulty_table, on=CROSS_KEY_COLUMN, how="left")
            .with_columns(
                pl.col(DIFFICULTY_COLUMN).fill_null(0.0),
                pl.col(CURRICULUM_COLUMN).fill_null(1).cast(pl.Int64),
            )
            .select(GLOBAL_LONG_REQUIRED_COLUMNS)
            .sort([CROSS_KEY_COLUMN, DATE_COLUMN])
        )
        validate_global_long(self.global_long)

        # Las categorías se aprenden sólo de las series de train. Divisas o tipos
        # unseen se asignan a __UNKNOWN__ sin ampliar el modelo después del fit.
        self.static_feature_encoder = StaticFeatureEncoder.fit(train_reference)

        train_dates = train_reference[DATE_COLUMN].unique().to_list()
        self.exogenous_scaler = ExogenousFeatureScaler.fit(
            raw_calendar,
            columns=self.exogenous_columns,
            train_dates=train_dates,
        )
        self.calendar = self.exogenous_scaler.transform(raw_calendar)
        self.calendar_checksum = _frame_sha256(self.calendar)
        self.temporal_axis = TemporalAxis.from_frame(
            self.calendar,
            timestamp_column=DATE_COLUMN,
            feature_columns=self.exogenous_columns,
        )

    @property
    def seen_target_start_dates(self) -> Mapping[str, str]:
        return dict(self._seen_target_start_dates)

    @property
    def eligible_series_count(self) -> int:
        return int(self.global_long.get_column(CROSS_KEY_COLUMN).n_unique())

    @property
    def scaler_contract(self) -> Mapping[str, float | str]:
        return ContextScaler().contract()

    @property
    def static_feature_contract(self) -> Mapping[str, object]:
        return self.static_feature_encoder.to_dict()

    @property
    def exogenous_contract(self) -> Mapping[str, Any]:
        return self.exogenous_scaler.to_dict()

    def __call__(self, window_size: int) -> GlobalDatasetBundle:
        frames = self.build_frames(window_size)
        bundle = GlobalDatasetBundle(
            train=self._dataset(frames.train, window_size),
            validation_seen=self._dataset(frames.validation_seen, window_size),
            validation_unseen=self._dataset(frames.validation_unseen, window_size),
        )
        bundle.validate()
        return bundle

    def build_test_unseen(self, window_size: int) -> GlobalWindowDataset:
        frames = self.build_frames(window_size)
        return self._dataset(frames.test_unseen, window_size)

    def build_frames(self, window_size: int) -> GlobalPreparedFrames:
        _positive_int(window_size, "window_size")
        if int(window_size) > self.max_window_size:
            raise ValueError(
                f"window_size={window_size} exceeds configured max_window_size="
                f"{self.max_window_size}"
            )
        train_parts: list[pl.DataFrame] = []
        validation_seen_parts: list[pl.DataFrame] = []
        for series_id in self.split.train_series:
            frame = self._series_frame(series_id)
            train_size = frame.height - self.seen_validation_size
            train_parts.append(frame.head(train_size))
            validation_seen_parts.append(
                frame.tail(int(window_size) + self.seen_validation_size)
            )

        validation_unseen = self._tail_frames(
            self.split.validation_unseen_series,
            int(window_size) + self.seen_validation_size,
        )
        test_unseen = self._tail_frames(
            self.split.test_unseen_series,
            int(window_size) + self.seen_validation_size,
        )
        prepared = GlobalPreparedFrames(
            train=pl.concat(train_parts).sort([CROSS_KEY_COLUMN, DATE_COLUMN]),
            validation_seen=pl.concat(validation_seen_parts).sort(
                [CROSS_KEY_COLUMN, DATE_COLUMN]
            ),
            validation_unseen=validation_unseen,
            test_unseen=test_unseen,
        )
        self._validate_prepared_frames(prepared, window_size=int(window_size))
        return prepared

    def summary(self) -> Mapping[str, Any]:
        return {
            "eligible_series": self.eligible_series_count,
            "excluded_series": int(
                self.eligibility_manifest.filter(~pl.col("eligible")).height
            ),
            "minimum_required_rows": self.minimum_required_rows,
            "scaler_contract": dict(self.scaler_contract),
            "split_unit": ACCOUNT_CURRENCY_ID_COLUMN,
            "paired_series_policy": "saldo/variacion and any other types stay in the same partition",
            "train_series": len(self.split.train_series),
            "validation_seen_series": len(self.split.validation_seen_series),
            "validation_unseen_series": len(self.split.validation_unseen_series),
            "test_unseen_series": len(self.split.test_unseen_series),
            "horizon": self.horizon,
            "seen_validation_size": self.seen_validation_size,
            "stride": self.stride,
            "max_window_size": self.max_window_size,
            "exogenous_columns": list(self.exogenous_columns),
            "exogenous_contract": dict(self.exogenous_contract),
            "calendar_checksum": self.calendar_checksum,
            "static_feature_contract": dict(self.static_feature_contract),
            "difficulty_contract": {
                "fit_scope": "train_targets_only",
                "identity": "cross_key_id_final",
                "easy_series_filtered": False,
                "components": ["histogram_entropy", "scaled_variance", "scaled_mean_abs_change"],
            },
            "mase_contract": {
                "metric": "robust_macro_mase",
                "denominator": "max(mean(abs(diff(reference_history))), 0.01*max(mean(abs(reference_history)),1), 1e-6)",
                "fit_scope": "pre-holdout history only per cross_key_id",
                "validation_target_used": False,
            },
            "temporal_axis_steps": len(self.temporal_axis.timestamps),
            "temporal_axis_first": self.temporal_axis.first_timestamp.isoformat(),
            "temporal_axis_last": self.temporal_axis.last_timestamp.isoformat(),
            "alignment_rows_excluded": int(
                self.temporal_alignment_report.get_column("excluded_rows").sum()
            ),
            "alignment_mean_coverage": float(
                self.temporal_alignment_report.get_column("coverage_ratio").mean()
            ),
        }

    def _dataset(self, frame: pl.DataFrame, window_size: int) -> GlobalWindowDataset:
        return GlobalWindowDataset(
            frame,
            window_size=int(window_size),
            horizon=self.horizon,
            exogenous=self.calendar,
            exogenous_columns=self.exogenous_columns,
            stride=self.stride,
            static_feature_encoder=self.static_feature_encoder,
            mase_scale_by_series={
                series_id: self._mase_scale_by_series[series_id]
                for series_id in frame.get_column(CROSS_KEY_COLUMN).unique().to_list()
            },
        )

    def _series_frame(self, series_id: str) -> pl.DataFrame:
        frame = self.global_long.filter(pl.col(CROSS_KEY_COLUMN) == str(series_id)).sort(
            DATE_COLUMN
        )
        if frame.is_empty():
            raise RuntimeError(f"Missing split identity {series_id!r}")
        return frame

    def _tail_frames(self, series_ids: Sequence[str], rows: int) -> pl.DataFrame:
        parts = [self._series_frame(series_id).tail(rows) for series_id in series_ids]
        if not parts:
            raise ValueError("Unseen partition must contain at least one series")
        return pl.concat(parts).sort([CROSS_KEY_COLUMN, DATE_COLUMN])

    def _training_reference_frame(self) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for series_id in self.split.train_series:
            frame = self._series_frame(series_id)
            train_size = frame.height - self.seen_validation_size
            if train_size <= 0:
                raise RuntimeError(f"Series {series_id!r} has no causal train partition")
            parts.append(frame.head(train_size))
        return pl.concat(parts).sort([CROSS_KEY_COLUMN, DATE_COLUMN])

    def _calculate_seen_target_start_dates(self) -> Mapping[str, str]:
        starts: Dict[str, str] = {}
        for series_id in self.split.train_series:
            frame = self._series_frame(series_id)
            target_start = frame.row(
                frame.height - self.seen_validation_size,
                named=True,
            )[DATE_COLUMN]
            starts[str(series_id)] = _date_to_iso(target_start)
        return starts

    def _validate_prepared_frames(
        self,
        prepared: GlobalPreparedFrames,
        *,
        window_size: int,
    ) -> None:
        train_ids = set(prepared.train.get_column(CROSS_KEY_COLUMN).unique().to_list())
        seen_ids = set(
            prepared.validation_seen.get_column(CROSS_KEY_COLUMN).unique().to_list()
        )
        unseen_ids = set(
            prepared.validation_unseen.get_column(CROSS_KEY_COLUMN).unique().to_list()
        )
        test_ids = set(prepared.test_unseen.get_column(CROSS_KEY_COLUMN).unique().to_list())
        if train_ids != set(self.split.train_series):
            raise RuntimeError("Prepared train identities differ from the fixed split")
        if seen_ids != train_ids:
            raise RuntimeError("Prepared validation_seen identities differ from train")
        if train_ids & unseen_ids or train_ids & test_ids or unseen_ids & test_ids:
            raise RuntimeError("Prepared seen/unseen identity partitions overlap")

        minimum_train_rows = window_size + self.horizon
        minimum_validation_rows = window_size + self.horizon
        for label, frame in (
            ("train", prepared.train),
            ("validation_seen", prepared.validation_seen),
            ("validation_unseen", prepared.validation_unseen),
            ("test_unseen", prepared.test_unseen),
        ):
            counts = frame.group_by(CROSS_KEY_COLUMN).len()
            minimum = int(counts.get_column("len").min())
            expected = minimum_train_rows if label == "train" else minimum_validation_rows
            if minimum < expected:
                raise RuntimeError(
                    f"{label} contains a series with {minimum} rows; requires {expected}"
                )

        for series_id, target_start_iso in self._seen_target_start_dates.items():
            train_frame = prepared.train.filter(pl.col(CROSS_KEY_COLUMN) == series_id)
            train_max = train_frame.get_column(DATE_COLUMN).max()
            if _date_to_iso(train_max) >= target_start_iso:
                raise RuntimeError(
                    f"Train targets overlap temporal holdout for {series_id!r}"
                )


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _normalized_histogram_entropy(values: np.ndarray, bins: int = 10) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or float(np.max(values) - np.min(values)) <= 1e-12:
        return 0.0
    counts, _ = np.histogram(values, bins=min(int(bins), max(2, values.size)))
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = float(np.log(len(counts)))
    return entropy / maximum if maximum > 0.0 else 0.0


def _build_causal_mase_scale_manifest(
    global_long: pl.DataFrame,
    *,
    split: GlobalSeriesSplit,
    holdout_size: int,
) -> pl.DataFrame:
    """Fit one robust MASE denominator per series without using holdout targets."""

    _positive_int(holdout_size, "holdout_size")
    train_ids = set(split.train_series)
    validation_ids = set(split.validation_unseen_series)
    test_ids = set(split.test_unseen_series)
    rows: list[dict[str, object]] = []
    for raw_key, frame in global_long.partition_by(
        CROSS_KEY_COLUMN, as_dict=True, maintain_order=False
    ).items():
        key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
        series_id = str(key)
        ordered = frame.sort(DATE_COLUMN)
        history_rows = ordered.height - int(holdout_size)
        if history_rows < 2:
            raise ValueError(
                f"Series {series_id!r} lacks pre-holdout history for MASE"
            )
        reference = ordered.head(history_rows)
        values = reference.get_column(TARGET_COLUMN).to_numpy()
        if series_id in train_ids:
            split_role = "train_seen"
        elif series_id in validation_ids:
            split_role = "validation_unseen"
        elif series_id in test_ids:
            split_role = "test_unseen"
        else:
            raise RuntimeError(f"Series {series_id!r} is absent from the split manifest")
        rows.append(
            {
                CROSS_KEY_COLUMN: series_id,
                "split_role": split_role,
                "mase_scale": robust_mase_scale(values),
                "reference_rows": int(reference.height),
                "reference_end": _date_to_iso(reference.get_column(DATE_COLUMN).max()),
            }
        )
    if not rows:
        raise ValueError("MASE scale manifest cannot be empty")
    return pl.DataFrame(rows).sort(CROSS_KEY_COLUMN)


def _build_train_only_difficulty(
    train_frame: pl.DataFrame,
    *,
    levels: int = 20,
) -> pl.DataFrame:
    """Dificultad separada por target final usando exclusivamente train."""

    rows: list[dict[str, Any]] = []
    scaler = ContextScaler()
    for raw_key, part in train_frame.partition_by(CROSS_KEY_COLUMN, as_dict=True).items():
        key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
        values = part.sort(DATE_COLUMN)[TARGET_COLUMN].to_numpy().astype(np.float64)
        params = scaler.fit(values)
        scaled = scaler.transform(values, params)
        rows.append({
            CROSS_KEY_COLUMN: str(key),
            "entropy_component": _normalized_histogram_entropy(scaled),
            "variance_component": float(np.var(scaled)),
            "change_component": float(np.mean(np.abs(np.diff(scaled)))) if scaled.size > 1 else 0.0,
        })
    if not rows:
        raise ValueError("Cannot calculate curriculum difficulty without train series")
    table = pl.DataFrame(rows).sort(CROSS_KEY_COLUMN)
    entropy_norm = _minmax(table["entropy_component"].to_numpy())
    variance_norm = _minmax(table["variance_component"].to_numpy())
    change_norm = _minmax(table["change_component"].to_numpy())
    difficulty = np.clip(0.50 * entropy_norm + 0.25 * variance_norm + 0.25 * change_norm, 0.0, 1.0)
    table = table.with_columns(pl.Series(DIFFICULTY_COLUMN, difficulty))
    count = table.height
    levels = max(1, min(int(levels), count))
    table = (
        table.sort([DIFFICULTY_COLUMN, CROSS_KEY_COLUMN])
        .with_row_index("_rank")
        .with_columns(
            (((pl.col("_rank") * levels) / count).floor() + 1)
            .clip(1, levels)
            .cast(pl.Int64)
            .alias(CURRICULUM_COLUMN)
        )
        .drop("_rank")
        .sort(CROSS_KEY_COLUMN)
    )
    return table


def _frame_sha256(frame: pl.DataFrame) -> str:
    payload = frame.sort(DATE_COLUMN).write_csv().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_global_inputs(config: GlobalNotebookConfig) -> GlobalInputFrames:
    """Carga y valida el dataset global y el calendario financiero."""

    config.validate()
    global_long = read_polars_uri(config.global_long_uri)
    global_long = normalize_date_column(global_long, DATE_COLUMN)
    global_long = upgrade_global_long_checkpoint19(global_long)
    global_long = global_long.select(GLOBAL_LONG_REQUIRED_COLUMNS).sort(
        [CROSS_KEY_COLUMN, DATE_COLUMN]
    )
    validate_global_long(global_long)

    raw_calendar = read_polars_uri(config.calendar_uri)
    calendar, columns = prepare_calendar_frame(
        raw_calendar,
        exogenous_columns=config.exogenous_columns,
        date_column=config.calendar_date_column,
    )
    return GlobalInputFrames(
        global_long=global_long,
        calendar=calendar,
        exogenous_columns=columns,
        global_long_uri=config.global_long_uri,
        calendar_uri=config.calendar_uri,
    )


def prepare_calendar_frame(
    calendar: pl.DataFrame,
    *,
    exogenous_columns: Sequence[str] = (),
    date_column: str = DATE_COLUMN,
) -> Tuple[pl.DataFrame, Tuple[str, ...]]:
    """Normaliza un calendario y conserva exclusivamente features numéricas."""

    if not isinstance(calendar, pl.DataFrame):
        raise TypeError("calendar must be a polars DataFrame")
    date_column = str(date_column)
    if date_column not in calendar.columns:
        raise ValueError(f"Calendar is missing date column {date_column!r}")
    prepared = calendar.rename({date_column: DATE_COLUMN}) if date_column != DATE_COLUMN else calendar
    prepared = normalize_date_column(prepared, DATE_COLUMN)

    requested = tuple(str(value) for value in exogenous_columns)
    if requested:
        missing = [value for value in requested if value not in prepared.columns]
        if missing:
            raise ValueError(f"Calendar is missing exogenous columns: {missing}")
        columns = requested
    else:
        columns = tuple(
            name
            for name, dtype in prepared.schema.items()
            if name != DATE_COLUMN and _is_numeric_or_boolean(dtype)
        )
    if not columns:
        raise ValueError(
            "No numeric exogenous columns were selected; provide EXOGENOUS_COLUMNS explicitly"
        )
    if DATE_COLUMN in columns:
        raise ValueError("Date column cannot be an exogenous feature")
    if len(set(columns)) != len(columns):
        raise ValueError("Exogenous columns must be unique")

    expressions = [pl.col(DATE_COLUMN)]
    for column in columns:
        dtype = prepared.schema[column]
        if not _is_numeric_or_boolean(dtype):
            raise TypeError(
                f"Exogenous column {column!r} must be numeric or boolean; got {dtype}"
            )
        expressions.append(pl.col(column).cast(pl.Float64).alias(column))
    prepared = prepared.select(expressions).sort(DATE_COLUMN)
    if prepared.get_column(DATE_COLUMN).n_unique() != prepared.height:
        raise ValueError("Calendar contains duplicate dates")
    null_counts = prepared.null_count().row(0, named=True)
    null_columns = [name for name, count in null_counts.items() if int(count) > 0]
    if null_columns:
        raise ValueError(f"Calendar contains null values: {null_columns}")
    for column in columns:
        if not prepared.get_column(column).is_finite().all():
            raise ValueError(f"Calendar column {column!r} contains non-finite values")
    return prepared, columns


def normalize_date_column(frame: pl.DataFrame, date_column: str) -> pl.DataFrame:
    if date_column not in frame.columns:
        raise ValueError(f"Frame is missing date column {date_column!r}")
    dtype = frame.schema[date_column]
    if dtype == pl.Date:
        return frame
    if dtype == pl.Datetime:
        return frame.with_columns(pl.col(date_column).dt.date())
    if dtype == pl.Utf8:
        return frame.with_columns(
            pl.col(date_column).str.to_date(strict=False).alias(date_column)
        ).drop_nulls([date_column])
    return frame.with_columns(pl.col(date_column).cast(pl.Date, strict=False)).drop_nulls(
        [date_column]
    )


def read_polars_uri(uri: str) -> pl.DataFrame:
    """Lee CSV/Parquet desde disco local o S3 sin hardcodear credenciales."""

    normalized = str(uri).strip()
    if not normalized:
        raise ValueError("uri must not be empty")
    suffix = Path(urlparse(normalized).path).suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input suffix={suffix!r}; expected {SUPPORTED_INPUT_SUFFIXES}"
        )
    if normalized.startswith("s3://"):
        payload = _read_s3_bytes(normalized)
        source: Any = BytesIO(payload)
    else:
        source = Path(normalized).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
    if suffix == ".parquet":
        return pl.read_parquet(source)
    return pl.read_csv(source, infer_schema_length=10_000, try_parse_dates=True)


def find_latest_global_long_uri(
    base_uri: str,
    *,
    filename: str = "global_series_long.parquet",
) -> str:
    """Encuentra el run lexicográficamente más reciente bajo un prefijo."""

    base = str(base_uri).strip()
    if not base:
        raise ValueError("base_uri must not be empty")
    if base.startswith("s3://"):
        return _find_latest_s3_object(base, filename=filename)

    root = Path(base).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = sorted(path for path in root.glob(f"*/{filename}") if path.is_file())
    direct = root / filename
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        raise FileNotFoundError(f"No {filename!r} found under {root}")
    return str(sorted(candidates, key=lambda path: (path.parent.name, str(path)))[-1])


def upload_directory_to_s3(local_directory: str | Path, destination_uri: str) -> str:
    """Sube recursivamente un run local a un prefijo S3 opcional."""

    source = Path(local_directory).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    parsed = urlparse(str(destination_uri).strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("destination_uri must be a valid s3:// URI")
    prefix = parsed.path.lstrip("/").rstrip("/")
    import boto3

    client = boto3.client("s3")
    for path in sorted(value for value in source.rglob("*") if value.is_file()):
        relative = path.relative_to(source).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        client.upload_file(str(path), parsed.netloc, key)
    return f"s3://{parsed.netloc}/{prefix}/" if prefix else f"s3://{parsed.netloc}/"


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return destination


def _read_s3_bytes(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    import boto3

    response = boto3.client("s3").get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    return response["Body"].read()


def _find_latest_s3_object(base_uri: str, *, filename: str) -> str:
    parsed = urlparse(base_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {base_uri!r}")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    import boto3

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    candidates: list[str] = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/" + filename) or key == prefix + filename:
                candidates.append(key)
    if not candidates:
        raise FileNotFoundError(f"No {filename!r} found under {base_uri}")
    key = sorted(candidates)[-1]
    return f"s3://{parsed.netloc}/{key}"


def _is_numeric_or_boolean(dtype: pl.DataType) -> bool:
    return bool(dtype.is_numeric() or dtype == pl.Boolean)


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _date_to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
