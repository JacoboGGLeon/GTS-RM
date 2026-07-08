"""Dataset global causal para Financial-GFM.

Checkpoint 19 introduce cuatro cambios de representación:

- escalamiento lineal por contexto, sin ``signed_log1p`` ni centrado móvil;
- ``context_mask`` sale del encoder porque el esquema exige targets finitos;
- covariables estáticas no identificadoras: tipo, divisa, log-scale y edad;
- categorías estáticas ajustadas con train y reutilizadas en validación/inferencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Final, Iterator, Mapping, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    CURRICULUM_COLUMN,
    DATE_COLUMN,
    DIFFICULTY_COLUMN,
    GLOBAL_LONG_REQUIRED_COLUMNS,
    GROUP_COLUMN,
    MODEL_INPUT_FIELDS,
    SERIES_AGE_COLUMN,
    SERIES_TYPE_COLUMN,
    TARGET_COLUMN,
    validate_model_input_fields,
)
from global_long_schema import upgrade_global_long_checkpoint19, validate_global_long
from temporal_axis import TemporalAxis, TemporalWindowAligner


MASE_SCALE_COLUMN: Final[str] = "mase_scale"
WINDOW_TARGET_FIELDS: Final[Tuple[str, ...]] = ("y_future", "y_future_raw")
WINDOW_METADATA_FIELDS: Final[Tuple[str, ...]] = (
    CROSS_KEY_COLUMN,
    ACCOUNT_CURRENCY_ID_COLUMN,
    CURRENCY_COLUMN,
    SERIES_TYPE_COLUMN,
    "cutoff",
    "center",
    "scale",
    "transform",
    "scale_component",
    "mean_abs_level",
    "mean_abs_change",
    "log_scale",
    MASE_SCALE_COLUMN,
    SERIES_AGE_COLUMN,
    DIFFICULTY_COLUMN,
    CURRICULUM_COLUMN,
    GROUP_COLUMN,
)
UNKNOWN_CATEGORY: Final[str] = "__UNKNOWN__"


@dataclass(frozen=True)
class ContextScale:
    """Parámetros lineales y reversibles calculados sólo con ``y_context``."""

    center: float
    scale: float
    transform: str = "linear_context_scale"
    scale_component: str = "unknown"
    mean_abs_level: float = 0.0
    mean_abs_change: float = 0.0


class ContextScaler:
    """Escalamiento causal mínimo inspirado en Global Forecasting Models.

    No comprime shocks con logaritmos y no elimina el nivel mediante una mediana
    móvil. Cada ventana se divide por una escala positiva obtenida únicamente de
    su contexto observado:

    ``max(mean(abs(y)), mean(abs(diff(y))), min_scale)``.
    """

    def __init__(self, min_scale: float = 1.0, **legacy_kwargs: float) -> None:
        # Acepta argumentos antiguos para poder cargar notebooks/configs previos,
        # pero ya no participan en el contrato de Checkpoint 19.
        del legacy_kwargs
        if not np.isfinite(min_scale) or float(min_scale) <= 0.0:
            raise ValueError("min_scale must be a positive finite number")
        self.min_scale = float(min_scale)

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        series_type: str | None = None,
    ) -> ContextScale:
        del series_type
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if mask is not None:
            observed = np.asarray(mask, dtype=bool).reshape(-1)
            if observed.shape != array.shape:
                raise ValueError("mask must have the same number of elements as values")
            array = array[observed]
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("ContextScaler requires a non-empty finite context")

        mean_abs_level = float(np.mean(np.abs(array)))
        mean_abs_change = (
            float(np.mean(np.abs(np.diff(array)))) if array.size > 1 else 0.0
        )
        candidates = {
            "mean_abs_level": mean_abs_level,
            "mean_abs_change": mean_abs_change,
            "minimum_floor": self.min_scale,
        }
        component, scale = max(candidates.items(), key=lambda item: item[1])
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("ContextScaler produced an invalid scale")
        return ContextScale(
            center=0.0,
            scale=float(scale),
            transform="linear_context_scale",
            scale_component=str(component),
            mean_abs_level=mean_abs_level,
            mean_abs_change=mean_abs_change,
        )

    def contract(self) -> Mapping[str, float | str]:
        return {
            "method": "causal_linear_context_scale",
            "transform": "y / scale",
            "inverse_transform": "y_scaled * scale",
            "center": "none (0.0)",
            "scale": "max(mean(abs(y_context)), mean(abs(diff(y_context))), min_scale)",
            "min_scale": self.min_scale,
            "fit_scope": "y_context_only",
            "series_type_dependency": "none",
            "nonlinear_target_transform": "none",
        }

    @staticmethod
    def transform(values: np.ndarray, parameters: ContextScale) -> np.ndarray:
        if parameters.transform not in {"linear_context_scale", "identity"}:
            raise ValueError(f"Unsupported context transform={parameters.transform!r}")
        values_array = np.asarray(values, dtype=np.float64)
        return (values_array - float(parameters.center)) / float(parameters.scale)

    @staticmethod
    def inverse_transform_with_diagnostics(
        values: np.ndarray,
        parameters: ContextScale,
    ) -> tuple[np.ndarray, Mapping[str, int]]:
        if parameters.transform not in {"linear_context_scale", "identity"}:
            raise ValueError(f"Unsupported context transform={parameters.transform!r}")
        values_array = np.asarray(values, dtype=np.float64)
        nonfinite_mask = ~np.isfinite(values_array)
        safe = np.nan_to_num(values_array, nan=0.0, posinf=0.0, neginf=0.0)
        scale = float(parameters.scale)
        max_raw = np.sqrt(np.finfo(np.float64).max) / 4.0
        max_normalized = max_raw / max(abs(scale), np.finfo(np.float64).tiny)
        clipped_mask = np.abs(safe) > max_normalized
        safe = np.clip(safe, -max_normalized, max_normalized)
        raw = safe * scale + float(parameters.center)
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("ContextScaler inverse transform produced non-finite values")
        return raw, {
            "nonfinite_inputs": int(nonfinite_mask.sum()),
            "clipped_values": int(clipped_mask.sum()),
        }

    @staticmethod
    def inverse_transform(values: np.ndarray, parameters: ContextScale) -> np.ndarray:
        raw, _ = ContextScaler.inverse_transform_with_diagnostics(values, parameters)
        return raw


def robust_mase_scale(
    values: np.ndarray,
    *,
    relative_floor: float = 0.01,
    min_scale: float = 1e-6,
) -> float:
    """Return a causal, finite denominator for MASE.

    The primary term is the in-sample one-step naive MAE. Constant or nearly
    constant histories receive a small floor tied to their absolute level so
    that the metric remains finite without allowing a tiny denominator to
    dominate global selection. Only the supplied history is inspected.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("robust_mase_scale requires a non-empty finite history")
    if not np.isfinite(relative_floor) or float(relative_floor) < 0.0:
        raise ValueError("relative_floor must be finite and non-negative")
    if not np.isfinite(min_scale) or float(min_scale) <= 0.0:
        raise ValueError("min_scale must be finite and positive")

    naive_mae = (
        float(np.mean(np.abs(np.diff(array)))) if array.size > 1 else 0.0
    )
    mean_abs_level = float(np.mean(np.abs(array)))
    level_floor = float(relative_floor) * max(mean_abs_level, 1.0)
    scale = max(naive_mae, level_floor, float(min_scale))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("robust_mase_scale produced an invalid denominator")
    return float(scale)


@dataclass(frozen=True)
class StaticFeatureEncoder:
    """One-hot train-only para tipo/divisa más dos descriptores causales."""

    series_type_categories: Tuple[str, ...]
    currency_categories: Tuple[str, ...]
    unknown_token: str = UNKNOWN_CATEGORY

    @classmethod
    def fit(cls, train_frame: pl.DataFrame) -> "StaticFeatureEncoder":
        if not isinstance(train_frame, pl.DataFrame) or train_frame.is_empty():
            raise ValueError("StaticFeatureEncoder requires a non-empty train frame")
        required = {SERIES_TYPE_COLUMN, CURRENCY_COLUMN}
        missing = sorted(required.difference(train_frame.columns))
        if missing:
            raise ValueError(f"Static feature frame is missing columns: {missing}")
        observed_types = {
            str(value).strip().lower()
            for value in train_frame.get_column(SERIES_TYPE_COLUMN).unique().to_list()
            if str(value).strip()
        }
        # Saldo y variación forman el contrato actual del producto. Se incluyen
        # siempre para que datasets construidos directamente (fuera de la factory)
        # conserven dimensiones estables aunque una partición contenga sólo un tipo.
        types = tuple(sorted(observed_types.union({"saldo", "variacion"})))
        currencies = tuple(
            sorted(
                str(value).strip().upper()
                for value in train_frame.get_column(CURRENCY_COLUMN).unique().to_list()
                if str(value).strip()
            )
        )
        if not types or not currencies:
            raise ValueError("Static feature categories must not be empty")
        return cls(types, currencies)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        type_names = tuple(f"tipo_serie={value}" for value in self.series_type_categories)
        currency_names = tuple(f"divisa={value}" for value in self.currency_categories)
        return (
            *type_names,
            f"tipo_serie={self.unknown_token}",
            *currency_names,
            f"divisa={self.unknown_token}",
            "log_scale_bounded",
            "series_age_bounded",
        )

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def _bounded_log(value: float) -> float:
        value = max(0.0, float(value))
        logged = float(np.log1p(value))
        return logged / (1.0 + logged)

    def encode(
        self,
        *,
        series_type: str,
        currency: str,
        scale: float,
        series_age: int | float,
    ) -> np.ndarray:
        normalized_type = str(series_type).strip().lower()
        normalized_currency = str(currency).strip().upper()
        result = np.zeros(self.dimension, dtype=np.float32)

        type_index = (
            self.series_type_categories.index(normalized_type)
            if normalized_type in self.series_type_categories
            else len(self.series_type_categories)
        )
        result[type_index] = 1.0

        currency_offset = len(self.series_type_categories) + 1
        currency_index = (
            self.currency_categories.index(normalized_currency)
            if normalized_currency in self.currency_categories
            else len(self.currency_categories)
        )
        result[currency_offset + currency_index] = 1.0
        result[-2] = self._bounded_log(float(scale))
        result[-1] = self._bounded_log(float(series_age))
        return result

    def to_dict(self) -> Mapping[str, object]:
        return {
            "series_type_categories": list(self.series_type_categories),
            "currency_categories": list(self.currency_categories),
            "unknown_token": self.unknown_token,
            "feature_names": list(self.feature_names),
            "dimension": self.dimension,
            "fit_scope": "training_series_only",
            "continuous_features": {
                "log_scale_bounded": "log1p(scale)/(1+log1p(scale))",
                "series_age_bounded": "log1p(age)/(1+log1p(age))",
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StaticFeatureEncoder":
        return cls(
            series_type_categories=tuple(str(v) for v in payload["series_type_categories"]),
            currency_categories=tuple(str(v) for v in payload["currency_categories"]),
            unknown_token=str(payload.get("unknown_token", UNKNOWN_CATEGORY)),
        )


@dataclass(frozen=True)
class GlobalSeriesSplit:
    """Partición reproducible de identidades para evaluación seen/unseen."""

    train_series: Tuple[str, ...]
    validation_seen_series: Tuple[str, ...]
    validation_unseen_series: Tuple[str, ...]
    test_unseen_series: Tuple[str, ...]
    seed: int = 42

    @classmethod
    def create(
        cls,
        global_long: pl.DataFrame,
        *,
        validation_unseen_fraction: float = 0.15,
        test_unseen_fraction: float = 0.15,
        seed: int = 42,
    ) -> "GlobalSeriesSplit":
        global_long = upgrade_global_long_checkpoint19(global_long)
        validate_global_long(global_long.select(GLOBAL_LONG_REQUIRED_COLUMNS))
        _validate_holdout_fraction(validation_unseen_fraction, "validation_unseen_fraction")
        _validate_holdout_fraction(test_unseen_fraction, "test_unseen_fraction")
        if validation_unseen_fraction + test_unseen_fraction >= 1.0:
            raise ValueError("validation and test unseen fractions must sum to less than 1")
        # El split se hace por cuenta-divisa base, no por target derivado. Así
        # saldo y variación de la misma entidad nunca quedan en particiones
        # distintas, evitando leakage estructural entre series matemáticamente ligadas.
        groups = sorted(
            str(v) for v in global_long[ACCOUNT_CURRENCY_ID_COLUMN].unique().to_list()
        )
        if len(groups) < 3:
            raise ValueError("At least three account_currency_id groups are required for splitting")
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(groups, dtype=object)
        rng.shuffle(shuffled)
        validation_count = _holdout_count(len(groups), validation_unseen_fraction)
        test_count = _holdout_count(len(groups), test_unseen_fraction)
        if validation_count + test_count >= len(groups):
            raise ValueError("Unseen holdouts leave no training account-currency groups")
        validation_groups = set(str(x) for x in shuffled[:validation_count])
        test_groups = set(str(x) for x in shuffled[validation_count:validation_count + test_count])
        train_groups = set(str(x) for x in shuffled[validation_count + test_count:])

        mapping = global_long.select(
            ACCOUNT_CURRENCY_ID_COLUMN, CROSS_KEY_COLUMN
        ).unique()
        def expand(selected_groups: set[str]) -> Tuple[str, ...]:
            return tuple(sorted(
                str(v)
                for v in mapping.filter(
                    pl.col(ACCOUNT_CURRENCY_ID_COLUMN).is_in(selected_groups)
                )[CROSS_KEY_COLUMN].to_list()
            ))
        validation_ids = expand(validation_groups)
        test_ids = expand(test_groups)
        train_ids = expand(train_groups)
        split = cls(train_ids, train_ids, validation_ids, test_ids, int(seed))
        split.validate()
        return split

    def validate(self) -> None:
        train = set(self.train_series)
        validation_unseen = set(self.validation_unseen_series)
        test_unseen = set(self.test_unseen_series)
        if not train or not validation_unseen or not test_unseen:
            raise ValueError("train and unseen partitions must not be empty")
        if set(self.validation_seen_series) != train:
            raise ValueError("validation_seen_series must contain exactly train identities")
        if train & validation_unseen or train & test_unseen or validation_unseen & test_unseen:
            raise ValueError("train, validation_unseen and test_unseen must be disjoint")

    def series_for(self, partition: str) -> Tuple[str, ...]:
        lookup = {
            "train": self.train_series,
            "validation_seen": self.validation_seen_series,
            "validation_unseen": self.validation_unseen_series,
            "test_unseen": self.test_unseen_series,
        }
        try:
            return lookup[partition]
        except KeyError as exc:
            raise ValueError(f"Unsupported partition={partition!r}; expected {tuple(lookup)}") from exc

    def filter_frame(self, global_long: pl.DataFrame, partition: str) -> pl.DataFrame:
        return global_long.filter(pl.col(CROSS_KEY_COLUMN).is_in(self.series_for(partition)))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "seed": self.seed,
            "split_unit": ACCOUNT_CURRENCY_ID_COLUMN,
            "paired_series_policy": "all series types from one account-currency stay together",
            "train_series": list(self.train_series),
            "validation_seen_series": list(self.validation_seen_series),
            "validation_unseen_series": list(self.validation_unseen_series),
            "test_unseen_series": list(self.test_unseen_series),
        }


@dataclass(frozen=True)
class _WindowReference:
    cross_key_id: str
    start: int


@dataclass(frozen=True)
class _SeriesArrays:
    dates: np.ndarray
    ages: np.ndarray
    target: np.ndarray
    exogenous: np.ndarray
    account_currency_id: str
    currency: str
    series_type: str
    mase_scale: float
    difficulty_score: float
    curriculum_level: int
    group: str


class GlobalWindowDataset(Dataset[Mapping[str, object]]):
    """Ventanas causales con categorías estáticas fuera de la identidad contable."""

    def __init__(
        self,
        global_long: pl.DataFrame,
        *,
        window_size: int,
        horizon: int,
        exogenous: pl.DataFrame | None = None,
        exogenous_columns: Sequence[str] = (),
        series_ids: Sequence[str] | None = None,
        stride: int = 1,
        scaler: ContextScaler | None = None,
        static_feature_encoder: StaticFeatureEncoder | None = None,
        mase_scale_by_series: Mapping[str, float] | None = None,
        tensor_dtype: torch.dtype = torch.float32,
    ) -> None:
        global_long = upgrade_global_long_checkpoint19(global_long)
        canonical_view = global_long.select(GLOBAL_LONG_REQUIRED_COLUMNS)
        validate_global_long(canonical_view)
        _require_positive_integer(window_size, "window_size")
        _require_positive_integer(horizon, "horizon")
        _require_positive_integer(stride, "stride")
        validate_model_input_fields(MODEL_INPUT_FIELDS)

        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.exogenous_columns = tuple(str(column) for column in exogenous_columns)
        self.scaler = scaler or ContextScaler()
        self.static_feature_encoder = static_feature_encoder or StaticFeatureEncoder.fit(global_long)
        self.mase_scale_by_series = {
            str(key): float(value) for key, value in (mase_scale_by_series or {}).items()
        }
        invalid_mase = {
            key: value
            for key, value in self.mase_scale_by_series.items()
            if not np.isfinite(value) or value <= 0.0
        }
        if invalid_mase:
            raise ValueError(f"Invalid MASE scales: {invalid_mase}")
        self.tensor_dtype = tensor_dtype

        selected = global_long
        if series_ids is not None:
            normalized_ids = tuple(str(value) for value in series_ids)
            if not normalized_ids:
                raise ValueError("series_ids must not be empty when provided")
            selected = selected.filter(pl.col(CROSS_KEY_COLUMN).is_in(normalized_ids))
            missing_ids = sorted(set(normalized_ids).difference(str(x) for x in selected[CROSS_KEY_COLUMN].unique().to_list()))
            if missing_ids:
                raise ValueError(f"Unknown series_ids: {missing_ids}")
        if selected.is_empty():
            raise ValueError("GlobalWindowDataset received no rows after filtering")

        prepared, self.alignment_report = _attach_exogenous(
            selected,
            exogenous=exogenous,
            exogenous_columns=self.exogenous_columns,
        )
        self._series: Dict[str, _SeriesArrays] = {}
        self._references: list[_WindowReference] = []
        self._indices_by_series: Dict[str, list[int]] = {}

        partitions = prepared.partition_by(CROSS_KEY_COLUMN, as_dict=True, maintain_order=False)
        for raw_key, frame in partitions.items():
            key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
            cross_key_id = str(key)
            ordered = frame.sort(DATE_COLUMN)
            row_count = ordered.height
            usable = row_count - self.window_size - self.horizon + 1
            if usable <= 0:
                continue
            metadata_columns = (
                ACCOUNT_CURRENCY_ID_COLUMN,
                CURRENCY_COLUMN,
                SERIES_TYPE_COLUMN,
                DIFFICULTY_COLUMN,
                CURRICULUM_COLUMN,
                GROUP_COLUMN,
            )
            cardinality = ordered.select([pl.col(c).n_unique().alias(c) for c in metadata_columns]).row(0, named=True)
            inconsistent = [c for c, count in cardinality.items() if int(count) != 1]
            if inconsistent:
                raise ValueError(f"Series {cross_key_id!r} has time-varying metadata: {inconsistent}")
            target = ordered[TARGET_COLUMN].to_numpy().astype(np.float64, copy=False)
            if not np.all(np.isfinite(target)):
                raise ValueError(f"Series {cross_key_id!r} contains non-finite targets")
            dates = ordered[DATE_COLUMN].to_numpy()
            ages = ordered[SERIES_AGE_COLUMN].to_numpy().astype(np.int64, copy=False)
            exogenous_values = (
                ordered.select(self.exogenous_columns).to_numpy().astype(np.float64, copy=False)
                if self.exogenous_columns
                else np.empty((row_count, 0), dtype=np.float64)
            )
            mase_scale = self.mase_scale_by_series.get(cross_key_id)
            if mase_scale is None:
                # Safe fallback for direct dataset construction outside the
                # notebook factory: only the earliest available context is used.
                mase_scale = robust_mase_scale(target[: self.window_size])
            first = ordered.row(0, named=True)
            self._series[cross_key_id] = _SeriesArrays(
                dates=dates,
                ages=ages,
                target=target,
                exogenous=exogenous_values,
                account_currency_id=str(first[ACCOUNT_CURRENCY_ID_COLUMN]),
                currency=str(first[CURRENCY_COLUMN]),
                series_type=str(first[SERIES_TYPE_COLUMN]),
                mase_scale=float(mase_scale),
                difficulty_score=float(first[DIFFICULTY_COLUMN]),
                curriculum_level=int(first[CURRICULUM_COLUMN]),
                group=str(first[GROUP_COLUMN]),
            )
            self._indices_by_series[cross_key_id] = []
            for start in range(0, usable, self.stride):
                dataset_index = len(self._references)
                self._references.append(_WindowReference(cross_key_id, start))
                self._indices_by_series[cross_key_id].append(dataset_index)
        if not self._references:
            raise ValueError("No valid windows were generated; each series needs window_size + horizon rows")

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        reference = self._references[index]
        series = self._series[reference.cross_key_id]
        context_end = reference.start + self.window_size
        future_end = context_end + self.horizon
        y_context_raw = np.array(series.target[reference.start:context_end].reshape(-1, 1), dtype=np.float64, copy=True)
        y_future_raw = np.array(series.target[context_end:future_end].reshape(-1, 1), dtype=np.float64, copy=True)
        parameters = self.scaler.fit(y_context_raw, series_type=series.series_type)
        y_context = self.scaler.transform(y_context_raw, parameters).astype(np.float32)
        y_future = self.scaler.transform(y_future_raw, parameters).astype(np.float32)
        x_history = np.array(series.exogenous[reference.start:context_end], dtype=np.float32, copy=True)
        x_future = np.array(series.exogenous[context_end:future_end], dtype=np.float32, copy=True)
        series_age = int(series.ages[context_end - 1])
        x_static = self.static_feature_encoder.encode(
            series_type=series.series_type,
            currency=series.currency,
            scale=parameters.scale,
            series_age=series_age,
        )
        cutoff = _date_to_iso(series.dates[context_end - 1])
        model_inputs: Mapping[str, torch.Tensor] = {
            "y_context": torch.as_tensor(y_context, dtype=self.tensor_dtype),
            "x_history": torch.as_tensor(x_history, dtype=self.tensor_dtype),
            "x_future": torch.as_tensor(x_future, dtype=self.tensor_dtype),
            "x_static": torch.as_tensor(x_static, dtype=self.tensor_dtype),
        }
        if tuple(model_inputs) != MODEL_INPUT_FIELDS:
            raise RuntimeError("Global window model_inputs violate canonical field order")
        return {
            "model_inputs": model_inputs,
            "targets": {
                "y_future": torch.as_tensor(y_future, dtype=self.tensor_dtype),
                "y_future_raw": torch.as_tensor(y_future_raw, dtype=self.tensor_dtype),
            },
            "metadata": {
                CROSS_KEY_COLUMN: reference.cross_key_id,
                ACCOUNT_CURRENCY_ID_COLUMN: series.account_currency_id,
                CURRENCY_COLUMN: series.currency,
                SERIES_TYPE_COLUMN: series.series_type,
                "cutoff": cutoff,
                "center": parameters.center,
                "scale": parameters.scale,
                "transform": parameters.transform,
                "scale_component": parameters.scale_component,
                "mean_abs_level": parameters.mean_abs_level,
                "mean_abs_change": parameters.mean_abs_change,
                "log_scale": float(np.log1p(parameters.scale)),
                MASE_SCALE_COLUMN: series.mase_scale,
                SERIES_AGE_COLUMN: series_age,
                DIFFICULTY_COLUMN: series.difficulty_score,
                CURRICULUM_COLUMN: series.curriculum_level,
                GROUP_COLUMN: series.group,
            },
        }

    def context_dates(self, index: int) -> Tuple[str, ...]:
        reference = self._references[index]
        series = self._series[reference.cross_key_id]
        end = reference.start + self.window_size
        return tuple(_date_to_iso(v) for v in series.dates[reference.start:end])

    def future_dates(self, index: int) -> Tuple[str, ...]:
        reference = self._references[index]
        series = self._series[reference.cross_key_id]
        start = reference.start + self.window_size
        end = start + self.horizon
        return tuple(_date_to_iso(v) for v in series.dates[start:end])

    @property
    def series_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._indices_by_series))

    @property
    def indices_by_series(self) -> Mapping[str, Tuple[int, ...]]:
        return {series_id: tuple(indices) for series_id, indices in self._indices_by_series.items()}

    @property
    def static_dim(self) -> int:
        return self.static_feature_encoder.dimension

    @property
    def static_feature_names(self) -> Tuple[str, ...]:
        return self.static_feature_encoder.feature_names

    @property
    def series_curriculum_levels(self) -> Mapping[str, int]:
        return {series_id: int(self._series[series_id].curriculum_level) for series_id in self.series_ids}

    @property
    def series_difficulty_scores(self) -> Mapping[str, float]:
        return {series_id: float(self._series[series_id].difficulty_score) for series_id in self.series_ids}

    @property
    def series_types(self) -> Mapping[str, str]:
        return {
            series_id: str(self._series[series_id].series_type).strip().lower()
            for series_id in self.series_ids
        }

    @property
    def series_mase_scales(self) -> Mapping[str, float]:
        return {
            series_id: float(self._series[series_id].mase_scale)
            for series_id in self.series_ids
        }


class GlobalBalancedSampler(Sampler[int]):
    """Balance samples by series type, curriculum level, series and window.

    When both ``saldo`` and ``variacion`` are present, each type receives the
    same expected share. Within a type, available curriculum levels are sampled
    uniformly and then a series/window is selected uniformly. This prevents a
    populous type, an easy level, or a long series from dominating the epoch.
    """

    def __init__(
        self,
        dataset: GlobalWindowDataset,
        *,
        num_samples: int | None = None,
        seed: int = 42,
        series_ids: Sequence[str] | None = None,
        balance_series_type: bool = True,
        balance_curriculum: bool = True,
    ) -> None:
        if not isinstance(dataset, GlobalWindowDataset):
            raise TypeError("dataset must be a GlobalWindowDataset")
        requested = len(dataset) if num_samples is None else num_samples
        _require_positive_integer(requested, "num_samples")
        selected = tuple(str(v) for v in (series_ids or dataset.series_ids))
        unknown = sorted(set(selected).difference(dataset.series_ids))
        if unknown:
            raise ValueError(f"Unknown series_ids for sampler: {unknown}")
        if not selected:
            raise ValueError("series_ids must not be empty")

        self.dataset = dataset
        self.num_samples = int(requested)
        self.seed = int(seed)
        self.epoch = 0
        self.balance_series_type = bool(balance_series_type)
        self.balance_curriculum = bool(balance_curriculum)
        self._indices_by_series = dataset.indices_by_series
        type_by_series = dataset.series_types
        level_by_series = dataset.series_curriculum_levels

        strata: Dict[tuple[str, int], list[str]] = {}
        for series_id in selected:
            key = (type_by_series[series_id], int(level_by_series[series_id]))
            strata.setdefault(key, []).append(series_id)
        self._strata = {key: tuple(sorted(values)) for key, values in strata.items()}
        self._types = tuple(sorted({key[0] for key in self._strata}))
        self._levels_by_type = {
            series_type: tuple(sorted(key[1] for key in self._strata if key[0] == series_type))
            for series_type in self._types
        }
        self._all_series = tuple(sorted(selected))
        self.last_draw_counts: Mapping[str, int] = {}

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        counts: Dict[str, int] = {}
        for _ in range(self.num_samples):
            if self.balance_series_type:
                series_type = self._types[int(rng.integers(0, len(self._types)))]
                levels = self._levels_by_type[series_type]
                if self.balance_curriculum:
                    level = levels[int(rng.integers(0, len(levels)))]
                    pool = self._strata[(series_type, level)]
                else:
                    pool = tuple(
                        sid
                        for lvl in levels
                        for sid in self._strata[(series_type, lvl)]
                    )
            elif self.balance_curriculum:
                levels = tuple(sorted({key[1] for key in self._strata}))
                level = levels[int(rng.integers(0, len(levels)))]
                pool = tuple(
                    sid for (stype, lvl), values in self._strata.items()
                    if lvl == level for sid in values
                )
                series_type = self.dataset.series_types[pool[0]]
            else:
                pool = self._all_series
                series_type = self.dataset.series_types[pool[0]]

            series_id = pool[int(rng.integers(0, len(pool)))]
            level = int(self.dataset.series_curriculum_levels[series_id])
            series_type = self.dataset.series_types[series_id]
            candidates = self._indices_by_series[series_id]
            counts[f"type:{series_type}"] = counts.get(f"type:{series_type}", 0) + 1
            counts[f"level:{level}"] = counts.get(f"level:{level}", 0) + 1
            yield int(candidates[int(rng.integers(0, len(candidates)))])
        self.last_draw_counts = dict(sorted(counts.items()))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch


class SeriesBalancedSampler(Sampler[int]):
    """Muestrea primero la serie y después una ventana de esa serie."""

    def __init__(self, dataset: GlobalWindowDataset, *, num_samples: int | None = None, seed: int = 42) -> None:
        if not isinstance(dataset, GlobalWindowDataset):
            raise TypeError("dataset must be a GlobalWindowDataset")
        requested = len(dataset) if num_samples is None else num_samples
        _require_positive_integer(requested, "num_samples")
        self.dataset = dataset
        self.num_samples = int(requested)
        self.seed = int(seed)
        self.epoch = 0
        self._series_ids = dataset.series_ids
        self._indices_by_series = dataset.indices_by_series

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_samples):
            series_id = self._series_ids[int(rng.integers(0, len(self._series_ids)))]
            candidates = self._indices_by_series[series_id]
            yield int(candidates[int(rng.integers(0, len(candidates)))])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch


def _attach_exogenous(
    global_long: pl.DataFrame,
    *,
    exogenous: pl.DataFrame | None,
    exogenous_columns: Tuple[str, ...],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not exogenous_columns:
        reports = (
            global_long.group_by(CROSS_KEY_COLUMN)
            .agg(
                pl.len().alias("original_rows"),
                pl.len().alias("aligned_rows"),
                pl.lit(0).cast(pl.UInt32).alias("excluded_rows"),
                pl.lit(1.0).alias("coverage_ratio"),
                pl.min(DATE_COLUMN).cast(pl.String).alias("first_original_timestamp"),
                pl.max(DATE_COLUMN).cast(pl.String).alias("last_original_timestamp"),
                pl.min(DATE_COLUMN).cast(pl.String).alias("first_aligned_timestamp"),
                pl.max(DATE_COLUMN).cast(pl.String).alias("last_aligned_timestamp"),
            ).sort(CROSS_KEY_COLUMN)
        )
        return global_long, reports
    if exogenous is None:
        missing = sorted(set(exogenous_columns).difference(global_long.columns))
        if missing:
            raise ValueError(f"Missing exogenous columns and no calendar supplied: {missing}")
        prepared = global_long
        reports = _attach_exogenous(global_long, exogenous=None, exogenous_columns=())[1]
    else:
        if not isinstance(exogenous, pl.DataFrame):
            raise TypeError("exogenous must be a polars.DataFrame")
        axis = TemporalAxis.from_frame(exogenous, timestamp_column=DATE_COLUMN, feature_columns=exogenous_columns)
        prepared, reports = TemporalWindowAligner(axis).align_global_long(global_long)
    if prepared.is_empty():
        raise ValueError("Temporal alignment removed every target observation")
    null_counts = prepared.select([pl.col(c).null_count().alias(c) for c in exogenous_columns]).row(0, named=True)
    missing_values = {c: int(n) for c, n in null_counts.items() if n}
    if missing_values:
        raise ValueError(f"Temporal alignment produced missing exogenous values: {missing_values}")
    for column in exogenous_columns:
        casted = prepared[column].cast(pl.Float64, strict=False)
        if casted.null_count() or int((~casted.is_finite()).sum()) > 0:
            raise ValueError(f"Exogenous feature {column!r} must be finite and numeric")
        prepared = prepared.with_columns(casted.alias(column))
    return prepared, reports


def _holdout_count(total: int, fraction: float) -> int:
    return max(1, int(round(total * fraction)))


def _validate_holdout_fraction(value: float, label: str) -> None:
    if not np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError(f"{label} must be in the open interval (0, 1)")


def _require_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _date_to_iso(value: object) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
