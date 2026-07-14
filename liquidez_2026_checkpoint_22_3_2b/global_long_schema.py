"""Esquema canónico del dataset largo para Financial-GFM.

El esquema conserva targets en escala original, separa la identidad base
``account_currency_id`` de la serie final y añade dos campos requeridos por
Checkpoint 19: ``divisa`` y ``series_age_step``. La construcción de ventanas,
el escalamiento causal y el ajuste train-only viven fuera de este módulo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Final, Mapping, Tuple

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
    TARGET_COLUMN,
    validate_global_long_columns,
)


LEGACY_LONG_REQUIRED_COLUMNS: Final[Tuple[str, ...]] = (
    DATE_COLUMN,
    CROSS_KEY_COLUMN,
    SERIES_TYPE_COLUMN,
    "total_amount",
    DIFFICULTY_COLUMN,
    "curriculum_bucket",
    GROUP_COLUMN,
)

GLOBAL_LONG_KEY_COLUMNS: Final[Tuple[str, ...]] = (
    CROSS_KEY_COLUMN,
    DATE_COLUMN,
)


@dataclass(frozen=True)
class GlobalLongValidationReport:
    """Evidencia compacta de que el dataset canónico es utilizable."""

    row_count: int
    series_count: int
    account_currency_count: int
    min_date: str
    max_date: str
    series_types: Tuple[str, ...]
    min_difficulty: float
    max_difficulty: float
    min_curriculum_level: int
    max_curriculum_level: int

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _require_columns(columns: Tuple[str, ...], required: Tuple[str, ...], label: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def _date_to_iso(value: date | datetime | str) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)




def upgrade_global_long_checkpoint19(global_long: pl.DataFrame) -> pl.DataFrame:
    """Enriquece artefactos canónicos anteriores con divisa y edad causal.

    La compatibilidad es sólo de datos: los pesos anteriores siguen siendo
    incompatibles porque el forward de Checkpoint 19 añade ``x_static``.
    Cuando no puede inferirse una divisa de tres letras desde
    ``account_currency_id``, utiliza ``__UNKNOWN__``.
    """

    if not isinstance(global_long, pl.DataFrame):
        raise TypeError("global_long must be a polars.DataFrame")
    frame = global_long
    if CURRENCY_COLUMN not in frame.columns:
        if ACCOUNT_CURRENCY_ID_COLUMN not in frame.columns:
            raise ValueError(
                f"Cannot infer {CURRENCY_COLUMN!r} without "
                f"{ACCOUNT_CURRENCY_ID_COLUMN!r}"
            )
        frame = frame.with_columns(
            pl.col(ACCOUNT_CURRENCY_ID_COLUMN)
            .cast(pl.String)
            .str.extract(r"([A-Za-z]{3})$", 1)
            .fill_null("__UNKNOWN__")
            .str.to_uppercase()
            .alias(CURRENCY_COLUMN)
        )
    if SERIES_AGE_COLUMN not in frame.columns:
        required = {CROSS_KEY_COLUMN, DATE_COLUMN}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Cannot infer series age; missing columns: {missing}")
        frame = (
            frame.sort([CROSS_KEY_COLUMN, DATE_COLUMN])
            .with_columns(
                pl.col(DATE_COLUMN)
                .cum_count()
                .over(CROSS_KEY_COLUMN)
                .cast(pl.Int64)
                .alias(SERIES_AGE_COLUMN)
            )
        )
    return frame

def _canonicalize_global_long(series_long: pl.DataFrame) -> pl.DataFrame:
    """Convierte ``series_long`` legacy al esquema canónico del modelo global.

    El identificador legacy ``cross_key_id`` se preserva como
    ``account_currency_id``. El nuevo ``cross_key_id`` añade ``tipo_serie``.
    El target siempre se toma de ``total_amount`` en escala original; el
    escalamiento contextual lineal se aplica después al construir cada ventana.
    """

    if not isinstance(series_long, pl.DataFrame):
        raise TypeError("series_long must be a polars.DataFrame")

    source_columns = tuple(series_long.columns)
    _require_columns(source_columns, LEGACY_LONG_REQUIRED_COLUMNS, "legacy long")

    account_currency = pl.col(CROSS_KEY_COLUMN).cast(pl.String).str.strip_chars()
    if CURRENCY_COLUMN in series_long.columns:
        currency_expression = (
            pl.col(CURRENCY_COLUMN).cast(pl.String).str.strip_chars().str.to_uppercase()
        )
    else:
        currency_expression = (
            account_currency.str.extract(r"([A-Za-z]{3})$", 1)
            .fill_null("__UNKNOWN__")
            .str.to_uppercase()
        )
    source_date_dtype = series_long.schema[DATE_COLUMN]
    if source_date_dtype == pl.String:
        date_expression = pl.col(DATE_COLUMN).str.to_date(strict=False)
    else:
        date_expression = pl.col(DATE_COLUMN).cast(pl.Date, strict=False)

    series_type = (
        pl.col(SERIES_TYPE_COLUMN)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", "_")
    )

    global_long = (
        series_long
        .sort([CROSS_KEY_COLUMN, DATE_COLUMN])
        .select(
            date_expression.alias(DATE_COLUMN),
            account_currency.alias(ACCOUNT_CURRENCY_ID_COLUMN),
            currency_expression.alias(CURRENCY_COLUMN),
            pl.concat_str(
                [account_currency, series_type],
                separator="_",
            ).alias(CROSS_KEY_COLUMN),
            series_type.alias(SERIES_TYPE_COLUMN),
            pl.col(DATE_COLUMN).cum_count().over(pl.concat_str([account_currency, series_type], separator="_")).cast(pl.Int64).alias(SERIES_AGE_COLUMN),
            pl.col("total_amount").cast(pl.Float64, strict=False).alias(TARGET_COLUMN),
            pl.col(DIFFICULTY_COLUMN)
            .cast(pl.Float64, strict=False)
            .alias(DIFFICULTY_COLUMN),
            pl.col("curriculum_bucket")
            .cast(pl.Int64, strict=False)
            .alias(CURRICULUM_COLUMN),
            pl.col(GROUP_COLUMN).cast(pl.String).str.strip_chars().alias(GROUP_COLUMN),
        )
        .sort([CROSS_KEY_COLUMN, DATE_COLUMN])
    )

    return global_long


def build_global_long(series_long: pl.DataFrame) -> pl.DataFrame:
    """Construye y valida el dataset canónico."""

    global_long = _canonicalize_global_long(series_long)
    validate_global_long(global_long)
    return global_long


def build_and_validate_global_long(
    series_long: pl.DataFrame,
) -> tuple[pl.DataFrame, GlobalLongValidationReport]:
    """Construye el dataset canónico y devuelve su evidencia de validación."""

    global_long = _canonicalize_global_long(series_long)
    report = validate_global_long(global_long)
    return global_long, report


def validate_global_long(global_long: pl.DataFrame) -> GlobalLongValidationReport:
    """Valida integridad, unicidad y consistencia del dataset canónico."""

    if not isinstance(global_long, pl.DataFrame):
        raise TypeError("global_long must be a polars.DataFrame")

    columns = tuple(global_long.columns)
    validate_global_long_columns(columns)

    if columns != GLOBAL_LONG_REQUIRED_COLUMNS:
        raise ValueError(
            "Global long columns must use the canonical order. "
            f"Expected {GLOBAL_LONG_REQUIRED_COLUMNS}, received {columns}."
        )
    if global_long.is_empty():
        raise ValueError("Global long dataset must not be empty")

    null_counts = {
        column: int(global_long.get_column(column).null_count())
        for column in GLOBAL_LONG_REQUIRED_COLUMNS
    }
    columns_with_nulls = {name: count for name, count in null_counts.items() if count}
    if columns_with_nulls:
        raise ValueError(f"Global long dataset contains nulls: {columns_with_nulls}")

    blank_identifiers = global_long.filter(
        (pl.col(ACCOUNT_CURRENCY_ID_COLUMN).str.len_chars() == 0)
        | (pl.col(CURRENCY_COLUMN).str.len_chars() == 0)
        | (pl.col(CROSS_KEY_COLUMN).str.len_chars() == 0)
        | (pl.col(GROUP_COLUMN).str.len_chars() == 0)
    ).height
    if blank_identifiers:
        raise ValueError(f"Global long dataset contains {blank_identifiers} blank identifiers")

    blank_series_types = global_long.filter(
        pl.col(SERIES_TYPE_COLUMN).cast(pl.String).str.strip_chars().str.len_chars() == 0
    ).height
    if blank_series_types:
        raise ValueError(
            f"Global long dataset contains {blank_series_types} blank series types"
        )
    observed_types = tuple(
        sorted(global_long.get_column(SERIES_TYPE_COLUMN).unique().to_list())
    )

    inconsistent_keys = global_long.filter(
        pl.col(CROSS_KEY_COLUMN)
        != pl.concat_str(
            [pl.col(ACCOUNT_CURRENCY_ID_COLUMN), pl.col(SERIES_TYPE_COLUMN)],
            separator="_",
        )
    ).height
    if inconsistent_keys:
        raise ValueError(
            f"Found {inconsistent_keys} rows whose cross_key_id is not "
            "account_currency_id + tipo_serie"
        )

    duplicate_rows = (
        global_long
        .group_by(list(GLOBAL_LONG_KEY_COLUMNS))
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_rows:
        raise ValueError(
            f"Global long dataset contains {duplicate_rows} duplicated cross_key_id/date keys"
        )

    invalid_age = global_long.filter(pl.col(SERIES_AGE_COLUMN) < 1).height
    if invalid_age:
        raise ValueError(f"Global long dataset contains {invalid_age} invalid series ages")

    age_order_violations = (
        global_long.sort([CROSS_KEY_COLUMN, DATE_COLUMN])
        .with_columns(pl.col(SERIES_AGE_COLUMN).diff().over(CROSS_KEY_COLUMN).alias("_age_diff"))
        .filter(pl.col("_age_diff").is_not_null() & (pl.col("_age_diff") != 1))
        .height
    )
    if age_order_violations:
        raise ValueError(
            f"Global long dataset contains {age_order_violations} non-sequential series ages"
        )

    non_finite_targets = global_long.filter(~pl.col(TARGET_COLUMN).is_finite()).height
    if non_finite_targets:
        raise ValueError(f"Global long dataset contains {non_finite_targets} non-finite targets")

    non_finite_difficulty = global_long.filter(
        ~pl.col(DIFFICULTY_COLUMN).is_finite()
    ).height
    if non_finite_difficulty:
        raise ValueError(
            f"Global long dataset contains {non_finite_difficulty} non-finite difficulty scores"
        )

    out_of_range_difficulty = global_long.filter(
        (pl.col(DIFFICULTY_COLUMN) < 0.0)
        | (pl.col(DIFFICULTY_COLUMN) > 1.0)
    ).height
    if out_of_range_difficulty:
        raise ValueError(
            f"Global long dataset contains {out_of_range_difficulty} difficulty scores outside [0, 1]"
        )

    invalid_curriculum = global_long.filter(pl.col(CURRICULUM_COLUMN) < 1).height
    if invalid_curriculum:
        raise ValueError(
            f"Global long dataset contains {invalid_curriculum} curriculum levels below 1"
        )

    summary = global_long.select(
        pl.len().alias("row_count"),
        pl.col(CROSS_KEY_COLUMN).n_unique().alias("series_count"),
        pl.col(ACCOUNT_CURRENCY_ID_COLUMN).n_unique().alias("account_currency_count"),
        pl.col(DATE_COLUMN).min().alias("min_date"),
        pl.col(DATE_COLUMN).max().alias("max_date"),
        pl.col(DIFFICULTY_COLUMN).min().alias("min_difficulty"),
        pl.col(DIFFICULTY_COLUMN).max().alias("max_difficulty"),
        pl.col(CURRICULUM_COLUMN).min().alias("min_curriculum_level"),
        pl.col(CURRICULUM_COLUMN).max().alias("max_curriculum_level"),
    ).row(0, named=True)

    return GlobalLongValidationReport(
        row_count=int(summary["row_count"]),
        series_count=int(summary["series_count"]),
        account_currency_count=int(summary["account_currency_count"]),
        min_date=_date_to_iso(summary["min_date"]),
        max_date=_date_to_iso(summary["max_date"]),
        series_types=observed_types,
        min_difficulty=float(summary["min_difficulty"]),
        max_difficulty=float(summary["max_difficulty"]),
        min_curriculum_level=int(summary["min_curriculum_level"]),
        max_curriculum_level=int(summary["max_curriculum_level"]),
    )
