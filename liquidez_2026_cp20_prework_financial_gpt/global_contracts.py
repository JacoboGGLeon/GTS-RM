"""Contratos canónicos del modelo global Financial-GFM.

Checkpoint 19 separa explícitamente:

- identidad/trazabilidad, que nunca entra al ``forward``;
- covariables temporales del calendario;
- covariables estáticas no identificadoras de la serie;
- target escalado de forma causal y lineal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Tuple


SUPPORTED_ARCHITECTURES: Final[Tuple[str, ...]] = (
    "mlp",
    "mlp_vae",
    "rnn",
    "rnn_bi",
)

DATE_COLUMN: Final[str] = "fecha"
ACCOUNT_CURRENCY_ID_COLUMN: Final[str] = "account_currency_id"
CURRENCY_COLUMN: Final[str] = "divisa"
CROSS_KEY_COLUMN: Final[str] = "cross_key_id"
SERIES_TYPE_COLUMN: Final[str] = "tipo_serie"
SERIES_AGE_COLUMN: Final[str] = "series_age_step"
TARGET_COLUMN: Final[str] = "target"
DIFFICULTY_COLUMN: Final[str] = "difficulty_score"
CURRICULUM_COLUMN: Final[str] = "nivel_curriculum"
GROUP_COLUMN: Final[str] = "grupo"

GLOBAL_LONG_REQUIRED_COLUMNS: Final[Tuple[str, ...]] = (
    DATE_COLUMN,
    ACCOUNT_CURRENCY_ID_COLUMN,
    CURRENCY_COLUMN,
    CROSS_KEY_COLUMN,
    SERIES_TYPE_COLUMN,
    SERIES_AGE_COLUMN,
    TARGET_COLUMN,
    DIFFICULTY_COLUMN,
    CURRICULUM_COLUMN,
    GROUP_COLUMN,
)

# El forward recibe únicamente tensores numéricos. ``x_static`` contiene one-hot
# de tipo/divisa y dos descriptores causales (log_scale y series_age), nunca el id.
MODEL_INPUT_FIELDS: Final[Tuple[str, ...]] = (
    "y_context",
    "x_history",
    "x_future",
    "x_static",
)
MODEL_METADATA_FIELDS: Final[Tuple[str, ...]] = (
    CROSS_KEY_COLUMN,
    ACCOUNT_CURRENCY_ID_COLUMN,
    CURRENCY_COLUMN,
    SERIES_TYPE_COLUMN,
    "cutoff",
    "center",
    "scale",
    "scale_component",
    SERIES_AGE_COLUMN,
)
FORBIDDEN_MODEL_INPUT_FIELDS: Final[Tuple[str, ...]] = (
    CROSS_KEY_COLUMN,
    ACCOUNT_CURRENCY_ID_COLUMN,
    CURRENCY_COLUMN,
    SERIES_TYPE_COLUMN,
    "serie",
)


@dataclass(frozen=True)
class GlobalModelContract:
    """Contrato estable compartido por dataset, modelos y orquestador global."""

    model_inputs: Tuple[str, ...] = MODEL_INPUT_FIELDS
    metadata_fields: Tuple[str, ...] = MODEL_METADATA_FIELDS
    supported_architectures: Tuple[str, ...] = SUPPORTED_ARCHITECTURES

    def validate(self) -> None:
        validate_model_input_fields(self.model_inputs)
        overlap = set(self.model_inputs).intersection(self.metadata_fields)
        if overlap:
            raise ValueError(
                "Model inputs and metadata must be disjoint. "
                f"Overlap: {sorted(overlap)}"
            )


def canonical_cross_key(account_currency_id: object, series_type: str) -> str:
    """Construye ``cuenta + divisa + tipo_serie`` sin usarlo como feature."""

    normalized_type = "_".join(str(series_type).strip().lower().split())
    if not normalized_type:
        raise ValueError("series_type must not be empty")

    normalized_id = str(account_currency_id).strip()
    if not normalized_id:
        raise ValueError("account_currency_id must not be empty")

    return f"{normalized_id}_{normalized_type}"


def validate_model_input_fields(fields: Iterable[str]) -> Tuple[str, ...]:
    """Impide que identificadores o categorías crudas entren a ``forward``."""

    normalized = tuple(str(field).strip() for field in fields)
    forbidden = sorted(set(normalized).intersection(FORBIDDEN_MODEL_INPUT_FIELDS))
    if forbidden:
        raise ValueError(
            "Identifiers/raw categories are metadata and cannot be model inputs: "
            f"{forbidden}"
        )
    return normalized


def validate_global_long_columns(columns: Iterable[str]) -> Tuple[str, ...]:
    """Valida la presencia de las columnas del esquema largo canónico."""

    normalized = tuple(str(column).strip() for column in columns)
    missing = sorted(set(GLOBAL_LONG_REQUIRED_COLUMNS).difference(normalized))
    if missing:
        raise ValueError(f"Missing global long columns: {missing}")
    return normalized


DEFAULT_GLOBAL_CONTRACT = GlobalModelContract()
DEFAULT_GLOBAL_CONTRACT.validate()
