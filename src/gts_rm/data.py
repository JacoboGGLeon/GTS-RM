from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._legacy import ensure_cp20_import_path
from .paths import MAC3_TEST_ROOT, REPO_ROOT

ensure_cp20_import_path()

from global_contracts import (  # noqa: E402
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    CURRICULUM_COLUMN,
    DATE_COLUMN,
    DEFAULT_GLOBAL_CONTRACT,
    DIFFICULTY_COLUMN,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    GLOBAL_LONG_REQUIRED_COLUMNS,
    GROUP_COLUMN,
    MODEL_INPUT_FIELDS,
    MODEL_METADATA_FIELDS,
    SERIES_AGE_COLUMN,
    SERIES_TYPE_COLUMN,
    TARGET_COLUMN,
    validate_global_long_columns,
    validate_model_input_fields,
)
from global_data import (  # noqa: E402
    ContextScale,
    ContextScaler,
    GlobalBalancedSampler,
    GlobalSeriesSplit,
    GlobalWindowDataset,
    SeriesBalancedSampler,
    StaticFeatureEncoder,
    robust_mase_scale,
)
from global_long_schema import (  # noqa: E402
    GlobalLongValidationReport,
    build_global_long,
    upgrade_global_long_checkpoint19,
    validate_global_long,
)
from global_notebook import (  # noqa: E402
    ExogenousFeatureScaler,
    GlobalInputFrames,
    GlobalNotebookConfig,
    GlobalNotebookDatasetFactory,
    GlobalPreparedFrames,
    prepare_calendar_frame,
)
from temporal_axis import (  # noqa: E402
    ForecastRequest,
    TemporalAxis,
    TemporalWindowAligner,
)

DATA_CONTRACT_CHECKPOINT = "CP25"


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Data contract must be a JSON object: {resolved}")
    return payload


def _as_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be a list of strings")
    normalized = tuple(str(item).strip() for item in value)
    if any(not item for item in normalized):
        raise ValueError(f"{key} must not contain empty values")
    return normalized


@dataclass(frozen=True)
class MAC3DataContract:
    name: str
    checkpoint: str
    baseline: str
    global_long_uri: str
    calendar_uri: str
    global_long_required_columns: tuple[str, ...]
    model_input_fields: tuple[str, ...]
    metadata_fields: tuple[str, ...]
    forbidden_model_input_fields: tuple[str, ...]
    calendar_date_column: str
    exogenous_columns: tuple[str, ...]
    target_column: str
    split_unit: str
    future_known_exogenous: bool
    temporal_axis_policy: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MAC3DataContract":
        return cls(
            name=str(payload["name"]),
            checkpoint=str(payload["checkpoint"]),
            baseline=str(payload["baseline"]),
            global_long_uri=str(payload["global_long_uri"]),
            calendar_uri=str(payload["calendar_uri"]),
            global_long_required_columns=_as_tuple(payload, "global_long_required_columns"),
            model_input_fields=_as_tuple(payload, "model_input_fields"),
            metadata_fields=_as_tuple(payload, "metadata_fields"),
            forbidden_model_input_fields=_as_tuple(payload, "forbidden_model_input_fields"),
            calendar_date_column=str(payload["calendar_date_column"]),
            exogenous_columns=_as_tuple(payload, "exogenous_columns"),
            target_column=str(payload["target_column"]),
            split_unit=str(payload["split_unit"]),
            future_known_exogenous=bool(payload["future_known_exogenous"]),
            temporal_axis_policy=str(payload["temporal_axis_policy"]),
        )

    def validate(self) -> None:
        if self.checkpoint != DATA_CONTRACT_CHECKPOINT:
            raise ValueError(f"data contract checkpoint must be {DATA_CONTRACT_CHECKPOINT}")
        if self.baseline != "CP20":
            raise ValueError("data contract baseline must be CP20")
        if not self.global_long_uri.strip():
            raise ValueError("global_long_uri must not be empty")
        if not self.calendar_uri.strip():
            raise ValueError("calendar_uri must not be empty")
        if self.global_long_required_columns != GLOBAL_LONG_REQUIRED_COLUMNS:
            raise ValueError("global_long_required_columns must match the CP20 canonical schema")
        if validate_model_input_fields(self.model_input_fields) != MODEL_INPUT_FIELDS:
            raise ValueError("model_input_fields must match the CP20 model input contract")
        if self.metadata_fields != MODEL_METADATA_FIELDS:
            raise ValueError("metadata_fields must match the CP20 metadata contract")
        if self.forbidden_model_input_fields != FORBIDDEN_MODEL_INPUT_FIELDS:
            raise ValueError("forbidden_model_input_fields must match the CP20 forbidden fields")
        validate_global_long_columns(self.global_long_required_columns)
        if self.calendar_date_column != DATE_COLUMN:
            raise ValueError(f"calendar_date_column must be {DATE_COLUMN!r}")
        if self.target_column != TARGET_COLUMN:
            raise ValueError(f"target_column must be {TARGET_COLUMN!r}")
        if self.split_unit != ACCOUNT_CURRENCY_ID_COLUMN:
            raise ValueError(f"split_unit must be {ACCOUNT_CURRENCY_ID_COLUMN!r}")
        if not self.future_known_exogenous:
            raise ValueError("future_known_exogenous must stay true for direct multi-horizon CP20")
        if not self.exogenous_columns:
            raise ValueError("exogenous_columns must not be empty")
        if len(set(self.exogenous_columns)) != len(self.exogenous_columns):
            raise ValueError("exogenous_columns must not contain duplicates")
        reserved = set(self.global_long_required_columns).union(self.model_input_fields).union(self.metadata_fields)
        overlap = sorted(set(self.exogenous_columns).intersection(reserved))
        if overlap:
            raise ValueError(f"exogenous_columns overlap reserved contract fields: {overlap}")
        if "provider temporal axis" not in self.temporal_axis_policy:
            raise ValueError("temporal_axis_policy must preserve the provider temporal axis")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "baseline": self.baseline,
            "global_long_uri": self.global_long_uri,
            "calendar_uri": self.calendar_uri,
            "global_long_required_columns": list(self.global_long_required_columns),
            "model_input_fields": list(self.model_input_fields),
            "metadata_fields": list(self.metadata_fields),
            "forbidden_model_input_fields": list(self.forbidden_model_input_fields),
            "calendar_date_column": self.calendar_date_column,
            "exogenous_columns": list(self.exogenous_columns),
            "target_column": self.target_column,
            "split_unit": self.split_unit,
            "future_known_exogenous": self.future_known_exogenous,
            "temporal_axis_policy": self.temporal_axis_policy,
        }


def load_mac3_data_contract(path: str | Path | None = None) -> MAC3DataContract:
    payload = _load_json(path or MAC3_TEST_ROOT / "configs" / "data_contract.json")
    contract = MAC3DataContract.from_mapping(payload)
    contract.validate()
    return contract


def mac3_data_contract_summary(path: str | Path | None = None) -> dict[str, Any]:
    contract = load_mac3_data_contract(path)
    return {
        "checkpoint": contract.checkpoint,
        "global_long_columns": len(contract.global_long_required_columns),
        "model_input_fields": list(contract.model_input_fields),
        "metadata_fields": list(contract.metadata_fields),
        "exogenous_columns": list(contract.exogenous_columns),
        "split_unit": contract.split_unit,
        "future_known_exogenous": contract.future_known_exogenous,
    }


__all__ = [
    "ACCOUNT_CURRENCY_ID_COLUMN",
    "CROSS_KEY_COLUMN",
    "CURRENCY_COLUMN",
    "CURRICULUM_COLUMN",
    "DATA_CONTRACT_CHECKPOINT",
    "DATE_COLUMN",
    "DEFAULT_GLOBAL_CONTRACT",
    "DIFFICULTY_COLUMN",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "GLOBAL_LONG_REQUIRED_COLUMNS",
    "GROUP_COLUMN",
    "MODEL_INPUT_FIELDS",
    "MODEL_METADATA_FIELDS",
    "SERIES_AGE_COLUMN",
    "SERIES_TYPE_COLUMN",
    "TARGET_COLUMN",
    "ContextScale",
    "ContextScaler",
    "ExogenousFeatureScaler",
    "ForecastRequest",
    "GlobalBalancedSampler",
    "GlobalInputFrames",
    "GlobalLongValidationReport",
    "GlobalNotebookConfig",
    "GlobalNotebookDatasetFactory",
    "GlobalPreparedFrames",
    "GlobalSeriesSplit",
    "GlobalWindowDataset",
    "MAC3DataContract",
    "SeriesBalancedSampler",
    "StaticFeatureEncoder",
    "TemporalAxis",
    "TemporalWindowAligner",
    "build_global_long",
    "load_mac3_data_contract",
    "mac3_data_contract_summary",
    "prepare_calendar_frame",
    "robust_mase_scale",
    "upgrade_global_long_checkpoint19",
    "validate_global_long",
    "validate_global_long_columns",
    "validate_model_input_fields",
]
