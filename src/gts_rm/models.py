from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._legacy import ensure_cp20_import_path

ensure_cp20_import_path()

from global_models import (  # noqa: E402
    GLOBAL_LATENT_FIELD,
    GLOBAL_MODEL_REGISTRY,
    GLOBAL_OUTPUT_FIELD,
    RECONSTRUCTION_FIELD,
    GlobalForecastModel,
    GlobalMLPEncoderDecoder,
    GlobalMLPVAEEncoderDecoder,
    GlobalModelDimensions,
    GlobalRNNBiEncoderDecoder,
    GlobalRNNEncoderDecoder,
    build_global_model,
    get_global_model_class,
    list_global_models,
    validate_global_forward_contract,
)


@dataclass(frozen=True)
class GlobalModelSpec:
    architecture: str
    window_size: int
    horizon: int
    exogenous_dim: int
    static_dim: int
    model_config: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GlobalModelSpec":
        return cls(
            architecture=str(payload["architecture"]),
            window_size=int(payload["window_size"]),
            horizon=int(payload["horizon"]),
            exogenous_dim=int(payload["exogenous_dim"]),
            static_dim=int(payload["static_dim"]),
            model_config=dict(payload["model_config"]),
        )

    def validate(self) -> None:
        if self.architecture not in list_global_models():
            raise ValueError(f"Unsupported architecture={self.architecture!r}")
        GlobalModelDimensions(
            self.window_size,
            self.horizon,
            self.exogenous_dim,
            self.static_dim,
        ).validate()
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")


def build_global_model_from_config(payload: Mapping[str, Any]) -> GlobalForecastModel:
    spec = GlobalModelSpec.from_mapping(payload)
    spec.validate()
    return build_global_model(
        spec.architecture,
        spec.model_config,
        window_size=spec.window_size,
        horizon=spec.horizon,
        exogenous_dim=spec.exogenous_dim,
        static_dim=spec.static_dim,
    )


def build_mac3_smoke_model(architecture: str) -> GlobalForecastModel:
    from . import config

    payload = config.load_smoke_config(architecture)
    return build_global_model_from_config(payload)


def mac3_model_specs() -> dict[str, GlobalModelSpec]:
    from . import config

    return {
        architecture: GlobalModelSpec.from_mapping(config.load_smoke_config(architecture))
        for architecture in list_global_models()
    }


__all__ = [
    "GLOBAL_LATENT_FIELD",
    "GLOBAL_MODEL_REGISTRY",
    "GLOBAL_OUTPUT_FIELD",
    "RECONSTRUCTION_FIELD",
    "GlobalForecastModel",
    "GlobalMLPEncoderDecoder",
    "GlobalMLPVAEEncoderDecoder",
    "GlobalModelDimensions",
    "GlobalModelSpec",
    "GlobalRNNBiEncoderDecoder",
    "GlobalRNNEncoderDecoder",
    "build_global_model",
    "build_global_model_from_config",
    "build_mac3_smoke_model",
    "get_global_model_class",
    "list_global_models",
    "mac3_model_specs",
    "validate_global_forward_contract",
]
