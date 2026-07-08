from __future__ import annotations

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

__all__ = [
    "GLOBAL_LATENT_FIELD",
    "GLOBAL_MODEL_REGISTRY",
    "GLOBAL_OUTPUT_FIELD",
    "RECONSTRUCTION_FIELD",
    "GlobalForecastModel",
    "GlobalMLPEncoderDecoder",
    "GlobalMLPVAEEncoderDecoder",
    "GlobalModelDimensions",
    "GlobalRNNBiEncoderDecoder",
    "GlobalRNNEncoderDecoder",
    "build_global_model",
    "get_global_model_class",
    "list_global_models",
    "validate_global_forward_contract",
]
