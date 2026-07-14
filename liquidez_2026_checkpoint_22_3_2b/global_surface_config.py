"""Superficie activa y tipada de los notebooks globales GTRM.

Checkpoint 22.3.2b separa explícitamente:

- capacidades activas del modelo;
- defaults de arquitectura usados sólo como fallback/ablation;
- espacio HPO realmente explorado por Optuna;
- decoder residual y heads auxiliares;
- presupuesto de HPO/entrenamiento pooled;
- configuración de inferencia y visualización.

Estos modelos Pydantic viven en la frontera del notebook. Los tensores, batches y
objetos internos de PyTorch permanecen fuera de Pydantic para no introducir
sobrecoste en el camino crítico de entrenamiento.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

ActivationName = Literal["relu", "gelu", "silu", "tanh"]
MagnitudeTransform = Literal["asinh", "log1p", "abs"]
LossName = Literal["rmse", "mae", "mse", "smape", "wmape", "log_cosh", "huber"]
SelectionMetric = Literal[
    "robust_macro_mase",
    "macro_mae",
    "macro_rmse",
    "micro_mae",
    "raw_macro_mae",
    "raw_macro_rmse",
    "raw_macro_wmape",
    "raw_macro_smape",
]


class _StrictConfig(BaseModel):
    """Base inmutable y estricta para configuración serializable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ModelFeatureConfig(_StrictConfig):
    """Capacidades que existen y participan realmente en el run actual."""

    use_static_context: bool = True
    use_modality_specific_encoders: bool = True
    use_auxiliary_autoencoder: bool = True


class TemporalForecastConfig(_StrictConfig):
    """Contrato temporal visible del entrenamiento y del forecast.

    ``forecast_horizon`` es el número total máximo de timestamps que se desea
    devolver cuando la inferencia se solicita por número de pasos.

    ``rollout_chunk_size`` es la longitud del target de cada ventana de
    entrenamiento y, por tanto, la cantidad de pasos que el modelo produce en
    una sola llamada ``forward``. Si el horizonte total es mayor, la inferencia
    repite bloques y agrega la media predicha al contexto antes del siguiente
    bloque.

    ``training_stride`` controla cuánto avanza el origen de las sliding windows
    al construir muestras. No controla el avance del forecast.
    """

    forecast_horizon: PositiveInt = 25
    rollout_chunk_size: PositiveInt = 3
    training_stride: PositiveInt = 1

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "TemporalForecastConfig":
        if self.rollout_chunk_size > self.forecast_horizon:
            raise ValueError(
                "rollout_chunk_size cannot exceed forecast_horizon"
            )
        return self

    @property
    def rollout_blocks(self) -> int:
        """Número máximo de bloques necesarios para cubrir el horizonte total."""

        return int(math.ceil(self.forecast_horizon / self.rollout_chunk_size))


class IntegerRange(_StrictConfig):
    """Rango entero inclusivo usado por Optuna."""

    minimum: PositiveInt
    maximum: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> "IntegerRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class FloatRange(_StrictConfig):
    """Rango flotante inclusivo usado por Optuna."""

    minimum: float = Field(ge=0.0)
    maximum: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> "FloatRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.maximum >= 1.0:
            raise ValueError("dropout maximum must be lower than 1.0")
        return self


class ModalityEncoderDefaults(_StrictConfig):
    """Fallbacks usados si el HPO arquitectónico se desactiva.

    Cuando ``ModalityEncoderHPOSpace.enabled=True``, estos valores inicializan el
    contrato base, pero el candidato productivo usa los valores seleccionados por
    Optuna.
    """

    target_dim: PositiveInt = 32
    historical_dim: PositiveInt = 32
    future_dim: PositiveInt = 32
    static_dim: PositiveInt = 16
    fusion_hidden_size: PositiveInt = 64
    target_num_layers: PositiveInt = 1
    historical_num_layers: PositiveInt = 1
    future_num_layers: PositiveInt = 1
    static_num_layers: PositiveInt = 1
    fusion_num_layers: PositiveInt = 1
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    activation: ActivationName = "gelu"

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "target_encoder_dim": int(self.target_dim),
            "historical_encoder_dim": int(self.historical_dim),
            "future_encoder_dim": int(self.future_dim),
            "static_encoder_dim": int(self.static_dim),
            "fusion_hidden_size": int(self.fusion_hidden_size),
            "target_encoder_num_layers": int(self.target_num_layers),
            "historical_encoder_num_layers": int(self.historical_num_layers),
            "future_encoder_num_layers": int(self.future_num_layers),
            "static_encoder_num_layers": int(self.static_num_layers),
            "fusion_num_layers": int(self.fusion_num_layers),
            "modality_encoder_dropout_rate": float(self.dropout_rate),
            "modality_encoder_activation": str(self.activation),
        }


class ModalityEncoderHPOSpace(_StrictConfig):
    """Espacio arquitectónico que Optuna explora de forma efectiva."""

    enabled: bool = True
    target_dim_choices: Tuple[PositiveInt, ...] = (16, 32, 64, 128)
    historical_dim_choices: Tuple[PositiveInt, ...] = (16, 32, 64, 128)
    future_dim_choices: Tuple[PositiveInt, ...] = (16, 32, 64, 128)
    static_dim_choices: Tuple[PositiveInt, ...] = (8, 16, 32, 64)
    fusion_hidden_size_choices: Tuple[PositiveInt, ...] = (32, 64, 128, 256)
    target_layers: IntegerRange = IntegerRange(minimum=1, maximum=3)
    historical_layers: IntegerRange = IntegerRange(minimum=1, maximum=3)
    future_layers: IntegerRange = IntegerRange(minimum=1, maximum=3)
    static_layers: IntegerRange = IntegerRange(minimum=1, maximum=2)
    fusion_layers: IntegerRange = IntegerRange(minimum=1, maximum=3)
    dropout: FloatRange = FloatRange(minimum=0.0, maximum=0.35)
    activations: Tuple[ActivationName, ...] = ("relu", "gelu", "silu", "tanh")

    @field_validator(
        "target_dim_choices",
        "historical_dim_choices",
        "future_dim_choices",
        "static_dim_choices",
        "fusion_hidden_size_choices",
        "activations",
    )
    @classmethod
    def validate_choices(cls, values: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not values:
            raise ValueError("HPO choices must not be empty")
        if len(set(values)) != len(values):
            raise ValueError("HPO choices must not contain duplicates")
        return values

    def as_mapping(self) -> Mapping[str, Any]:
        return self.model_dump(mode="python")


class ResidualDecoderConfig(_StrictConfig):
    """Configuración fija del residual local vigente en Stage 2."""

    enabled: bool = True
    regularization_lambda: NonNegativeFloat = 0.01
    global_aux_alpha: NonNegativeFloat = 0.20
    hidden_size: PositiveInt = 32
    num_layers: PositiveInt = 1
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "use_local_residual_decoder": bool(self.enabled),
            "local_residual_lambda": float(self.regularization_lambda),
            "global_aux_alpha": float(self.global_aux_alpha),
            "local_residual_hidden_size": int(self.hidden_size),
            "local_residual_num_layers": int(self.num_layers),
            "local_residual_dropout_rate": float(self.dropout_rate),
        }


class AuxiliaryHeadsConfig(_StrictConfig):
    """Heads agnósticas y ponderación normalizada de sus losses."""

    use_event_head: bool = True
    use_magnitude_head: bool = True
    use_direction_head: bool = True
    use_normalized_loss_block: bool = True
    auxiliary_loss_weight: NonNegativeFloat = 0.20
    event_loss_share: NonNegativeFloat = 0.40
    magnitude_loss_share: NonNegativeFloat = 0.40
    direction_loss_share: NonNegativeFloat = 0.20
    hpo_loss_weights: bool = True
    legacy_event_loss_weight: NonNegativeFloat = 0.10
    legacy_magnitude_loss_weight: NonNegativeFloat = 0.10
    legacy_direction_loss_weight: NonNegativeFloat = 0.05
    hidden_size: PositiveInt = 32
    num_layers: PositiveInt = 1
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    event_threshold: PositiveFloat = 1.0
    magnitude_transform: MagnitudeTransform = "asinh"

    @model_validator(mode="after")
    def validate_loss_shares(self) -> "AuxiliaryHeadsConfig":
        enabled_heads = (
            self.use_event_head,
            self.use_magnitude_head,
            self.use_direction_head,
        )
        if self.use_normalized_loss_block and any(enabled_heads):
            total = (
                float(self.event_loss_share)
                + float(self.magnitude_loss_share)
                + float(self.direction_loss_share)
            )
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("auxiliary loss shares must sum to 1.0")
        return self

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "use_event_head": bool(self.use_event_head),
            "event_loss_weight": float(self.legacy_event_loss_weight),
            "use_magnitude_head": bool(self.use_magnitude_head),
            "magnitude_loss_weight": float(self.legacy_magnitude_loss_weight),
            "use_direction_head": bool(self.use_direction_head),
            "direction_loss_weight": float(self.legacy_direction_loss_weight),
            "use_auxiliary_loss_block": bool(self.use_normalized_loss_block),
            "auxiliary_loss_weight": float(self.auxiliary_loss_weight),
            "event_loss_share": float(self.event_loss_share),
            "magnitude_loss_share": float(self.magnitude_loss_share),
            "direction_loss_share": float(self.direction_loss_share),
            "hpo_auxiliary_loss_weights": bool(self.hpo_loss_weights),
            "auxiliary_head_hidden_size": int(self.hidden_size),
            "auxiliary_head_num_layers": int(self.num_layers),
            "auxiliary_head_dropout_rate": float(self.dropout_rate),
            "event_threshold": float(self.event_threshold),
            "magnitude_transform": str(self.magnitude_transform),
        }


class TrainingBudgetConfig(_StrictConfig):
    """Presupuesto de búsqueda y entrenamiento productivo pooled."""

    hpo_trials: PositiveInt = 80
    hpo_epochs: PositiveInt = 5
    hpo_windows_per_series: PositiveInt = 8
    hpo_validation_windows_per_series: PositiveInt = 5
    hpo_batch: PositiveInt = 512
    hpo_reduction_factor: PositiveInt = 3
    hpo_finalists: PositiveInt = 8
    hpo_fidelity_epochs: PositiveInt = 12
    hpo_fidelity_windows_per_series: PositiveInt = 16
    hpo_timeout_seconds: PositiveFloat | None = None
    training_strategy: Literal["pooled_balanced"] = "pooled_balanced"
    pooled_train_epochs: PositiveInt = 60
    pooled_train_batch: PositiveInt = 512
    pooled_continuation_epochs: NonNegativeInt = 0
    pooled_continuation_lr_factor: float = Field(default=0.20, gt=0.0, le=1.0)
    train_samples_per_epoch: NonNegativeInt = 16384
    patience: PositiveInt = 5
    nonfinite_max_retries: NonNegativeInt = 3
    nonfinite_lr_factor: float = Field(default=0.20, gt=0.0, lt=1.0)
    loss_function: LossName = "huber"
    selection_metric: SelectionMetric = "robust_macro_mase"

    @model_validator(mode="after")
    def validate_hpo_fidelity(self) -> "TrainingBudgetConfig":
        if self.hpo_reduction_factor < 2:
            raise ValueError("hpo_reduction_factor must be at least 2")
        if self.hpo_finalists > self.hpo_trials:
            raise ValueError("hpo_finalists cannot exceed hpo_trials")
        if self.hpo_fidelity_epochs < self.hpo_epochs:
            raise ValueError("medium-fidelity epochs must be >= proxy epochs")
        if self.hpo_fidelity_windows_per_series < self.hpo_windows_per_series:
            raise ValueError(
                "medium-fidelity windows must be >= proxy windows per series"
            )
        return self


class InferenceConfig(_StrictConfig):
    """Inferencia MC-Dropout, visualización y exportación."""

    n_monte_carlo: PositiveInt = 100
    mc_batch_size: PositiveInt = 128
    show_plots: bool = True
    plot_series: Tuple[str, ...] = ()
    plot_max_series: NonNegativeInt = 50
    backtest_start: str = ""
    backtest_end: str = ""
    forecast_start: str = ""
    forecast_end: str = ""
    forecast_batch_size: PositiveInt = 256
    export_forecasts: bool = True

    @field_validator("plot_series")
    @classmethod
    def validate_plot_series(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("plot_series must not contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("plot_series must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_forecast_date_pair(self) -> "InferenceConfig":
        # El forecast por rango necesita ambos límites. La visualización de
        # backtest permite fijar sólo uno de ellos.
        if bool(self.forecast_start) != bool(self.forecast_end):
            raise ValueError("forecast_start and forecast_end must be defined together")
        return self


class GlobalActiveConfiguration(_StrictConfig):
    """Contrato raíz de la superficie activa de los cuatro notebooks."""

    schema_version: Literal["22.3.2b"] = "22.3.2b"
    features: ModelFeatureConfig
    temporal: TemporalForecastConfig = TemporalForecastConfig()
    modality_defaults: ModalityEncoderDefaults
    modality_hpo: ModalityEncoderHPOSpace
    residual: ResidualDecoderConfig
    auxiliary: AuxiliaryHeadsConfig
    budget: TrainingBudgetConfig
    inference: InferenceConfig

    @model_validator(mode="after")
    def validate_cross_config(self) -> "GlobalActiveConfiguration":
        if self.modality_hpo.enabled and not self.features.use_modality_specific_encoders:
            raise ValueError(
                "modality HPO requires use_modality_specific_encoders=True"
            )
        return self

    def training_kwargs(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "use_auxiliary_autoencoder": bool(
                self.features.use_auxiliary_autoencoder
            ),
            "use_modality_specific_encoders": bool(
                self.features.use_modality_specific_encoders
            ),
        }
        payload.update(self.modality_defaults.training_kwargs())
        payload.update(self.residual.training_kwargs())
        payload.update(self.auxiliary.training_kwargs())
        return payload

    def to_dict(self) -> Mapping[str, Any]:
        return self.model_dump(mode="python")
