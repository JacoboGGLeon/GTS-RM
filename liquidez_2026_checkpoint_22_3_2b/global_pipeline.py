"""Contratos Pydantic y workflow público de los notebooks globales GTRM.

Checkpoint 22.3.2b mantiene la ruta productiva y endurece el contrato temporal:

HPO proxy -> selección medium-fidelity -> pooled full training
-> pooled continuation opcional -> backtest -> forecast.

Las ablations curriculares permanecen en ``global_curriculum.py``, pero no
forman parte del contrato público de los cuatro notebooks globales.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from global_curriculum import (
    GlobalCurriculumTrainingResult,
    GlobalTrainingScheduleConfig,
)
from global_notebook import GlobalNotebookConfig
from global_training import GlobalHPOConfig, GlobalTrainingConfig
from gtrm_config import GTRMModelConfig
from global_surface_config import GlobalActiveConfiguration


class TrainingPhase(str, Enum):
    """Fases públicas del notebook en orden causal."""

    HPO_TRAINING = "hpo_and_pooled_training"
    BACKTEST = "backtest"
    FORECAST = "forecast"


TRAINING_PHASE_ORDER: Tuple[TrainingPhase, ...] = (
    TrainingPhase.HPO_TRAINING,
    TrainingPhase.BACKTEST,
    TrainingPhase.FORECAST,
)


class _StrictContract(BaseModel):
    """Base estricta para contratos serializables de frontera."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class GlobalNotebookRunContract(_StrictContract):
    """Fuente de verdad validada para un run global completo."""

    schema_version: Literal["22.3.2b"] = "22.3.2b"
    surface: GlobalActiveConfiguration | None = None
    notebook: GlobalNotebookConfig
    model: GTRMModelConfig
    training: GlobalTrainingConfig
    hpo: GlobalHPOConfig
    schedule: GlobalTrainingScheduleConfig
    phase_order: Tuple[TrainingPhase, ...] = TRAINING_PHASE_ORDER

    @model_validator(mode="after")
    def validate_cross_contract(self) -> "GlobalNotebookRunContract":
        self.notebook.validate()
        self.model.validate(stage=int(self.notebook.gtrm_stage))
        self.training.validate()
        self.hpo.validate()
        self.schedule.validate()

        architecture = str(self.notebook.architecture).strip().lower()
        if self.model.normalized_architecture() != architecture:
            raise ValueError("notebook and model architectures must match")
        if self.notebook.model_config != self.model:
            raise ValueError("notebook.model_config and model must be identical")
        if str(self.training.loss).strip().lower() != str(self.model.loss_type).strip().lower():
            raise ValueError("model.loss_type and training.loss must match")
        if int(self.training.epochs) != int(self.hpo.epochs):
            raise ValueError("training.epochs and hpo.epochs must match for the proxy stage")
        if self.hpo.objective_metric != self.training.selection_metric:
            raise ValueError(
                "hpo.objective_metric and training.selection_metric must match"
            )
        if self.schedule.training_order != "pooled_balanced":
            raise ValueError(
                "the public notebook contract requires training_order='pooled_balanced'"
            )
        if tuple(self.phase_order) != TRAINING_PHASE_ORDER:
            raise ValueError(
                "phase_order is immutable and must be: "
                + " -> ".join(phase.value for phase in TRAINING_PHASE_ORDER)
            )

        if self.surface is not None:
            surface = self.surface
            if surface.schema_version != self.schema_version:
                raise ValueError("surface and run contract schema versions must match")
            if int(surface.temporal.forecast_horizon) != int(
                self.notebook.forecast_horizon
            ):
                raise ValueError(
                    "surface temporal forecast_horizon and notebook must match"
                )
            if int(surface.temporal.rollout_chunk_size) != int(
                self.notebook.horizon
            ):
                raise ValueError(
                    "surface rollout_chunk_size and notebook model horizon must match"
                )
            if int(surface.temporal.training_stride) != int(self.notebook.stride):
                raise ValueError(
                    "surface training_stride and notebook stride must match"
                )
            if bool(surface.features.use_static_context) != bool(self.model.use_static_context):
                raise ValueError("surface and model use_static_context must match")
            if bool(surface.features.use_auxiliary_autoencoder) != bool(
                self.training.use_auxiliary_autoencoder
            ):
                raise ValueError(
                    "surface and training use_auxiliary_autoencoder must match"
                )
            expected_training = surface.training_kwargs()
            for field_name, expected_value in expected_training.items():
                actual_value = getattr(self.training, field_name)
                if actual_value != expected_value:
                    raise ValueError(
                        f"surface and training {field_name} must match"
                    )
            if dict(self.hpo.modality_encoder_hpo_space) != dict(
                surface.modality_hpo.model_dump(mode="python")
            ):
                raise ValueError("surface and HPO modality search spaces must match")
            budget = surface.budget
            hpo_pairs = {
                "epochs": budget.hpo_epochs,
                "windows_per_series_per_epoch": budget.hpo_windows_per_series,
                "validation_windows_per_series": budget.hpo_validation_windows_per_series,
                "reduction_factor": budget.hpo_reduction_factor,
                "finalists": budget.hpo_finalists,
                "fidelity_epochs": budget.hpo_fidelity_epochs,
                "fidelity_windows_per_series_per_epoch": (
                    budget.hpo_fidelity_windows_per_series
                ),
            }
            for field_name, expected_value in hpo_pairs.items():
                if getattr(self.hpo, field_name) != expected_value:
                    raise ValueError(f"surface and HPO {field_name} must match")
            if self.schedule.pooled_train_epochs != budget.pooled_train_epochs:
                raise ValueError("surface and schedule pooled_train_epochs must match")
            if (
                self.schedule.pooled_continuation_epochs
                != budget.pooled_continuation_epochs
            ):
                raise ValueError(
                    "surface and schedule pooled_continuation_epochs must match"
                )
            if (
                self.schedule.pooled_continuation_lr_factor
                != budget.pooled_continuation_lr_factor
            ):
                raise ValueError(
                    "surface and schedule pooled_continuation_lr_factor must match"
                )

        boolean_pairs = (
            (
                "use_modality_specific_encoders",
                self.model.use_modality_specific_encoders,
            ),
            ("use_local_residual_decoder", self.model.use_local_residual_decoder),
            ("use_event_head", self.model.use_event_head),
            ("use_magnitude_head", self.model.use_magnitude_head),
            ("use_direction_head", self.model.use_direction_head),
        )
        for field_name, model_value in boolean_pairs:
            training_value = bool(getattr(self.training, field_name))
            if training_value != bool(model_value):
                raise ValueError(
                    f"model.{field_name} and training.{field_name} must match"
                )
        if float(self.training.event_threshold) != float(self.model.event_threshold):
            raise ValueError("model and training event_threshold must match")
        if (
            str(self.training.magnitude_transform).strip().lower()
            != str(self.model.magnitude_transform).strip().lower()
        ):
            raise ValueError("model and training magnitude_transform must match")
        return self

    def to_dict(self) -> Mapping[str, Any]:
        """Serialización estable sin objetos runtime."""

        return {
            "schema_version": self.schema_version,
            "phase_order": [phase.value for phase in self.phase_order],
            "surface": (
                None if self.surface is None else self.surface.model_dump(mode="python")
            ),
            "notebook": _config_to_dict(self.notebook),
            "model": self.model.as_dict(),
            "training": _config_to_dict(self.training),
            "hpo": _config_to_dict(self.hpo),
            "schedule": _config_to_dict(self.schedule),
        }


class PooledTrainingRequest(_StrictContract):
    """Presupuesto de HPO y entrenamiento pooled del candidato ganador."""

    n_trials: PositiveInt
    train_epochs: PositiveInt
    continuation_epochs: NonNegativeInt = 0
    continuation_lr_factor: float = Field(default=0.2, gt=0.0, le=1.0)
    batch_size: PositiveInt
    timeout_seconds: PositiveFloat | None = None
    study_name: str = Field(min_length=1)
    storage_uri: str | None = None
    load_if_exists: bool = False

    @field_validator("storage_uri")
    @classmethod
    def normalize_storage_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class BacktestRequest(_StrictContract):
    """Contrato de backtest MC-Dropout posterior al entrenamiento."""

    n_mc: PositiveInt = 100
    batch_size: PositiveInt | None = None
    device: str | None = None
    show_progress: bool = True

    @field_validator("device")
    @classmethod
    def normalize_device(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class ForecastRequest(_StrictContract):
    """Contrato mutuamente excluyente para forecast por rango o por pasos.

    ``max_steps`` es el horizonte total autorizado por el run. También limita
    rangos explícitos una vez resueltos sobre el ``TemporalAxis``.
    """

    start_date: str | None = None
    end_date: str | None = None
    n_steps: PositiveInt | None = None
    max_steps: PositiveInt | None = None
    n_mc: PositiveInt = 100
    batch_size: PositiveInt | None = None
    device: str | None = None
    series_ids: Tuple[str, ...] = ()

    @field_validator("start_date", "end_date", "device")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("series_ids")
    @classmethod
    def normalize_series_ids(cls, values: Sequence[str]) -> Tuple[str, ...]:
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("series_ids must not contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("series_ids must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_forecast_mode(self) -> "ForecastRequest":
        has_start = self.start_date is not None
        has_end = self.end_date is not None
        if has_start != has_end:
            raise ValueError("start_date and end_date must be provided together")
        has_range = has_start and has_end
        if has_range == (self.n_steps is not None):
            raise ValueError(
                "forecast requires exactly one mode: date range or n_steps"
            )
        if (
            self.n_steps is not None
            and self.max_steps is not None
            and self.n_steps > self.max_steps
        ):
            raise ValueError("n_steps cannot exceed max_steps")
        return self


class WorkflowSnapshot(_StrictContract):
    """Estado serializable del workflow público."""

    completed_phases: Tuple[TrainingPhase, ...] = ()
    next_phase: TrainingPhase | None = TrainingPhase.HPO_TRAINING
    is_complete: bool = False


class GlobalTrainingWorkflow:
    """Máquina de estados que impide ejecutar fases fuera de orden."""

    def __init__(self, manager: Any, contract: GlobalNotebookRunContract) -> None:
        self.manager = manager
        self.contract = contract
        self._completed: list[TrainingPhase] = []

    @property
    def snapshot(self) -> WorkflowSnapshot:
        completed = tuple(self._completed)
        next_phase = (
            TRAINING_PHASE_ORDER[len(completed)]
            if len(completed) < len(TRAINING_PHASE_ORDER)
            else None
        )
        return WorkflowSnapshot(
            completed_phases=completed,
            next_phase=next_phase,
            is_complete=next_phase is None,
        )

    def run_hpo_and_train(
        self,
        dataset_factory: Callable[[int], Any],
        request: PooledTrainingRequest,
        *,
        split_manifest: Any = None,
        exogenous_columns: Sequence[str] = (),
        run_metadata: Mapping[str, Any] | None = None,
        epoch_callback: Callable[[Any], None] | None = None,
        show_progress: bool = True,
    ) -> GlobalCurriculumTrainingResult:
        self._require_next(TrainingPhase.HPO_TRAINING)
        result = self.manager.run_hpo_and_train(
            dataset_factory,
            n_trials=request.n_trials,
            train_epochs=request.train_epochs,
            continuation_epochs=request.continuation_epochs,
            continuation_lr_factor=request.continuation_lr_factor,
            batch=request.batch_size,
            timeout=request.timeout_seconds,
            study_name=request.study_name,
            hpo_storage=request.storage_uri,
            hpo_load_if_exists=request.load_if_exists,
            split_manifest=split_manifest,
            exogenous_columns=tuple(exogenous_columns),
            run_metadata=run_metadata,
            curriculum_epoch_callback=epoch_callback,
            show_progress=show_progress,
        )
        self._completed.append(TrainingPhase.HPO_TRAINING)
        return result

    def run_backtest(self, request: BacktestRequest) -> Mapping[str, Any]:
        self._require_next(TrainingPhase.BACKTEST)
        result = self.manager.run_backtest(
            n_mc=request.n_mc,
            batch_size=request.batch_size,
            device=request.device,
            show_progress=request.show_progress,
        )
        self._completed.append(TrainingPhase.BACKTEST)
        return result

    def run_forecast(self, request: ForecastRequest) -> Mapping[str, Any]:
        self._require_next(TrainingPhase.FORECAST)
        result = self.manager.run_future_forecast(
            start_date=request.start_date,
            end_date=request.end_date,
            n_steps=request.n_steps,
            max_steps=request.max_steps,
            n_mc=request.n_mc,
            batch_size=request.batch_size,
            device=request.device,
            series_ids=request.series_ids or None,
        )
        self._completed.append(TrainingPhase.FORECAST)
        return result

    def _require_next(self, requested: TrainingPhase) -> None:
        snapshot = self.snapshot
        if snapshot.next_phase != requested:
            actual = "complete" if snapshot.next_phase is None else snapshot.next_phase.value
            raise RuntimeError(
                f"Training phase out of order: requested={requested.value}, expected={actual}"
            )


def _config_to_dict(config: Any) -> Mapping[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="python")
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    if is_dataclass(config):
        return asdict(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")
