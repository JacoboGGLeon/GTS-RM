"""Plan de entrenamiento productivo para los modelos globales.

La ruta oficial es ``pooled_balanced``: todas las series elegibles participan
desde el primer epoch con sampling balanceado. El curriculum secuencial y el
orden shuffled se conservan únicamente como ablations reproducibles.

``cross_key_id`` y ``nivel_curriculum`` se usan sólo para sampling y métricas;
nunca se entregan al ``forward`` del modelo.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import random
from typing import Any, Callable, Dict, Final, Iterator, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler

from global_contracts import SUPPORTED_ARCHITECTURES
from global_data import GlobalBalancedSampler, GlobalWindowDataset
from global_models import GlobalForecastModel, build_global_model
from global_training import (
    GlobalDatasetBundle,
    GlobalHPOResult,
    GlobalTrainingConfig,
    GlobalValidationMetrics,
    NonFiniteValidationError,
    evaluate_global_model,
    global_forecast_loss,
    validation_objective,
)


CURRICULUM_PHASES: Final[Tuple[str, ...]] = (
    "productive_training",
    "pooled_continuation",
    "warmup",
    "finetune",
    "consolidation",
)
TRAINING_ORDERS: Final[Tuple[str, ...]] = (
    "pooled_balanced",
    "curriculum",
    "shuffled",
)
LEGACY_TRAINING_ORDERS: Final[Tuple[str, ...]] = ("pooled",)


@dataclass(frozen=True)
class GlobalTrainingScheduleConfig:
    """Plan posterior al HPO con pooled balanceado como ruta oficial.

    ``pooled_balanced`` entrena todas las series elegibles desde el primer
    epoch mediante el sampler balanceado. La continuación opcional conserva el
    mismo dataset y objetivo, y sólo reduce el learning rate. ``curriculum`` y
    ``shuffled`` permanecen disponibles como ablations históricas explícitas.
    """

    pooled_train_epochs: int = 60
    pooled_continuation_epochs: int = 0
    pooled_continuation_lr_factor: float = 0.2
    training_order: str = "pooled_balanced"

    # Parámetros exclusivos de las ablations históricas.
    warmup_epochs: int = 20
    fine_tune_epochs_per_level: int = 20
    consolidation_epochs: int = 10
    replay_fraction: float = 0.25
    fine_tune_lr_factor: float = 0.2
    consolidation_lr_factor: float = 0.1

    @property
    def is_pooled(self) -> bool:
        return self.training_order in {"pooled_balanced", "pooled"}

    def validate(self) -> None:
        _positive_int(self.pooled_train_epochs, "pooled_train_epochs")
        _non_negative_int(
            self.pooled_continuation_epochs,
            "pooled_continuation_epochs",
        )
        _positive_factor(
            self.pooled_continuation_lr_factor,
            "pooled_continuation_lr_factor",
        )
        if float(self.pooled_continuation_lr_factor) > 1.0:
            raise ValueError("pooled_continuation_lr_factor cannot exceed 1.0")

        _positive_int(self.warmup_epochs, "warmup_epochs")
        _positive_int(self.fine_tune_epochs_per_level, "fine_tune_epochs_per_level")
        _non_negative_int(self.consolidation_epochs, "consolidation_epochs")
        if not 0.0 <= float(self.replay_fraction) < 1.0:
            raise ValueError("replay_fraction must be in the interval [0, 1)")
        _positive_factor(self.fine_tune_lr_factor, "fine_tune_lr_factor")
        _positive_factor(self.consolidation_lr_factor, "consolidation_lr_factor")
        if self.consolidation_lr_factor > self.fine_tune_lr_factor:
            raise ValueError(
                "consolidation_lr_factor cannot exceed fine_tune_lr_factor"
            )
        supported = TRAINING_ORDERS + LEGACY_TRAINING_ORDERS
        if self.training_order not in supported:
            raise ValueError(f"training_order must be one of {supported}")

    def build_stages(self, curriculum_levels: Sequence[int]) -> Tuple["CurriculumStage", ...]:
        """Construye el plan productivo o una ablation histórica explícita."""

        self.validate()
        levels = tuple(sorted({int(level) for level in curriculum_levels}))
        if not levels:
            raise ValueError("At least one curriculum level is required")
        if any(level <= 0 for level in levels):
            raise ValueError("Curriculum levels must be positive integers")

        if self.training_order == "pooled_balanced":
            stages: list[CurriculumStage] = [
                CurriculumStage(
                    name="pooled_full_training",
                    phase="productive_training",
                    current_levels=levels,
                    replay_levels=(),
                    epochs=self.pooled_train_epochs,
                    learning_rate_factor=1.0,
                    replay_fraction=0.0,
                )
            ]
            if self.pooled_continuation_epochs > 0:
                stages.append(
                    CurriculumStage(
                        name="pooled_continuation",
                        phase="pooled_continuation",
                        current_levels=levels,
                        replay_levels=(),
                        epochs=self.pooled_continuation_epochs,
                        learning_rate_factor=self.pooled_continuation_lr_factor,
                        replay_fraction=0.0,
                    )
                )
            return tuple(stages)

        # Compatibilidad de carga para artefactos 22.1-22.3. Los notebooks
        # nuevos nunca emiten este valor; usan ``pooled_balanced``.
        if self.training_order == "pooled":
            return (
                CurriculumStage(
                    name="pooled_balanced_all_levels",
                    phase="warmup",
                    current_levels=levels,
                    replay_levels=(),
                    epochs=self.warmup_epochs,
                    learning_rate_factor=1.0,
                    replay_fraction=0.0,
                ),
            )

        if self.training_order == "shuffled":
            stages: list[CurriculumStage] = [
                CurriculumStage(
                    name="shuffled_warmup_all_levels",
                    phase="warmup",
                    current_levels=levels,
                    replay_levels=(),
                    epochs=self.warmup_epochs,
                    learning_rate_factor=1.0,
                    replay_fraction=0.0,
                )
            ]
            for index, _ in enumerate(levels[1:], start=1):
                stages.append(
                    CurriculumStage(
                        name=f"shuffled_finetune_all_levels_{index}",
                        phase="finetune",
                        current_levels=levels,
                        replay_levels=(),
                        epochs=self.fine_tune_epochs_per_level,
                        learning_rate_factor=self.fine_tune_lr_factor,
                        replay_fraction=0.0,
                    )
                )
            if self.consolidation_epochs > 0:
                stages.append(
                    CurriculumStage(
                        name="shuffled_consolidation_all_levels",
                        phase="consolidation",
                        current_levels=levels,
                        replay_levels=(),
                        epochs=self.consolidation_epochs,
                        learning_rate_factor=self.consolidation_lr_factor,
                        replay_fraction=0.0,
                    )
                )
            return tuple(stages)

        warmup_level = levels[0]
        stages: list[CurriculumStage] = [
            CurriculumStage(
                name=f"warmup_level_{warmup_level}",
                phase="warmup",
                current_levels=(warmup_level,),
                replay_levels=(),
                epochs=self.warmup_epochs,
                learning_rate_factor=1.0,
                replay_fraction=0.0,
            )
        ]

        previous_levels: list[int] = [warmup_level]
        for level in levels[1:]:
            stages.append(
                CurriculumStage(
                    name=f"finetune_level_{level}",
                    phase="finetune",
                    current_levels=(level,),
                    replay_levels=tuple(previous_levels),
                    epochs=self.fine_tune_epochs_per_level,
                    learning_rate_factor=self.fine_tune_lr_factor,
                    replay_fraction=self.replay_fraction,
                )
            )
            previous_levels.append(level)

        if self.consolidation_epochs > 0:
            stages.append(
                CurriculumStage(
                    name="consolidation_all_levels",
                    phase="consolidation",
                    current_levels=levels,
                    replay_levels=(),
                    epochs=self.consolidation_epochs,
                    learning_rate_factor=self.consolidation_lr_factor,
                    replay_fraction=0.0,
                )
            )
        return tuple(stages)


@dataclass(frozen=True)
class GlobalCurriculumConfig(GlobalTrainingScheduleConfig):
    """Configuración legacy para reproducir la ablation curricular."""

    training_order: str = "curriculum"


@dataclass(frozen=True)
class CurriculumStage:
    """Una fase de entrenamiento sin reinicializar el modelo."""

    name: str
    phase: str
    current_levels: Tuple[int, ...]
    replay_levels: Tuple[int, ...]
    epochs: int
    learning_rate_factor: float
    replay_fraction: float

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("stage name must not be empty")
        if self.phase not in CURRICULUM_PHASES:
            raise ValueError(f"Unsupported curriculum phase={self.phase!r}")
        if not self.current_levels:
            raise ValueError("current_levels must not be empty")
        if set(self.current_levels) & set(self.replay_levels):
            raise ValueError("current_levels and replay_levels must be disjoint")
        if any(int(level) <= 0 for level in (*self.current_levels, *self.replay_levels)):
            raise ValueError("Curriculum levels must be positive integers")
        _positive_int(self.epochs, "stage epochs")
        _positive_factor(self.learning_rate_factor, "learning_rate_factor")
        if not 0.0 <= float(self.replay_fraction) < 1.0:
            raise ValueError("stage replay_fraction must be in the interval [0, 1)")
        if self.replay_fraction > 0.0 and not self.replay_levels:
            raise ValueError("replay_levels are required when replay_fraction is positive")


class CurriculumReplaySampler(Sampler[int]):
    """Muestrea series del nivel actual y replay de niveles anteriores.

    Cada extracción selecciona primero el pool (actual/replay), después una
    identidad uniformemente y finalmente una ventana de esa identidad. Así el
    replay no queda dominado por series largas.
    """

    def __init__(
        self,
        dataset: GlobalWindowDataset,
        *,
        current_levels: Sequence[int],
        replay_levels: Sequence[int] = (),
        replay_fraction: float = 0.0,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset, GlobalWindowDataset):
            raise TypeError("dataset must be a GlobalWindowDataset")
        self.dataset = dataset
        self.current_levels = tuple(sorted({int(level) for level in current_levels}))
        self.replay_levels = tuple(sorted({int(level) for level in replay_levels}))
        if not self.current_levels:
            raise ValueError("current_levels must not be empty")
        if set(self.current_levels) & set(self.replay_levels):
            raise ValueError("current_levels and replay_levels must be disjoint")
        if not 0.0 <= float(replay_fraction) < 1.0:
            raise ValueError("replay_fraction must be in the interval [0, 1)")
        if replay_fraction > 0.0 and not self.replay_levels:
            raise ValueError("replay_levels are required when replay_fraction is positive")

        requested_samples = len(dataset) if num_samples is None else num_samples
        _positive_int(requested_samples, "num_samples")
        self.num_samples = int(requested_samples)
        self.replay_fraction = float(replay_fraction)
        if self.replay_fraction > 0.0 and self.num_samples < 2:
            raise ValueError(
                "num_samples must be at least 2 when replay_fraction is positive"
            )
        self.seed = int(seed)
        self.epoch = 0

        level_by_series = dataset.series_curriculum_levels
        self._current_series = tuple(
            series_id
            for series_id in dataset.series_ids
            if level_by_series[series_id] in self.current_levels
        )
        self._replay_series = tuple(
            series_id
            for series_id in dataset.series_ids
            if level_by_series[series_id] in self.replay_levels
        )
        if not self._current_series:
            raise ValueError(
                f"No training series found for current_levels={self.current_levels}"
            )
        if self.replay_fraction > 0.0 and not self._replay_series:
            raise ValueError(
                f"No replay series found for replay_levels={self.replay_levels}"
            )
        self._indices_by_series = dataset.indices_by_series
        self.last_draw_counts: Mapping[str, int] = {"current": 0, "replay": 0}
        self.last_stratum_counts: Mapping[str, int] = {}

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        replay_target = 0
        if self._replay_series and self.replay_fraction > 0.0:
            replay_target = max(
                1,
                min(
                    self.num_samples - 1,
                    int(round(self.num_samples * self.replay_fraction)),
                ),
            )
        current_target = self.num_samples - replay_target

        current_sampler = GlobalBalancedSampler(
            self.dataset,
            num_samples=current_target,
            seed=self.seed + 17,
            series_ids=self._current_series,
        )
        current_sampler.set_epoch(self.epoch)
        current_indices = list(iter(current_sampler))

        replay_indices: list[int] = []
        replay_counts: Mapping[str, int] = {}
        if replay_target > 0:
            replay_sampler = GlobalBalancedSampler(
                self.dataset,
                num_samples=replay_target,
                seed=self.seed + 31,
                series_ids=self._replay_series,
            )
            replay_sampler.set_epoch(self.epoch)
            replay_indices = list(iter(replay_sampler))
            replay_counts = replay_sampler.last_draw_counts

        tagged = [(index, False) for index in current_indices] + [
            (index, True) for index in replay_indices
        ]
        rng.shuffle(tagged)
        self.last_draw_counts = {
            "current": len(current_indices),
            "replay": len(replay_indices),
        }
        stratum_counts: Dict[str, int] = {}
        for prefix, counts in (
            ("current", current_sampler.last_draw_counts),
            ("replay", replay_counts),
        ):
            for key, value in counts.items():
                stratum_counts[f"{prefix}:{key}"] = int(value)
        self.last_stratum_counts = dict(sorted(stratum_counts.items()))
        for index, _ in tagged:
            yield int(index)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch


@dataclass(frozen=True)
class GlobalCurriculumEpochRecord:
    stage_name: str
    phase: str
    global_epoch: int
    stage_epoch: int
    train_loss: float
    validation_objective: float
    learning_rate: float
    current_samples: int
    replay_samples: int
    stratum_samples: Mapping[str, int]
    validation: Mapping[str, GlobalValidationMetrics]
    recovery_retries: int = 0
    stability_events: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GlobalCurriculumStageResult:
    stage: CurriculumStage
    history: Tuple[GlobalCurriculumEpochRecord, ...]
    best_epoch: int
    best_score: float
    validation: Mapping[str, GlobalValidationMetrics]
    start_state_digest: str
    end_state_digest: str
    stopped_early: bool


@dataclass
class GlobalCurriculumTrainingResult:
    """Modelo final y evidencia del plan de entrenamiento ejecutado."""

    architecture: str
    model: GlobalForecastModel
    model_config: Mapping[str, Any]
    training_config: GlobalTrainingConfig
    curriculum_config: GlobalCurriculumConfig
    stages: Tuple[GlobalCurriculumStageResult, ...]
    history: Tuple[GlobalCurriculumEpochRecord, ...]
    validation: Mapping[str, GlobalValidationMetrics]
    best_score: float
    total_epochs: int

    def to_summary(self) -> Mapping[str, Any]:
        return {
            "architecture": self.architecture,
            "model_config": dict(self.model_config),
            "training_config": asdict(self.training_config),
            "training_schedule_config": asdict(self.curriculum_config),
            "best_score": self.best_score,
            "total_epochs": self.total_epochs,
            "stages": [
                {
                    "name": result.stage.name,
                    "phase": result.stage.phase,
                    "current_levels": list(result.stage.current_levels),
                    "replay_levels": list(result.stage.replay_levels),
                    "replay_fraction": result.stage.replay_fraction,
                    "best_epoch": result.best_epoch,
                    "best_score": result.best_score,
                    "epochs_completed": len(result.history),
                    "stopped_early": result.stopped_early,
                    "start_state_digest": result.start_state_digest,
                    "end_state_digest": result.end_state_digest,
                }
                for result in self.stages
            ],
        }


CurriculumEpochCallback = Callable[[GlobalCurriculumEpochRecord], None]


class GlobalCurriculumSession:
    """Sesión continua para un único modelo global.

    Conserva el mismo modelo y optimizador entre etapas del schedule. En la
    ruta oficial pooled, la continuación opcional mantiene la misma distribución
    de datos y sólo aplica un factor menor de learning rate.
    """

    def __init__(
        self,
        architecture: str,
        model_config: Mapping[str, Any],
        datasets: GlobalDatasetBundle,
        training_config: GlobalTrainingConfig | None = None,
        curriculum_config: GlobalCurriculumConfig | None = None,
    ) -> None:
        normalized = str(architecture).strip().lower()
        if normalized not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture={architecture!r}; expected {SUPPORTED_ARCHITECTURES}"
            )
        datasets.validate()
        self.architecture = normalized
        self.model_config = dict(model_config)
        self.datasets = datasets
        self.training_config = training_config or GlobalTrainingConfig()
        self.curriculum_config = curriculum_config or GlobalCurriculumConfig()
        self.training_config.validate()
        self.curriculum_config.validate()
        _seed_everything(self.training_config.seed)
        self.device = _resolve_device(self.training_config.device)

        levels = tuple(sorted(set(datasets.train.series_curriculum_levels.values())))
        self.stages = self.curriculum_config.build_stages(levels)
        for stage in self.stages:
            stage.validate()

        self.model = build_global_model(
            self.architecture,
            self.model_config,
            window_size=datasets.window_size,
            horizon=datasets.horizon,
            exogenous_dim=datasets.exogenous_dim,
            static_dim=datasets.static_dim,
        ).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.validation_loaders = {
            name: _make_loader(
                dataset,
                batch_size=self.training_config.batch_size,
                sampler=None,
                num_workers=self.training_config.num_workers,
                device=self.device,
            )
            for name, dataset in datasets.validation_datasets.items()
        }
        self.global_epoch = 0
        self.stage_cursor = 0
        self.history: list[GlobalCurriculumEpochRecord] = []
        self.stage_results: list[GlobalCurriculumStageResult] = []

    @property
    def completed_phases(self) -> Tuple[str, ...]:
        return tuple(result.stage.phase for result in self.stage_results)

    @property
    def pending_stages(self) -> Tuple[CurriculumStage, ...]:
        return tuple(self.stages[self.stage_cursor :])

    def run_phases(
        self,
        phases: Sequence[str],
        *,
        epoch_callback: CurriculumEpochCallback | None = None,
        epochs_override: int | None = None,
        batch_size_override: int | None = None,
    ) -> GlobalCurriculumTrainingResult:
        requested = tuple(str(value).strip().lower() for value in phases)
        if not requested:
            raise ValueError("phases must not be empty")
        invalid = [phase for phase in requested if phase not in CURRICULUM_PHASES]
        if invalid:
            raise ValueError(f"Unsupported curriculum phases: {invalid}")
        if epochs_override is not None:
            _positive_int(epochs_override, "epochs_override")
        if batch_size_override is not None:
            _positive_int(batch_size_override, "batch_size_override")

        executed = 0
        while self.stage_cursor < len(self.stages):
            stage = self.stages[self.stage_cursor]
            if stage.phase not in requested:
                break
            effective_stage = stage
            if epochs_override is not None:
                effective_stage = CurriculumStage(
                    name=stage.name,
                    phase=stage.phase,
                    current_levels=stage.current_levels,
                    replay_levels=stage.replay_levels,
                    epochs=int(epochs_override),
                    learning_rate_factor=stage.learning_rate_factor,
                    replay_fraction=stage.replay_fraction,
                )
            self._run_stage(
                effective_stage,
                stage_index=self.stage_cursor,
                epoch_callback=epoch_callback,
                batch_size=batch_size_override or self.training_config.batch_size,
            )
            self.stage_cursor += 1
            executed += 1
        if executed == 0:
            next_phase = (
                self.stages[self.stage_cursor].phase
                if self.stage_cursor < len(self.stages)
                else None
            )
            raise RuntimeError(
                f"No pending stage matches phases={requested}; next_phase={next_phase!r}"
            )
        return self.result()

    def run_all(
        self,
        *,
        epoch_callback: CurriculumEpochCallback | None = None,
    ) -> GlobalCurriculumTrainingResult:
        while self.stage_cursor < len(self.stages):
            phase = self.stages[self.stage_cursor].phase
            self.run_phases((phase,), epoch_callback=epoch_callback)
        return self.result()

    def result(self) -> GlobalCurriculumTrainingResult:
        if not self.stage_results:
            raise RuntimeError("Curriculum session has not completed any stage")
        self.model.eval()
        final_stage = self.stage_results[-1]
        return GlobalCurriculumTrainingResult(
            architecture=self.architecture,
            model=self.model,
            model_config=dict(self.model_config),
            training_config=self.training_config,
            curriculum_config=self.curriculum_config,
            stages=tuple(self.stage_results),
            history=tuple(self.history),
            validation=final_stage.validation,
            best_score=final_stage.best_score,
            total_epochs=len(self.history),
        )

    def _run_stage(
        self,
        stage: CurriculumStage,
        *,
        stage_index: int,
        epoch_callback: CurriculumEpochCallback | None,
        batch_size: int,
    ) -> None:
        config = self.training_config
        start_digest = state_dict_digest(self.model.state_dict())
        requested_stage_lr = config.learning_rate * stage.learning_rate_factor
        current_lr = min(float(group["lr"]) for group in self.optimizer.param_groups)
        stage_lr = requested_stage_lr if stage_index == 0 else min(current_lr, requested_stage_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = stage_lr

        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=min(config.min_learning_rate, stage_lr),
        )
        sampler = CurriculumReplaySampler(
            self.datasets.train,
            current_levels=stage.current_levels,
            replay_levels=stage.replay_levels,
            replay_fraction=stage.replay_fraction,
            num_samples=config.samples_per_epoch,
            seed=config.seed + stage_index * 10_000,
        )
        train_loader = _make_loader(
            self.datasets.train,
            batch_size=int(batch_size),
            sampler=sampler,
            num_workers=config.num_workers,
            device=self.device,
        )

        best_score = math.inf
        best_epoch = 0
        best_model_state: Dict[str, torch.Tensor] | None = None
        best_optimizer_state: MutableMapping[str, Any] | None = None
        best_validation: Mapping[str, GlobalValidationMetrics] = {}
        epochs_without_improvement = 0
        stage_history: list[GlobalCurriculumEpochRecord] = []

        for stage_epoch in range(1, stage.epochs + 1):
            self.global_epoch += 1
            sampler.set_epoch(self.global_epoch - 1)
            retry_count = 0
            stability_events: list[str] = []

            while True:
                attempt_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                attempt_optimizer_state = deepcopy(self.optimizer.state_dict())
                try:
                    train_loss = _train_one_epoch(
                        self.model,
                        train_loader,
                        self.optimizer,
                        config,
                        self.device,
                    )
                    validation = {
                        name: evaluate_global_model(self.model, loader, device=self.device)
                        for name, loader in self.validation_loaders.items()
                    }
                    objective = validation_objective(
                        validation, metric=config.selection_metric
                    )
                    break
                except (FloatingPointError, NonFiniteValidationError) as exc:
                    self.model.load_state_dict(attempt_model_state)
                    self.model.to(self.device)
                    self.optimizer.load_state_dict(attempt_optimizer_state)
                    _move_optimizer_state(self.optimizer, self.device)
                    retry_count += 1
                    previous_lr = min(
                        float(group["lr"]) for group in self.optimizer.param_groups
                    )
                    reduced_lr = max(
                        float(config.min_learning_rate),
                        previous_lr * float(config.nonfinite_lr_factor),
                    )
                    event = (
                        f"retry={retry_count} reason={type(exc).__name__} "
                        f"lr={previous_lr:.3e}->{reduced_lr:.3e}"
                    )
                    stability_events.append(event)
                    if (
                        retry_count > config.nonfinite_max_retries
                        or reduced_lr >= previous_lr
                    ):
                        raise RuntimeError(
                            f"Curriculum stage {stage.name!r} epoch {stage_epoch} "
                            f"remained numerically unstable after {retry_count} retries; "
                            f"last_error={exc}"
                        ) from exc
                    for group in self.optimizer.param_groups:
                        group["lr"] = reduced_lr

            scheduler.step(objective)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            draw_counts = sampler.last_draw_counts
            record = GlobalCurriculumEpochRecord(
                stage_name=stage.name,
                phase=stage.phase,
                global_epoch=self.global_epoch,
                stage_epoch=stage_epoch,
                train_loss=train_loss,
                validation_objective=objective,
                learning_rate=learning_rate,
                current_samples=int(draw_counts["current"]),
                replay_samples=int(draw_counts["replay"]),
                stratum_samples=dict(sampler.last_stratum_counts),
                validation=validation,
                recovery_retries=retry_count,
                stability_events=tuple(stability_events),
            )
            stage_history.append(record)
            self.history.append(record)
            if epoch_callback is not None:
                epoch_callback(record)

            if objective < best_score - config.min_delta:
                best_score = objective
                best_epoch = stage_epoch
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                best_optimizer_state = deepcopy(self.optimizer.state_dict())
                best_validation = deepcopy(validation)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

        if best_model_state is None or best_optimizer_state is None:
            raise RuntimeError(f"Curriculum stage {stage.name!r} produced no finite checkpoint")
        self.model.load_state_dict(best_model_state)
        self.model.to(self.device)
        self.optimizer.load_state_dict(best_optimizer_state)
        _move_optimizer_state(self.optimizer, self.device)
        end_digest = state_dict_digest(self.model.state_dict())
        self.stage_results.append(
            GlobalCurriculumStageResult(
                stage=stage,
                history=tuple(stage_history),
                best_epoch=best_epoch,
                best_score=float(best_score),
                validation=best_validation,
                start_state_digest=start_digest,
                end_state_digest=end_digest,
                stopped_early=len(stage_history) < stage.epochs,
            )
        )


class GlobalCurriculumTrainer:
    """Entrenador de conveniencia que ejecuta una sesión completa."""

    def __init__(
        self,
        architecture: str,
        model_config: Mapping[str, Any],
        training_config: GlobalTrainingConfig | None = None,
        curriculum_config: GlobalCurriculumConfig | None = None,
    ) -> None:
        self.architecture = str(architecture).strip().lower()
        self.model_config = dict(model_config)
        self.training_config = training_config or GlobalTrainingConfig()
        self.curriculum_config = curriculum_config or GlobalCurriculumConfig()

    def fit(
        self,
        datasets: GlobalDatasetBundle,
        *,
        epoch_callback: CurriculumEpochCallback | None = None,
    ) -> GlobalCurriculumTrainingResult:
        return GlobalCurriculumSession(
            self.architecture,
            self.model_config,
            datasets,
            self.training_config,
            self.curriculum_config,
        ).run_all(epoch_callback=epoch_callback)


def fit_best_candidate_with_curriculum(
    hpo_result: GlobalHPOResult,
    datasets: GlobalDatasetBundle,
    *,
    curriculum_config: GlobalCurriculumConfig | None = None,
    epoch_callback: CurriculumEpochCallback | None = None,
) -> GlobalCurriculumTrainingResult:
    """Entrena productivamente el candidato ganador del HPO con curriculum."""

    if not isinstance(hpo_result, GlobalHPOResult):
        raise TypeError("hpo_result must be a GlobalHPOResult")
    candidate = hpo_result.best_candidate
    if candidate.window_size != datasets.window_size:
        raise ValueError(
            "HPO candidate window_size does not match the curriculum datasets"
        )
    return GlobalCurriculumTrainer(
        hpo_result.architecture,
        candidate.model_config,
        candidate.training_config,
        curriculum_config,
    ).fit(datasets, epoch_callback=epoch_callback)



@dataclass(frozen=True)
class GlobalTrainingOrderAblationResult:
    """Controlled curriculum-vs-shuffled comparison with equal epoch budgets."""

    curriculum: GlobalCurriculumTrainingResult
    shuffled: GlobalCurriculumTrainingResult
    objective_metric: str

    @property
    def winner(self) -> str:
        if self.curriculum.best_score <= self.shuffled.best_score:
            return "curriculum"
        return "shuffled"

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "objective_metric": self.objective_metric,
            "winner": self.winner,
            "curriculum_score": float(self.curriculum.best_score),
            "shuffled_score": float(self.shuffled.best_score),
            "delta_shuffled_minus_curriculum": float(
                self.shuffled.best_score - self.curriculum.best_score
            ),
            "curriculum_epochs": int(self.curriculum.total_epochs),
            "shuffled_epochs": int(self.shuffled.total_epochs),
        }


def compare_curriculum_vs_shuffled(
    architecture: str,
    model_config: Mapping[str, Any],
    datasets: GlobalDatasetBundle,
    training_config: GlobalTrainingConfig,
    curriculum_config: GlobalCurriculumConfig,
) -> GlobalTrainingOrderAblationResult:
    """Run a paired ablation with identical data, seeds, stages and LR factors."""

    curriculum_cfg = replace(curriculum_config, training_order="curriculum")
    shuffled_cfg = replace(curriculum_config, training_order="shuffled")
    curriculum_result = GlobalCurriculumTrainer(
        architecture, model_config, training_config, curriculum_cfg
    ).fit(datasets)
    shuffled_result = GlobalCurriculumTrainer(
        architecture, model_config, training_config, shuffled_cfg
    ).fit(datasets)
    if curriculum_result.total_epochs != shuffled_result.total_epochs:
        raise RuntimeError("Training-order ablation did not preserve the epoch budget")
    return GlobalTrainingOrderAblationResult(
        curriculum=curriculum_result,
        shuffled=shuffled_result,
        objective_metric=training_config.selection_metric,
    )

def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Huella estable para probar continuidad de pesos entre etapas."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _train_one_epoch(
    model: GlobalForecastModel,
    loader: DataLoader,
    optimizer: AdamW,
    config: GlobalTrainingConfig,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    observed = 0
    for batch in loader:
        model_inputs = {
            name: tensor.to(device)
            for name, tensor in batch["model_inputs"].items()
        }
        target = batch["targets"]["y_future"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs)
        loss = global_forecast_loss(
            output,
            target,
            loss=config.loss,
            huber_delta=config.huber_delta,
            auxiliary_targets=batch["targets"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite curriculum training loss")
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        bad_parameters = [
            name
            for name, parameter in model.named_parameters()
            if not torch.all(torch.isfinite(parameter.detach()))
        ]
        if bad_parameters:
            raise FloatingPointError(
                "Non-finite curriculum parameters after optimizer.step: "
                + ", ".join(bad_parameters[:5])
            )

        batch_size = int(target.shape[0])
        weighted_loss += float(loss.detach().cpu()) * batch_size
        observed += batch_size

    if observed == 0:
        raise ValueError("Curriculum training loader produced no batches")
    return weighted_loss / observed


def _make_loader(
    dataset: GlobalWindowDataset,
    *,
    batch_size: int,
    sampler: Sampler[int] | None,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _non_negative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive_factor(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
        raise ValueError(f"{label} must be in the interval (0, 1]")
