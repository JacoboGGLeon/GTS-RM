"""HPO y entrenamiento compartido para las arquitecturas globales.

Checkpoint 4 agrega únicamente la optimización y el ciclo de entrenamiento de
un modelo global por arquitectura. No incorpora curriculum learning,
orquestación de runs, notebooks ni persistencia en S3.

Principios del contrato:

- un único ``state_dict`` compartido por todas las series;
- batches balanceados primero por ``cross_key_id``;
- ``cross_key_id`` se usa sólo para agregar métricas, nunca como feature;
- HPO macro-promediado entre series y entre validación seen/unseen;
- el conjunto test unseen permanece fuera del HPO.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import math
import random
from typing import Any, Callable, Dict, Final, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import optuna
from optuna import Trial
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from global_contracts import (
    CROSS_KEY_COLUMN,
    GLOBAL_COMPONENT_FIELD,
    LOCAL_RESIDUAL_FIELD,
    SUPPORTED_ARCHITECTURES,
)
from global_models import EVENT_LOGITS_FIELD, MAGNITUDE_PRED_FIELD, DIRECTION_LOGITS_FIELD
from global_data import (
    ContextScale,
    ContextScaler,
    GlobalBalancedSampler,
    GlobalWindowDataset,
    MASE_SCALE_COLUMN,
    robust_mase_scale,
)
from global_models import GlobalForecastModel, build_global_model
from global_surface_config import ModalityEncoderHPOSpace


SUPPORTED_GLOBAL_LOSSES: Final[Tuple[str, ...]] = (
    "rmse", "mae", "mse", "smape", "wmape", "log_cosh", "huber"
)
SUPPORTED_SELECTION_METRICS: Final[Tuple[str, ...]] = (
    "robust_macro_mase",
    "macro_mae",
    "macro_rmse",
    "micro_mae",
    "raw_macro_mae",
    "raw_macro_rmse",
    "raw_macro_wmape",
    "raw_macro_smape",
)
DEFAULT_OBJECTIVE_METRIC: Final[str] = "robust_macro_mase"


class NonFiniteValidationError(ValueError):
    """Raised when a validation partition cannot produce a finite objective."""


@dataclass(frozen=True)
class GlobalTrainingConfig:
    """Configuración de un entrenamiento global reproducible."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "huber"
    huber_delta: float = 1.0
    patience: int = 8
    min_delta: float = 1e-5
    grad_clip_norm: float | None = 1.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    samples_per_epoch: int | None = None
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    selection_metric: str = DEFAULT_OBJECTIVE_METRIC
    nonfinite_max_retries: int = 3
    nonfinite_lr_factor: float = 0.2
    use_auxiliary_autoencoder: bool = True
    use_modality_specific_encoders: bool = False
    hpo_modality_encoder_architecture: bool = True
    target_encoder_dim: int = 32
    historical_encoder_dim: int = 32
    future_encoder_dim: int = 32
    static_encoder_dim: int = 16
    fusion_hidden_size: int = 64
    target_encoder_num_layers: int = 1
    historical_encoder_num_layers: int = 1
    future_encoder_num_layers: int = 1
    static_encoder_num_layers: int = 1
    fusion_num_layers: int = 1
    modality_encoder_dropout_rate: float = 0.0
    modality_encoder_activation: str = "gelu"
    use_local_residual_decoder: bool = False
    local_residual_autoregressive: bool = True
    local_residual_lambda: float = 0.01
    global_aux_alpha: float = 0.2
    local_residual_hidden_size: int = 32
    local_residual_num_layers: int = 1
    local_residual_dropout_rate: float = 0.0
    use_event_head: bool = False
    event_loss_weight: float = 0.1
    use_magnitude_head: bool = False
    magnitude_loss_weight: float = 0.1
    use_direction_head: bool = False
    direction_loss_weight: float = 0.05
    use_auxiliary_loss_block: bool = True
    auxiliary_loss_weight: float = 0.20
    event_loss_share: float = 0.40
    magnitude_loss_share: float = 0.40
    direction_loss_share: float = 0.20
    hpo_auxiliary_loss_weights: bool = False
    auxiliary_head_hidden_size: int = 32
    auxiliary_head_num_layers: int = 1
    auxiliary_head_dropout_rate: float = 0.0
    event_threshold: float = 1.0
    magnitude_transform: str = "asinh"

    def validate(self) -> None:
        _positive_int(self.epochs, "epochs")
        _positive_int(self.batch_size, "batch_size")
        _positive_float(self.learning_rate, "learning_rate")
        _non_negative_float(self.weight_decay, "weight_decay")
        if self.loss not in SUPPORTED_GLOBAL_LOSSES:
            raise ValueError(
                f"Unsupported loss={self.loss!r}; expected {SUPPORTED_GLOBAL_LOSSES}"
            )
        _positive_float(self.huber_delta, "huber_delta")
        _positive_int(self.patience, "patience")
        _non_negative_float(self.min_delta, "min_delta")
        if self.grad_clip_norm is not None:
            _positive_float(self.grad_clip_norm, "grad_clip_norm")
        _positive_int(self.scheduler_patience, "scheduler_patience")
        if not 0.0 < float(self.scheduler_factor) < 1.0:
            raise ValueError("scheduler_factor must be in the open interval (0, 1)")
        _positive_float(self.min_learning_rate, "min_learning_rate")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.samples_per_epoch is not None:
            _positive_int(self.samples_per_epoch, "samples_per_epoch")
        _non_negative_int(self.num_workers, "num_workers")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not str(self.device).strip():
            raise ValueError("device must not be empty")
        if self.selection_metric not in SUPPORTED_SELECTION_METRICS:
            raise ValueError(
                f"Unsupported selection_metric={self.selection_metric!r}; "
                f"expected {SUPPORTED_SELECTION_METRICS}"
            )
        _non_negative_int(self.nonfinite_max_retries, "nonfinite_max_retries")
        if not 0.0 < float(self.nonfinite_lr_factor) < 1.0:
            raise ValueError("nonfinite_lr_factor must be in the open interval (0, 1)")
        if not isinstance(self.use_auxiliary_autoencoder, bool):
            raise TypeError("use_auxiliary_autoencoder must be a boolean")
        if not isinstance(self.use_modality_specific_encoders, bool):
            raise TypeError("use_modality_specific_encoders must be a boolean")
        if not isinstance(self.hpo_modality_encoder_architecture, bool):
            raise TypeError("hpo_modality_encoder_architecture must be a boolean")
        for name in (
            "target_encoder_dim",
            "historical_encoder_dim",
            "future_encoder_dim",
            "static_encoder_dim",
            "fusion_hidden_size",
            "target_encoder_num_layers",
            "historical_encoder_num_layers",
            "future_encoder_num_layers",
            "static_encoder_num_layers",
            "fusion_num_layers",
        ):
            _positive_int(getattr(self, name), name)
        if not 0.0 <= float(self.modality_encoder_dropout_rate) < 1.0:
            raise ValueError("modality_encoder_dropout_rate must be in [0, 1)")
        if str(self.modality_encoder_activation).strip().lower() not in {
            "relu", "gelu", "mish", "silu", "tanh"
        }:
            raise ValueError("Unsupported modality_encoder_activation")
        if not isinstance(self.use_local_residual_decoder, bool):
            raise TypeError("use_local_residual_decoder must be a boolean")
        if not isinstance(self.local_residual_autoregressive, bool):
            raise TypeError("local_residual_autoregressive must be a boolean")
        _non_negative_float(self.local_residual_lambda, "local_residual_lambda")
        _non_negative_float(self.global_aux_alpha, "global_aux_alpha")
        _positive_int(self.local_residual_hidden_size, "local_residual_hidden_size")
        _positive_int(self.local_residual_num_layers, "local_residual_num_layers")
        if not 0.0 <= float(self.local_residual_dropout_rate) < 1.0:
            raise ValueError("local_residual_dropout_rate must be in [0, 1)")
        for name in ("use_event_head", "use_magnitude_head", "use_direction_head"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        _non_negative_float(self.event_loss_weight, "event_loss_weight")
        _non_negative_float(self.magnitude_loss_weight, "magnitude_loss_weight")
        _non_negative_float(self.direction_loss_weight, "direction_loss_weight")
        if not isinstance(self.use_auxiliary_loss_block, bool):
            raise TypeError("use_auxiliary_loss_block must be a boolean")
        _non_negative_float(self.auxiliary_loss_weight, "auxiliary_loss_weight")
        _non_negative_float(self.event_loss_share, "event_loss_share")
        _non_negative_float(self.magnitude_loss_share, "magnitude_loss_share")
        _non_negative_float(self.direction_loss_share, "direction_loss_share")
        share_sum = (
            float(self.event_loss_share)
            + float(self.magnitude_loss_share)
            + float(self.direction_loss_share)
        )
        if self.use_auxiliary_loss_block and not math.isclose(share_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "event_loss_share + magnitude_loss_share + direction_loss_share must equal 1.0 "
                "when use_auxiliary_loss_block=True"
            )
        if not isinstance(self.hpo_auxiliary_loss_weights, bool):
            raise TypeError("hpo_auxiliary_loss_weights must be a boolean")
        _positive_int(self.auxiliary_head_hidden_size, "auxiliary_head_hidden_size")
        _positive_int(self.auxiliary_head_num_layers, "auxiliary_head_num_layers")
        if not 0.0 <= float(self.auxiliary_head_dropout_rate) < 1.0:
            raise ValueError("auxiliary_head_dropout_rate must be in [0, 1)")
        _positive_float(self.event_threshold, "event_threshold")
        if str(self.magnitude_transform).strip().lower() not in {"asinh", "log1p", "abs", "none", "identity"}:
            raise ValueError("magnitude_transform must be one of: asinh, log1p, abs")


@dataclass(frozen=True)
class GlobalHPOConfig:
    """Presupuesto HPO y espacio arquitectónico explícito.

    ``modality_encoder_hpo_space`` es un mapping serializable validado por
    :class:`ModalityEncoderHPOSpace`. De este modo, los defaults de la red y el
    espacio realmente explorado dejan de confundirse.
    """

    epochs: int = 3
    windows_per_series_per_epoch: int = 4
    validation_windows_per_series: int = 3
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    min_resource: int = 1
    reduction_factor: int = 3
    finalists: int = 5
    fidelity_epochs: int = 8
    fidelity_windows_per_series_per_epoch: int = 8
    modality_encoder_hpo_space: Mapping[str, Any] = field(
        default_factory=lambda: ModalityEncoderHPOSpace().model_dump(mode="python")
    )

    def modality_encoder_space(self) -> ModalityEncoderHPOSpace:
        return ModalityEncoderHPOSpace.model_validate(
            dict(self.modality_encoder_hpo_space)
        )

    def validate(self) -> None:
        _positive_int(self.epochs, "hpo epochs")
        _positive_int(
            self.windows_per_series_per_epoch,
            "windows_per_series_per_epoch",
        )
        _positive_int(
            self.validation_windows_per_series,
            "validation_windows_per_series",
        )
        if self.objective_metric not in SUPPORTED_SELECTION_METRICS:
            raise ValueError(
                f"Unsupported objective_metric={self.objective_metric!r}; "
                f"expected {SUPPORTED_SELECTION_METRICS}"
            )
        _positive_int(self.min_resource, "min_resource")
        _positive_int(self.reduction_factor, "reduction_factor")
        if self.min_resource > self.epochs:
            raise ValueError("min_resource cannot exceed HPO epochs")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor must be at least 2")
        _positive_int(self.finalists, "finalists")
        _positive_int(self.fidelity_epochs, "fidelity_epochs")
        _positive_int(
            self.fidelity_windows_per_series_per_epoch,
            "fidelity_windows_per_series_per_epoch",
        )
        self.modality_encoder_space()


@dataclass(frozen=True)
class GlobalDatasetBundle:
    """Datasets leak-free usados por entrenamiento y HPO.

    ``validation_seen`` debe contener identidades presentes en ``train`` pero
    ventanas temporalmente posteriores. ``validation_unseen`` debe contener
    identidades completamente disjuntas. La construcción temporal se mantiene
    fuera de este checkpoint y será orquestada posteriormente.
    """

    train: GlobalWindowDataset
    validation_seen: GlobalWindowDataset
    validation_unseen: GlobalWindowDataset

    def validate(self) -> None:
        datasets = (self.train, self.validation_seen, self.validation_unseen)
        if not all(isinstance(dataset, GlobalWindowDataset) for dataset in datasets):
            raise TypeError("All bundle members must be GlobalWindowDataset instances")

        dimensions = {
            (
                dataset.window_size,
                dataset.horizon,
                tuple(dataset.exogenous_columns),
                tuple(dataset.static_feature_names),
            )
            for dataset in datasets
        }
        if len(dimensions) != 1:
            raise ValueError(
                "train and validation datasets must share window, horizon, exogenous and static features"
            )

        train_ids = set(self.train.series_ids)
        seen_ids = set(self.validation_seen.series_ids)
        unseen_ids = set(self.validation_unseen.series_ids)
        if not seen_ids.issubset(train_ids):
            raise ValueError("validation_seen identities must be present in train")
        if train_ids & unseen_ids:
            raise ValueError("validation_unseen identities must be disjoint from train")
        if not unseen_ids:
            raise ValueError("validation_unseen must contain at least one identity")

    @property
    def window_size(self) -> int:
        return self.train.window_size

    @property
    def horizon(self) -> int:
        return self.train.horizon

    @property
    def exogenous_dim(self) -> int:
        return len(self.train.exogenous_columns)

    @property
    def static_dim(self) -> int:
        return self.train.static_dim

    @property
    def static_feature_names(self) -> Tuple[str, ...]:
        return self.train.static_feature_names

    @property
    def validation_datasets(self) -> Mapping[str, GlobalWindowDataset]:
        return {
            "validation_seen": self.validation_seen,
            "validation_unseen": self.validation_unseen,
        }


@dataclass(frozen=True)
class GlobalCandidateConfig:
    """Configuración completa sugerida por un trial global."""

    window_size: int
    model_config: Mapping[str, Any]
    training_config: GlobalTrainingConfig

    def validate(self) -> None:
        _positive_int(self.window_size, "window_size")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        self.training_config.validate()

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "window_size": self.window_size,
            "model_config": dict(self.model_config),
            "training_config": asdict(self.training_config),
        }


@dataclass(frozen=True)
class GlobalValidationMetrics:
    """Métricas agregadas sin permitir que una serie larga domine."""

    macro_mae: float
    macro_rmse: float
    micro_mae: float
    raw_macro_mae: float
    raw_macro_rmse: float
    raw_macro_wmape: float
    raw_macro_smape: float
    num_series: int
    num_points: int
    per_series: Mapping[str, Mapping[str, float]]
    robust_macro_mase: float = math.nan
    num_clipped_predictions: int = 0
    num_nonfinite_predictions: int = 0

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalEpochRecord:
    epoch: int
    train_loss: float
    validation_objective: float
    learning_rate: float
    validation: Mapping[str, GlobalValidationMetrics]


@dataclass
class GlobalTrainingResult:
    """Único modelo compartido y evidencia de su entrenamiento."""

    architecture: str
    model: GlobalForecastModel
    model_config: Mapping[str, Any]
    training_config: GlobalTrainingConfig
    history: Tuple[GlobalEpochRecord, ...]
    best_epoch: int
    best_score: float
    validation: Mapping[str, GlobalValidationMetrics]
    stopped_early: bool


@dataclass
class GlobalHPOResult:
    """Estudio proxy y selección final entre candidatos a fidelidad media."""

    architecture: str
    study: optuna.Study
    best_candidate: GlobalCandidateConfig
    training: GlobalTrainingResult
    selected_trial_number: int | None = None
    fidelity_scores: Mapping[int, float] | None = None


EpochCallback = Callable[[GlobalEpochRecord], None]
DatasetFactory = Callable[[int], GlobalDatasetBundle]
CandidateFactory = Callable[[Trial, str, GlobalTrainingConfig], GlobalCandidateConfig]


class GlobalTrainer:
    """Entrena un único modelo cuyos pesos se comparten entre todas las series."""

    def __init__(
        self,
        architecture: str,
        model_config: Mapping[str, Any],
        training_config: GlobalTrainingConfig | None = None,
    ) -> None:
        self.architecture = _normalize_architecture(architecture)
        self.model_config = dict(model_config)
        self.training_config = training_config or GlobalTrainingConfig()
        self.training_config.validate()

    def fit(
        self,
        datasets: GlobalDatasetBundle,
        *,
        epoch_callback: EpochCallback | None = None,
        validation_windows_per_series: int | None = None,
        objective_metric: str | None = None,
    ) -> GlobalTrainingResult:
        datasets.validate()
        config = self.training_config
        selected_objective = objective_metric or config.selection_metric
        _seed_everything(config.seed)
        device = _resolve_device(config.device)

        model = build_global_model(
            self.architecture,
            self.model_config,
            window_size=datasets.window_size,
            horizon=datasets.horizon,
            exogenous_dim=datasets.exogenous_dim,
            static_dim=datasets.static_dim,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
        )

        sampler = GlobalBalancedSampler(
            datasets.train,
            num_samples=config.samples_per_epoch,
            seed=config.seed,
        )
        train_loader = _make_loader(
            datasets.train,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            device=device,
        )
        validation_loaders = {
            name: _make_loader(
                _validation_subset(
                    dataset,
                    windows_per_series=validation_windows_per_series,
                ),
                batch_size=config.batch_size,
                sampler=None,
                num_workers=config.num_workers,
                device=device,
            )
            for name, dataset in datasets.validation_datasets.items()
        }

        best_score = math.inf
        best_epoch = 0
        best_state: Dict[str, torch.Tensor] | None = None
        best_validation: Mapping[str, GlobalValidationMetrics] = {}
        epochs_without_improvement = 0
        history: list[GlobalEpochRecord] = []

        for epoch in range(1, config.epochs + 1):
            sampler.set_epoch(epoch - 1)
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                config,
                device,
            )
            validation = {
                name: evaluate_global_model(model, loader, device=device)
                for name, loader in validation_loaders.items()
            }
            objective = validation_objective(validation, metric=selected_objective)
            scheduler.step(objective)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            record = GlobalEpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_objective=objective,
                learning_rate=learning_rate,
                validation=validation,
            )
            history.append(record)

            if epoch_callback is not None:
                epoch_callback(record)

            if objective < best_score - config.min_delta:
                best_score = objective
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_validation = deepcopy(validation)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                break

        if best_state is None:
            raise RuntimeError("Training finished without a finite validation checkpoint")
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        return GlobalTrainingResult(
            architecture=self.architecture,
            model=model,
            model_config=dict(self.model_config),
            training_config=config,
            history=tuple(history),
            best_epoch=best_epoch,
            best_score=float(best_score),
            validation=best_validation,
            stopped_early=len(history) < config.epochs,
        )


class GlobalHPOTrainer:
    """HPO en dos fidelidades: screening con pruning y comparación de finalistas."""

    def __init__(
        self,
        architecture: str,
        *,
        base_training_config: GlobalTrainingConfig | None = None,
        hpo_config: GlobalHPOConfig | None = None,
        candidate_factory: CandidateFactory | None = None,
        seed: int = 42,
    ) -> None:
        self.architecture = _normalize_architecture(architecture)
        self.base_training_config = base_training_config or GlobalTrainingConfig(seed=seed)
        self.base_training_config.validate()
        self.hpo_config = hpo_config or GlobalHPOConfig(
            epochs=min(3, self.base_training_config.epochs)
        )
        self.hpo_config.validate()
        if self.hpo_config.objective_metric != self.base_training_config.selection_metric:
            raise ValueError(
                "HPO objective_metric and productive selection_metric must be identical"
            )
        if candidate_factory is None:
            self.candidate_factory = lambda trial, architecture, base_config: (
                suggest_global_candidate(
                    trial,
                    architecture,
                    base_config,
                    hpo_config=self.hpo_config,
                )
            )
        else:
            self.candidate_factory = candidate_factory
        self.seed = int(seed)

    def search_and_fit(
        self,
        dataset_factory: DatasetFactory,
        *,
        n_trials: int,
        timeout: float | None = None,
        study_name: str | None = None,
        storage: str | None = None,
        load_if_exists: bool = False,
    ) -> GlobalHPOResult:
        """Busca hiperparámetros y reevalúa los mejores trials a fidelidad media.

        Los pesos de HPO siguen siendo evidencia diagnóstica. El entrenamiento
        productivo empieza después con un modelo nuevo dentro del currículo global.
        """

        _positive_int(n_trials, "n_trials")
        if timeout is not None:
            _positive_float(timeout, "timeout")
        if not callable(dataset_factory):
            raise TypeError("dataset_factory must be callable")

        hpo = self.hpo_config
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=hpo.min_resource,
                max_resource=hpo.epochs,
                reduction_factor=hpo.reduction_factor,
            ),
            storage=storage,
            load_if_exists=bool(load_if_exists),
        )
        dataset_cache: Dict[int, GlobalDatasetBundle] = {}
        trial_results: Dict[int, GlobalTrainingResult] = {}

        def cached_datasets(window_size: int) -> GlobalDatasetBundle:
            key = int(window_size)
            if key not in dataset_cache:
                bundle = dataset_factory(key)
                bundle.validate()
                if bundle.window_size != key:
                    raise ValueError(
                        "dataset_factory returned a window_size different from the trial candidate"
                    )
                dataset_cache[key] = bundle
            return dataset_cache[key]

        def objective(trial: Trial) -> float:
            candidate = self.candidate_factory(
                trial,
                self.architecture,
                self.base_training_config,
            )
            candidate.validate()
            datasets = cached_datasets(candidate.window_size)
            proxy_samples = (
                len(datasets.train.series_ids)
                * hpo.windows_per_series_per_epoch
            )
            proxy_config = replace(
                candidate.training_config,
                epochs=hpo.epochs,
                samples_per_epoch=proxy_samples,
                patience=max(hpo.epochs, 1),
            )
            trainer = GlobalTrainer(
                self.architecture,
                candidate.model_config,
                proxy_config,
            )

            def report_epoch(record: GlobalEpochRecord) -> None:
                trial.report(record.validation_objective, step=record.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            result = trainer.fit(
                datasets,
                epoch_callback=report_epoch,
                validation_windows_per_series=hpo.validation_windows_per_series,
                objective_metric=hpo.objective_metric,
            )
            trial_results[trial.number] = result
            trial.set_user_attr("candidate", candidate.to_dict())
            trial.set_user_attr("best_epoch", result.best_epoch)
            trial.set_user_attr("proxy_samples_per_epoch", proxy_samples)
            trial.set_user_attr(
                "validation_windows_per_series",
                hpo.validation_windows_per_series,
            )
            trial.set_user_attr("objective_metric", hpo.objective_metric)
            return result.best_score

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
        )
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and math.isfinite(float(trial.value))
        ]
        if not completed_trials:
            raise RuntimeError("HPO completed without a valid completed trial")

        proxy_finalists = sorted(
            completed_trials, key=lambda item: float(item.value)
        )[: min(int(hpo.finalists), len(completed_trials))]
        fidelity_results: Dict[int, GlobalTrainingResult] = {}
        fidelity_scores: Dict[int, float] = {}
        for frozen_trial in proxy_finalists:
            candidate = _candidate_from_user_attrs(frozen_trial.user_attrs)
            datasets = cached_datasets(candidate.window_size)
            fidelity_samples = (
                len(datasets.train.series_ids)
                * hpo.fidelity_windows_per_series_per_epoch
            )
            fidelity_config = replace(
                candidate.training_config,
                epochs=hpo.fidelity_epochs,
                samples_per_epoch=fidelity_samples,
                patience=max(hpo.fidelity_epochs, 1),
            )
            result = GlobalTrainer(
                self.architecture,
                candidate.model_config,
                fidelity_config,
            ).fit(
                datasets,
                validation_windows_per_series=hpo.validation_windows_per_series,
                objective_metric=hpo.objective_metric,
            )
            fidelity_results[int(frozen_trial.number)] = result
            fidelity_scores[int(frozen_trial.number)] = float(result.best_score)

        selected_trial_number = min(fidelity_scores, key=fidelity_scores.get)
        selected_trial = next(
            trial for trial in proxy_finalists
            if int(trial.number) == int(selected_trial_number)
        )
        best_candidate = _candidate_from_user_attrs(selected_trial.user_attrs)
        best_fidelity_training = fidelity_results[selected_trial_number]
        study.set_user_attr(
            "medium_fidelity_selection",
            {
                "objective_metric": hpo.objective_metric,
                "selected_trial_number": int(selected_trial_number),
                "scores": {str(k): float(v) for k, v in sorted(fidelity_scores.items())},
                "epochs": int(hpo.fidelity_epochs),
                "windows_per_series_per_epoch": int(
                    hpo.fidelity_windows_per_series_per_epoch
                ),
            },
        )

        return GlobalHPOResult(
            architecture=self.architecture,
            study=study,
            best_candidate=best_candidate,
            training=best_fidelity_training,
            selected_trial_number=int(selected_trial_number),
            fidelity_scores=dict(fidelity_scores),
        )


def suggest_global_candidate(
    trial: Trial,
    architecture: str,
    base_training_config: GlobalTrainingConfig,
    *,
    hpo_config: GlobalHPOConfig | None = None,
) -> GlobalCandidateConfig:
    """Sugiere un candidato usando un espacio HPO explícito y validado.

    ``base_training_config`` aporta defaults/fallbacks. ``hpo_config`` define
    qué valores explora Optuna. La firma conserva compatibilidad con llamadas
    directas de tres argumentos mediante los defaults del espacio 22.3.2.
    """

    architecture = _normalize_architecture(architecture)
    effective_hpo_config = hpo_config or GlobalHPOConfig()
    effective_hpo_config.validate()
    modality_space = effective_hpo_config.modality_encoder_space()
    window_size = trial.suggest_categorical("window_size", [5, 10, 15, 20])
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    activation = trial.suggest_categorical("activation", ["gelu", "silu"])

    model_config: Dict[str, Any] = {
        "latent_dim": latent_dim,
        "dropout_rate": dropout_rate,
        "activation": activation,
        "use_auxiliary_autoencoder": base_training_config.use_auxiliary_autoencoder,
        "use_modality_specific_encoders": base_training_config.use_modality_specific_encoders,
        "target_encoder_dim": base_training_config.target_encoder_dim,
        "historical_encoder_dim": base_training_config.historical_encoder_dim,
        "future_encoder_dim": base_training_config.future_encoder_dim,
        "static_encoder_dim": base_training_config.static_encoder_dim,
        "fusion_hidden_size": base_training_config.fusion_hidden_size,
        "target_encoder_num_layers": base_training_config.target_encoder_num_layers,
        "historical_encoder_num_layers": base_training_config.historical_encoder_num_layers,
        "future_encoder_num_layers": base_training_config.future_encoder_num_layers,
        "static_encoder_num_layers": base_training_config.static_encoder_num_layers,
        "fusion_num_layers": base_training_config.fusion_num_layers,
        "modality_encoder_dropout_rate": base_training_config.modality_encoder_dropout_rate,
        "modality_encoder_activation": base_training_config.modality_encoder_activation,
        "use_local_residual_decoder": base_training_config.use_local_residual_decoder,
        "local_residual_autoregressive": base_training_config.local_residual_autoregressive,
        "local_residual_lambda": base_training_config.local_residual_lambda,
        "global_aux_alpha": base_training_config.global_aux_alpha,
        "local_residual_hidden_size": base_training_config.local_residual_hidden_size,
        "local_residual_num_layers": base_training_config.local_residual_num_layers,
        "local_residual_dropout_rate": base_training_config.local_residual_dropout_rate,
        "use_event_head": base_training_config.use_event_head,
        "event_loss_weight": base_training_config.event_loss_weight,
        "use_magnitude_head": base_training_config.use_magnitude_head,
        "magnitude_loss_weight": base_training_config.magnitude_loss_weight,
        "use_direction_head": base_training_config.use_direction_head,
        "direction_loss_weight": base_training_config.direction_loss_weight,
        "use_auxiliary_loss_block": base_training_config.use_auxiliary_loss_block,
        "auxiliary_loss_weight": base_training_config.auxiliary_loss_weight,
        "event_loss_share": base_training_config.event_loss_share,
        "magnitude_loss_share": base_training_config.magnitude_loss_share,
        "direction_loss_share": base_training_config.direction_loss_share,
        "auxiliary_head_hidden_size": base_training_config.auxiliary_head_hidden_size,
        "auxiliary_head_num_layers": base_training_config.auxiliary_head_num_layers,
        "auxiliary_head_dropout_rate": base_training_config.auxiliary_head_dropout_rate,
    }

    if (
        base_training_config.hpo_auxiliary_loss_weights
        and base_training_config.use_auxiliary_loss_block
        and any(
            (
                base_training_config.use_event_head,
                base_training_config.use_magnitude_head,
                base_training_config.use_direction_head,
            )
        )
    ):
        auxiliary_loss_weight = trial.suggest_categorical(
            "auxiliary_loss_weight", [0.0, 0.05, 0.10, 0.20, 0.30]
        )
        event_raw = trial.suggest_float("event_loss_share_raw", 0.05, 1.0)
        magnitude_raw = trial.suggest_float("magnitude_loss_share_raw", 0.05, 1.0)
        direction_raw = trial.suggest_float("direction_loss_share_raw", 0.05, 1.0)
        raw_total = max(event_raw + magnitude_raw + direction_raw, 1e-12)
        event_share = event_raw / raw_total
        magnitude_share = magnitude_raw / raw_total
        direction_share = direction_raw / raw_total
        model_config.update(
            {
                "auxiliary_loss_weight": auxiliary_loss_weight,
                "event_loss_share": event_share,
                "magnitude_loss_share": magnitude_share,
                "direction_loss_share": direction_share,
                "event_loss_weight": auxiliary_loss_weight * event_share,
                "magnitude_loss_weight": auxiliary_loss_weight * magnitude_share,
                "direction_loss_weight": auxiliary_loss_weight * direction_share,
            }
        )

    if (
        base_training_config.use_modality_specific_encoders
        and base_training_config.hpo_modality_encoder_architecture
        and modality_space.enabled
    ):
        shared_dimensions = sorted(
            set(modality_space.target_dim_choices)
            & set(modality_space.historical_dim_choices)
            & set(modality_space.future_dim_choices)
        )
        can_couple = bool(
            modality_space.couple_temporal_encoders and shared_dimensions
        )
        if can_couple:
            temporal_dim = trial.suggest_categorical(
                "temporal_encoder_dim", shared_dimensions
            )
            layer_min = max(
                int(modality_space.target_layers.minimum),
                int(modality_space.historical_layers.minimum),
                int(modality_space.future_layers.minimum),
            )
            layer_max = min(
                int(modality_space.target_layers.maximum),
                int(modality_space.historical_layers.maximum),
                int(modality_space.future_layers.maximum),
            )
            can_couple = layer_min <= layer_max
        if can_couple:
            temporal_layers = trial.suggest_int(
                "temporal_encoder_num_layers", layer_min, layer_max
            )
            temporal_payload = {
                "target_encoder_dim": temporal_dim,
                "historical_encoder_dim": temporal_dim,
                "future_encoder_dim": temporal_dim,
                "target_encoder_num_layers": temporal_layers,
                "historical_encoder_num_layers": temporal_layers,
                "future_encoder_num_layers": temporal_layers,
            }
        else:
            temporal_payload = {
                "target_encoder_dim": trial.suggest_categorical(
                    "target_encoder_dim", list(modality_space.target_dim_choices)
                ),
                "historical_encoder_dim": trial.suggest_categorical(
                    "historical_encoder_dim", list(modality_space.historical_dim_choices)
                ),
                "future_encoder_dim": trial.suggest_categorical(
                    "future_encoder_dim", list(modality_space.future_dim_choices)
                ),
                "target_encoder_num_layers": trial.suggest_int(
                    "target_encoder_num_layers", int(modality_space.target_layers.minimum), int(modality_space.target_layers.maximum)
                ),
                "historical_encoder_num_layers": trial.suggest_int(
                    "historical_encoder_num_layers", int(modality_space.historical_layers.minimum), int(modality_space.historical_layers.maximum)
                ),
                "future_encoder_num_layers": trial.suggest_int(
                    "future_encoder_num_layers", int(modality_space.future_layers.minimum), int(modality_space.future_layers.maximum)
                ),
            }
        model_config.update(
            {
                **temporal_payload,
                "static_encoder_dim": trial.suggest_categorical(
                    "static_encoder_dim", list(modality_space.static_dim_choices)
                ),
                "fusion_hidden_size": trial.suggest_categorical(
                    "fusion_hidden_size",
                    list(modality_space.fusion_hidden_size_choices),
                ),
                "static_encoder_num_layers": trial.suggest_int(
                    "static_encoder_num_layers",
                    int(modality_space.static_layers.minimum),
                    int(modality_space.static_layers.maximum),
                ),
                "fusion_num_layers": trial.suggest_int(
                    "fusion_num_layers",
                    int(modality_space.fusion_layers.minimum),
                    int(modality_space.fusion_layers.maximum),
                ),
                "modality_encoder_dropout_rate": trial.suggest_float(
                    "modality_encoder_dropout_rate",
                    float(modality_space.dropout.minimum),
                    float(modality_space.dropout.maximum),
                ),
                "modality_encoder_activation": trial.suggest_categorical(
                    "modality_encoder_activation",
                    list(modality_space.activations),
                ),
            }
        )

    if architecture in {"mlp", "mlp_vae"}:
        shared_hidden_size = trial.suggest_categorical(
            "mlp_hidden_size", [64, 128]
        )
        shared_num_layers = trial.suggest_int("mlp_num_layers", 1, 2)
        model_config.update(
            {
                "enc_hidden_size": shared_hidden_size,
                "enc_num_layers": shared_num_layers,
                "dec_hidden_size": shared_hidden_size,
                "dec_num_layers": shared_num_layers,
            }
        )
        if architecture == "mlp_vae":
            model_config["beta_kl"] = trial.suggest_float("beta_kl", 1e-4, 1.0, log=True)
    else:
        model_config.update(
            {
                "rnn_hidden_size": trial.suggest_categorical(
                    "rnn_hidden_size", [64, 128]
                ),
                "rnn_num_layers": trial.suggest_int("rnn_num_layers", 1, 2),
                "decoder_num_layers": 1,
                "rnn_activation": activation,
            }
        )

    training_config = replace(
        base_training_config,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
    )
    return GlobalCandidateConfig(
        window_size=window_size,
        model_config=model_config,
        training_config=training_config,
    )


def _point_forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss: str,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute the base point loss for one forecast tensor."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match: {prediction.shape} != {target.shape}"
        )
    error = prediction - target
    normalized_loss = str(loss).strip().lower()
    if normalized_loss == "mae":
        return torch.mean(torch.abs(error))
    if normalized_loss == "mse":
        return torch.mean(torch.square(error))
    if normalized_loss == "rmse":
        return torch.sqrt(torch.mean(torch.square(error)) + 1e-12)
    if normalized_loss == "smape":
        denominator = torch.clamp(prediction.abs() + target.abs(), min=1e-6)
        return torch.mean(200.0 * error.abs() / denominator)
    if normalized_loss == "wmape":
        return error.abs().sum() / torch.clamp(target.abs().sum(), min=1e-6)
    if normalized_loss == "log_cosh":
        absolute = error.abs()
        return torch.mean(absolute + nn.functional.softplus(-2.0 * absolute) - math.log(2.0))
    if normalized_loss == "huber":
        return nn.functional.huber_loss(
            prediction, target, reduction="mean", delta=float(huber_delta)
        )
    raise ValueError(f"Unsupported loss={loss!r}; expected {SUPPORTED_GLOBAL_LOSSES}")



def _auxiliary_effective_loss_weight(extras: Mapping[str, Any], head: str) -> float:
    """Return the effective auxiliary loss weight for one agnostic head.

    Checkpoint 22.2.1 introduces a normalized auxiliary block:

    loss_aux = auxiliary_loss_weight * (
        event_loss_share * event_loss
        + magnitude_loss_share * magnitude_loss
        + direction_loss_share * direction_loss
    )

    For backwards compatibility, setting use_auxiliary_loss_block=False falls back
    to the direct per-head weights from Checkpoint 22.2.
    """

    if bool(extras.get("use_auxiliary_loss_block", False)):
        auxiliary_weight = float(extras.get("auxiliary_loss_weight", 0.0))
        if auxiliary_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative")
        share_name = f"{head}_loss_share"
        share = float(extras.get(share_name, 0.0))
        if share < 0.0:
            raise ValueError(f"{share_name} must be non-negative")
        return auxiliary_weight * share
    legacy_name = f"{head}_loss_weight"
    weight = float(extras.get(legacy_name, 0.0))
    if weight < 0.0:
        raise ValueError(f"{legacy_name} must be non-negative")
    return weight


def global_forecast_loss(
    output: Mapping[str, Any],
    target: torch.Tensor,
    *,
    loss: str,
    huber_delta: float = 1.0,
    auxiliary_targets: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Multi-horizon forecast loss plus Stage 2 regularizers.

    Stage 2 keeps ``y_pred`` as the final forecast while optionally adding:

    - L1 regularization on ``delta_local`` so the residual decoder remains a
      correction instead of replacing the global decoder;
    - an auxiliary point loss on ``y_global`` so the global head stays useful.
    """

    prediction = output.get("y_pred")
    if not isinstance(prediction, torch.Tensor):
        raise KeyError("Model output must contain 'y_pred'")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    base = _point_forecast_loss(
        prediction,
        target,
        loss=loss,
        huber_delta=huber_delta,
    )

    auxiliary = output.get("losses", {}) or {}
    if not isinstance(auxiliary, Mapping):
        raise TypeError("output['losses'] must be a mapping when provided")
    for name in ("weighted_kl", "weighted_reconstruction"):
        value = auxiliary.get(name)
        if value is not None:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise TypeError(f"{name} must be a scalar tensor")
            base = base + value

    extras = output.get("extras", {}) or {}
    if not isinstance(extras, Mapping):
        raise TypeError("output['extras'] must be a mapping when provided")

    delta_local = extras.get(LOCAL_RESIDUAL_FIELD)
    if delta_local is not None:
        if not isinstance(delta_local, torch.Tensor):
            raise TypeError("delta_local must be a tensor when provided")
        if delta_local.shape != prediction.shape:
            raise ValueError("delta_local shape must match y_pred")
        local_lambda = float(extras.get("local_residual_lambda", 0.0))
        if local_lambda < 0.0:
            raise ValueError("local_residual_lambda must be non-negative")
        base = base + local_lambda * delta_local.abs().mean()

    y_global = extras.get(GLOBAL_COMPONENT_FIELD)
    if y_global is not None:
        if not isinstance(y_global, torch.Tensor):
            raise TypeError("y_global must be a tensor when provided")
        if y_global.shape != prediction.shape:
            raise ValueError("y_global shape must match y_pred")
        global_aux_alpha = float(extras.get("global_aux_alpha", 0.0))
        if global_aux_alpha < 0.0:
            raise ValueError("global_aux_alpha must be non-negative")
        if global_aux_alpha > 0.0:
            base = base + global_aux_alpha * _point_forecast_loss(
                y_global,
                target,
                loss=loss,
                huber_delta=huber_delta,
            )

    event_logits = extras.get(EVENT_LOGITS_FIELD)
    if event_logits is not None:
        if auxiliary_targets is None or "event_target" not in auxiliary_targets:
            raise KeyError("event_head requires targets['event_target']")
        event_target = auxiliary_targets["event_target"].to(event_logits.device)
        if event_logits.shape != event_target.shape:
            raise ValueError("event_logits shape must match event_target")
        weight = _auxiliary_effective_loss_weight(extras, "event")
        if weight > 0.0:
            base = base + weight * nn.functional.binary_cross_entropy_with_logits(
                event_logits,
                event_target.to(dtype=event_logits.dtype),
            )

    magnitude_pred = extras.get(MAGNITUDE_PRED_FIELD)
    if magnitude_pred is not None:
        if auxiliary_targets is None or "magnitude_target" not in auxiliary_targets:
            raise KeyError("magnitude_head requires targets['magnitude_target']")
        magnitude_target = auxiliary_targets["magnitude_target"].to(magnitude_pred.device)
        if magnitude_pred.shape != magnitude_target.shape:
            raise ValueError("magnitude_pred shape must match magnitude_target")
        weight = _auxiliary_effective_loss_weight(extras, "magnitude")
        if weight > 0.0:
            base = base + weight * nn.functional.huber_loss(
                magnitude_pred,
                magnitude_target.to(dtype=magnitude_pred.dtype),
                delta=float(huber_delta),
            )

    direction_logits = extras.get(DIRECTION_LOGITS_FIELD)
    if direction_logits is not None:
        if auxiliary_targets is None or "direction_target" not in auxiliary_targets:
            raise KeyError("direction_head requires targets['direction_target']")
        direction_target = auxiliary_targets["direction_target"].to(direction_logits.device)
        if direction_logits.ndim != 3 or direction_logits.shape[-1] != 3:
            raise ValueError("direction_logits must have shape [batch, horizon, 3]")
        if tuple(direction_logits.shape[:2]) != tuple(direction_target.shape):
            raise ValueError("direction_logits horizon shape must match direction_target")
        weight = _auxiliary_effective_loss_weight(extras, "direction")
        if weight > 0.0:
            base = base + weight * nn.functional.cross_entropy(
                direction_logits.reshape(-1, 3),
                direction_target.reshape(-1).long(),
            )
    return base

def evaluate_global_model(
    model: GlobalForecastModel,
    loader: DataLoader,
    *,
    device: torch.device | str,
) -> GlobalValidationMetrics:
    """Evalúa por serie y macro-promedia con inversión numéricamente segura."""

    resolved_device = torch.device(device)
    model.eval()
    accumulators: MutableMapping[str, Dict[str, float]] = defaultdict(
        lambda: {
            "abs": 0.0,
            "sq": 0.0,
            "raw_abs": 0.0,
            "raw_sq": 0.0,
            "raw_den": 0.0,
            "raw_smape_sum": 0.0,
            "raw_mase_sum": 0.0,
            "count": 0.0,
        }
    )
    clipped_predictions = 0
    nonfinite_predictions = 0

    with torch.no_grad():
        for batch in loader:
            model_inputs = _move_model_inputs(batch["model_inputs"], resolved_device)
            target = batch["targets"]["y_future"].to(resolved_device)
            target_raw = batch["targets"]["y_future_raw"].to(resolved_device)
            output = model(**model_inputs)
            prediction = output.get("y_pred")
            if not isinstance(prediction, torch.Tensor):
                raise KeyError("Model output must contain 'y_pred'")
            if prediction.shape != target.shape:
                raise ValueError("Model prediction shape does not match y_future")
            if not torch.all(torch.isfinite(prediction)):
                count = int((~torch.isfinite(prediction)).sum().cpu())
                raise NonFiniteValidationError(
                    f"Model produced {count} non-finite normalized predictions"
                )

            prediction_np = prediction.detach().cpu().numpy()
            prediction_raw_np = np.empty_like(prediction_np, dtype=np.float64)
            centers = list(batch["metadata"]["center"])
            scales = list(batch["metadata"]["scale"])
            transforms = list(batch["metadata"].get("transform", ["identity"] * int(target.shape[0])))
            for row in range(prediction_np.shape[0]):
                raw_values, diagnostics = ContextScaler.inverse_transform_with_diagnostics(
                    prediction_np[row],
                    ContextScale(
                        center=float(centers[row]),
                        scale=float(scales[row]),
                        transform=str(transforms[row]),
                    ),
                )
                prediction_raw_np[row] = raw_values
                clipped_predictions += int(diagnostics["clipped_values"])
                nonfinite_predictions += int(diagnostics["nonfinite_inputs"])

            prediction_raw = torch.as_tensor(
                prediction_raw_np, dtype=torch.float64, device=resolved_device
            )
            target_raw64 = target_raw.to(dtype=torch.float64)

            series_ids = list(batch["metadata"][CROSS_KEY_COLUMN])
            metadata_mase_scales = batch["metadata"].get(MASE_SCALE_COLUMN)
            if metadata_mase_scales is None:
                # Compatibilidad causal con loaders previos: reconstruir el contexto
                # observado de la propia ventana, nunca usar validation/test future.
                normalized_context = model_inputs["y_context"].detach().cpu().numpy()
                mase_scales = []
                for row in range(normalized_context.shape[0]):
                    raw_context = ContextScaler.inverse_transform(
                        normalized_context[row],
                        ContextScale(
                            center=float(centers[row]),
                            scale=float(scales[row]),
                            transform=str(transforms[row]),
                        ),
                    )
                    mase_scales.append(robust_mase_scale(raw_context))
            else:
                mase_scales = list(metadata_mase_scales)
            for row, series_id in enumerate(series_ids):
                normalized_error = prediction[row] - target[row]
                raw_error = prediction_raw[row] - target_raw64[row]
                state = accumulators[str(series_id)]
                state["abs"] += float(normalized_error.abs().sum().cpu())
                state["sq"] += float(normalized_error.square().sum().cpu())
                state["raw_abs"] += float(raw_error.abs().sum().cpu())
                state["raw_sq"] += float(raw_error.square().sum().cpu())
                state["raw_den"] += float(target_raw64[row].abs().sum().cpu())
                smape_denominator = prediction_raw[row].abs() + target_raw64[row].abs()
                raw_smape = torch.where(
                    smape_denominator > 1e-12,
                    200.0 * raw_error.abs() / smape_denominator,
                    torch.zeros_like(smape_denominator),
                )
                if not torch.all(torch.isfinite(raw_smape)):
                    raise NonFiniteValidationError(
                        f"Stable raw sMAPE still became non-finite for series {series_id!r}"
                    )
                state["raw_smape_sum"] += float(raw_smape.sum().cpu())
                mase_scale = float(mase_scales[row])
                if not math.isfinite(mase_scale) or mase_scale <= 0.0:
                    raise NonFiniteValidationError(
                        f"Invalid causal MASE scale for series {series_id!r}: {mase_scale}"
                    )
                state["raw_mase_sum"] += float(raw_error.abs().sum().cpu()) / mase_scale
                state["count"] += float(target[row].numel())

    if not accumulators:
        raise ValueError("Validation loader produced no batches")

    per_series: Dict[str, Mapping[str, float]] = {}
    total_abs = 0.0
    total_count = 0.0
    for series_id, state in sorted(accumulators.items()):
        count = state["count"]
        metrics = {
            "mae": state["abs"] / count,
            "rmse": math.sqrt(state["sq"] / count),
            "raw_mae": state["raw_abs"] / count,
            "raw_rmse": math.sqrt(state["raw_sq"] / count),
            "raw_wmape": state["raw_abs"] / max(state["raw_den"], 1e-12),
            "raw_smape": state["raw_smape_sum"] / count,
            "robust_mase": state["raw_mase_sum"] / count,
            "num_points": count,
        }
        if not all(math.isfinite(float(value)) for key, value in metrics.items() if key != "num_points"):
            raise NonFiniteValidationError(
                f"Validation metrics became non-finite for series {series_id!r}: {metrics}"
            )
        per_series[series_id] = metrics
        total_abs += state["abs"]
        total_count += count

    report = GlobalValidationMetrics(
        robust_macro_mase=float(
            np.mean([metrics["robust_mase"] for metrics in per_series.values()])
        ),
        macro_mae=float(np.mean([metrics["mae"] for metrics in per_series.values()])),
        macro_rmse=float(np.mean([metrics["rmse"] for metrics in per_series.values()])),
        micro_mae=total_abs / total_count,
        raw_macro_mae=float(np.mean([metrics["raw_mae"] for metrics in per_series.values()])),
        raw_macro_rmse=float(np.mean([metrics["raw_rmse"] for metrics in per_series.values()])),
        raw_macro_wmape=float(np.mean([metrics["raw_wmape"] for metrics in per_series.values()])),
        raw_macro_smape=float(np.mean([metrics["raw_smape"] for metrics in per_series.values()])),
        num_series=len(per_series),
        num_points=int(total_count),
        per_series=per_series,
        num_clipped_predictions=int(clipped_predictions),
        num_nonfinite_predictions=int(nonfinite_predictions),
    )
    return report


def validation_objective(
    validation: Mapping[str, GlobalValidationMetrics],
    *,
    metric: str = DEFAULT_OBJECTIVE_METRIC,
) -> float:
    """Promedia por igual las particiones seen y unseen del HPO."""

    if not validation:
        raise ValueError("validation metrics must not be empty")
    values: list[float] = []
    for partition, report in validation.items():
        if not hasattr(report, metric):
            raise ValueError(f"Unknown validation metric {metric!r} for {partition}")
        value = float(getattr(report, metric))
        if not math.isfinite(value):
            raise NonFiniteValidationError(
                f"Non-finite validation metric for {partition}: {value}"
            )
        values.append(value)
    return float(np.mean(values))


def _assert_finite_model_state(model: GlobalForecastModel, *, context: str) -> None:
    bad = [
        name
        for name, parameter in model.named_parameters()
        if not torch.all(torch.isfinite(parameter.detach()))
    ]
    if bad:
        preview = ", ".join(bad[:5])
        raise FloatingPointError(
            f"Non-finite model parameters after {context}: {preview}"
        )


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
        model_inputs = _move_model_inputs(batch["model_inputs"], device)
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
            raise FloatingPointError("Non-finite global training loss")
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        _assert_finite_model_state(model, context="optimizer.step")

        batch_size = int(target.shape[0])
        weighted_loss += float(loss.detach().cpu()) * batch_size
        observed += batch_size

    if observed == 0:
        raise ValueError("Training loader produced no batches")
    return weighted_loss / observed


def _validation_subset(
    dataset: GlobalWindowDataset,
    *,
    windows_per_series: int | None,
) -> GlobalWindowDataset | Subset:
    """Selecciona los orígenes más recientes por serie para el HPO proxy."""

    if windows_per_series is None:
        return dataset
    _positive_int(windows_per_series, "validation_windows_per_series")
    selected: list[int] = []
    for series_id in dataset.series_ids:
        indices = dataset.indices_by_series[series_id]
        selected.extend(indices[-int(windows_per_series) :])
    if not selected:
        raise ValueError("Validation proxy selection produced no windows")
    return Subset(dataset, selected)


def _make_loader(
    dataset: GlobalWindowDataset,
    *,
    batch_size: int,
    sampler: GlobalBalancedSampler | None,
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


def _move_model_inputs(
    model_inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in model_inputs.items()}


def _candidate_from_user_attrs(user_attrs: Mapping[str, Any]) -> GlobalCandidateConfig:
    payload = user_attrs.get("candidate")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Best trial does not contain a serialized candidate")
    training_payload = payload.get("training_config")
    if not isinstance(training_payload, Mapping):
        raise RuntimeError("Serialized candidate lacks training_config")
    candidate = GlobalCandidateConfig(
        window_size=int(payload["window_size"]),
        model_config=dict(payload["model_config"]),
        training_config=GlobalTrainingConfig(**dict(training_payload)),
    )
    candidate.validate()
    return candidate


def _normalize_architecture(architecture: str) -> str:
    normalized = str(architecture or "").strip().lower()
    if normalized not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture={architecture!r}; expected {SUPPORTED_ARCHITECTURES}"
        )
    return normalized


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _non_negative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive_float(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")


def _non_negative_float(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
