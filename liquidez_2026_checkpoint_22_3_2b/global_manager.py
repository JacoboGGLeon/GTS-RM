"""Orquestación y persistencia del modelo global Financial-GPT.

El manager conecta HPO proxy/medium-fidelity, entrenamiento pooled balanceado,
evaluación, forecast y persistencia de un único ``state_dict`` compartido. Checkpoint 10
extiende la persistencia local con save/load S3 verificable mediante checksums,
marker ``_SUCCESS`` y puntero ``latest.json`` por arquitectura.

Los identificadores contables sólo aparecen en metadata, métricas y salidas. Nunca
se pasan al ``forward`` del modelo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader

from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    SERIES_TYPE_COLUMN,
    SUPPORTED_ARCHITECTURES,
)
from global_curriculum import (
    CurriculumEpochCallback,
    GlobalCurriculumConfig,
    GlobalTrainingScheduleConfig,
    GlobalCurriculumSession,
    GlobalCurriculumTrainingResult,
    fit_best_candidate_with_curriculum,
    state_dict_digest,
)
from global_data import (
    ContextScale,
    ContextScaler,
    GlobalSeriesSplit,
    GlobalWindowDataset,
    StaticFeatureEncoder,
)
from global_models import GlobalForecastModel, build_global_model
from global_s3 import (
    DEFAULT_FINANCIAL_GPT_S3_ROOT,
    build_run_uri,
    download_verified_run,
    resolve_latest_run_uri,
    upload_atomic_run,
    validate_component,
)
from global_monitoring import (
    MCDropoutConfig,
    build_legacy_series_and_outliers,
    build_train_reference_outliers,
    forecast_future_mc,
    mc_dropout_backtest,
    visualise_legacy_contract,
)
from global_training import (
    CandidateFactory,
    DatasetFactory,
    GlobalDatasetBundle,
    GlobalHPOConfig,
    GlobalHPOResult,
    GlobalHPOTrainer,
    GlobalTrainingConfig,
    GlobalValidationMetrics,
    evaluate_global_model,
)
from p0_diagnostics import diagnose_patience, evaluate_auxiliary_heads


ARTIFACT_SCHEMA_VERSION: Final[str] = "1.5"
MANIFEST_FILENAME: Final[str] = "manifest.json"
MODEL_FILENAME: Final[str] = "model_state.pt"
METRICS_FILENAME: Final[str] = "metrics.json"
HISTORY_FILENAME: Final[str] = "history.json"
HPO_FILENAME: Final[str] = "hpo_summary.json"
SPLIT_FILENAME: Final[str] = "split_manifest.json"
FORECAST_COLUMNS: Final[Tuple[str, ...]] = (
    CROSS_KEY_COLUMN,
    ACCOUNT_CURRENCY_ID_COLUMN,
    SERIES_TYPE_COLUMN,
    "cutoff",
    "horizon_step",
    "prediction",
    "actual",
    "prediction_scaled",
    "actual_scaled",
    "center",
    "scale",
)


@dataclass(frozen=True)
class GlobalRunDimensions:
    """Dimensiones necesarias para reconstruir el modelo persistido."""

    window_size: int
    horizon: int
    exogenous_dim: int
    static_dim: int
    exogenous_columns: Tuple[str, ...] = ()
    static_feature_names: Tuple[str, ...] = ()

    @property
    def rollout_chunk_size(self) -> int:
        """Cantidad de pasos emitidos por una llamada ``forward``."""

        return int(self.horizon)

    def validate(self) -> None:
        for name, value in (
            ("window_size", self.window_size),
            ("horizon", self.horizon),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.exogenous_dim, bool) or int(self.exogenous_dim) < 0:
            raise ValueError("exogenous_dim must be a non-negative integer")
        if isinstance(self.static_dim, bool) or int(self.static_dim) <= 0:
            raise ValueError("static_dim must be a positive integer")
        if len(self.exogenous_columns) != int(self.exogenous_dim):
            raise ValueError(
                "exogenous_columns length must equal exogenous_dim: "
                f"{len(self.exogenous_columns)} != {self.exogenous_dim}"
            )
        if len(set(self.exogenous_columns)) != len(self.exogenous_columns):
            raise ValueError("exogenous_columns must not contain duplicates")
        if len(self.static_feature_names) != int(self.static_dim):
            raise ValueError("static_feature_names length must equal static_dim")
        if len(set(self.static_feature_names)) != len(self.static_feature_names):
            raise ValueError("static_feature_names must not contain duplicates")


@dataclass(frozen=True)
class GlobalRunSummary:
    """Resumen serializable del run productivo."""

    architecture: str
    best_score: float
    total_epochs: int
    best_hpo_value: float
    num_hpo_trials: int
    dimensions: GlobalRunDimensions
    state_digest: str

    def to_dict(self) -> Mapping[str, Any]:
        return _jsonable(asdict(self))


class GlobalManager:
    """Orquesta y conserva exactamente un modelo global por instancia.

    ``dataset_factory`` sigue siendo externo para no hardcodear fuentes, S3,
    calendario o reglas de split. El manager consume el contrato de Checkpoint 2
    y conserva únicamente el dataset ganador durante la sesión activa.
    """

    def __init__(
        self,
        architecture: str,
        *,
        base_training_config: GlobalTrainingConfig | None = None,
        hpo_config: GlobalHPOConfig | None = None,
        schedule_config: GlobalTrainingScheduleConfig | None = None,
        curriculum_config: GlobalCurriculumConfig | None = None,
        candidate_factory: CandidateFactory | None = None,
        seed: int = 42,
    ) -> None:
        normalized = str(architecture).strip().lower()
        if normalized not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture={architecture!r}; expected {SUPPORTED_ARCHITECTURES}"
            )
        self.architecture = normalized
        self.base_training_config = base_training_config or GlobalTrainingConfig(seed=seed)
        self.hpo_config = hpo_config or GlobalHPOConfig(
            epochs=min(3, self.base_training_config.epochs)
        )
        if schedule_config is not None and curriculum_config is not None:
            raise ValueError("Provide schedule_config or curriculum_config, not both")
        self.schedule_config = (
            schedule_config
            or curriculum_config
            or GlobalTrainingScheduleConfig()
        )
        # Alias de lectura para cargar artefactos/checkpoints históricos.
        self.curriculum_config = self.schedule_config
        self.base_training_config.validate()
        self.hpo_config.validate()
        self.schedule_config.validate()
        self.candidate_factory = candidate_factory
        self.seed = int(seed)

        self.hpo_result: GlobalHPOResult | None = None
        self.training_result: GlobalCurriculumTrainingResult | None = None
        self.datasets: GlobalDatasetBundle | None = None
        self.dimensions: GlobalRunDimensions | None = None
        self.split_manifest: Mapping[str, Any] = {}
        self.run_metadata: Mapping[str, Any] = {}
        self.loaded_manifest: Mapping[str, Any] | None = None
        self.static_feature_encoder: StaticFeatureEncoder | None = None

        # Contrato analítico compatible con los code_02 locales. Los pesos del
        # modelo siguen siendo únicos y globales; sólo las salidas se separan
        # por cross_key_id.
        self._curriculum_session: GlobalCurriculumSession | None = None
        self._dataset_factory: DatasetFactory | None = None
        self._source_global_long: pl.DataFrame | None = None
        self._source_calendar: pl.DataFrame | None = None
        self._productive_training_results: Mapping[str, Any] = {}
        self._warmup_results: Mapping[str, Any] = {}
        self._finetune_results: Mapping[str, Any] = {}
        self._backtest_results: Mapping[str, Any] = {}
        self._future_results: Mapping[str, pd.DataFrame] = {}
        self._df_forecasts: pd.DataFrame = pd.DataFrame()
        self._df_outliers: pd.DataFrame = pd.DataFrame()
        self._series_status: MutableMapping[str, Mapping[str, Any]] = {}
        self._temporal_alignment_report: pl.DataFrame = pl.DataFrame()

    @property
    def is_fitted(self) -> bool:
        return self.training_result is not None

    @property
    def model(self) -> GlobalForecastModel:
        if self.training_result is None:
            raise RuntimeError("GlobalManager has no fitted or loaded model")
        return self.training_result.model

    @property
    def best_candidate(self) -> Mapping[str, Any]:
        if self.hpo_result is not None:
            return self.hpo_result.best_candidate.to_dict()
        if self.loaded_manifest is not None:
            return dict(self.loaded_manifest.get("best_candidate", {}))
        raise RuntimeError("GlobalManager has no HPO result or loaded candidate")

    def fit_global(
        self,
        dataset_factory: DatasetFactory,
        *,
        n_trials: int,
        timeout: float | None = None,
        study_name: str | None = None,
        hpo_storage: str | None = None,
        hpo_load_if_exists: bool = False,
        split_manifest: GlobalSeriesSplit | Mapping[str, Any] | None = None,
        exogenous_columns: Sequence[str] = (),
        run_metadata: Mapping[str, Any] | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
    ) -> GlobalCurriculumTrainingResult:
        """Ejecuta la ruta configurada; pooled balanceado es el estándar."""

        if self.schedule_config.training_order == "pooled_balanced":
            return self.run_hpo_and_train(
                dataset_factory,
                n_trials=n_trials,
                timeout=timeout,
                study_name=study_name,
                hpo_storage=hpo_storage,
                hpo_load_if_exists=hpo_load_if_exists,
                split_manifest=split_manifest,
                exogenous_columns=exogenous_columns,
                run_metadata=run_metadata,
                curriculum_epoch_callback=curriculum_epoch_callback,
            )

        # Ablations históricas: conservan el recorrido curricular original.
        self.run_hpo_and_warmup(
            dataset_factory,
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name,
            hpo_storage=hpo_storage,
            hpo_load_if_exists=hpo_load_if_exists,
            split_manifest=split_manifest,
            exogenous_columns=exogenous_columns,
            run_metadata=run_metadata,
            curriculum_epoch_callback=curriculum_epoch_callback,
        )
        return self.run_finetune(
            curriculum_epoch_callback=curriculum_epoch_callback,
        )

    def run_hpo_and_train(
        self,
        dataset_factory: DatasetFactory,
        *,
        n_trials: int,
        train_epochs: int | None = None,
        continuation_epochs: int | None = None,
        continuation_lr_factor: float | None = None,
        batch: int | None = None,
        timeout: float | None = None,
        study_name: str | None = None,
        hpo_storage: str | None = None,
        hpo_load_if_exists: bool = False,
        split_manifest: GlobalSeriesSplit | Mapping[str, Any] | None = None,
        exogenous_columns: Sequence[str] = (),
        run_metadata: Mapping[str, Any] | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
        show_progress: bool = True,
    ) -> GlobalCurriculumTrainingResult:
        """HPO de dos fidelidades seguido por entrenamiento pooled productivo.

        El candidato ganador se reinicializa y se entrena con todas las series
        elegibles bajo el sampler balanceado. La continuación opcional conserva
        la misma distribución y reduce únicamente el learning rate.
        """

        if not callable(dataset_factory):
            raise TypeError("dataset_factory must be callable")
        schedule = self.schedule_config
        if schedule.training_order != "pooled_balanced":
            raise ValueError(
                "run_hpo_and_train requires training_order='pooled_balanced'"
            )
        if train_epochs is not None:
            if isinstance(train_epochs, bool) or int(train_epochs) <= 0:
                raise ValueError("train_epochs must be a positive integer")
            schedule = replace(schedule, pooled_train_epochs=int(train_epochs))
        if continuation_epochs is not None:
            if isinstance(continuation_epochs, bool) or int(continuation_epochs) < 0:
                raise ValueError("continuation_epochs must be a non-negative integer")
            schedule = replace(
                schedule,
                pooled_continuation_epochs=int(continuation_epochs),
            )
        if continuation_lr_factor is not None:
            schedule = replace(
                schedule,
                pooled_continuation_lr_factor=float(continuation_lr_factor),
            )
        if batch is not None and (isinstance(batch, bool) or int(batch) <= 0):
            raise ValueError("batch must be a positive integer")
        self.base_training_config.validate()
        schedule.validate()

        hpo = GlobalHPOTrainer(
            self.architecture,
            base_training_config=self.base_training_config,
            hpo_config=self.hpo_config,
            candidate_factory=self.candidate_factory,
            seed=self.seed,
        ).search_and_fit(
            dataset_factory,
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name,
            storage=hpo_storage,
            load_if_exists=hpo_load_if_exists,
        )
        datasets = dataset_factory(hpo.best_candidate.window_size)
        datasets.validate()
        requested_columns = tuple(str(value) for value in exogenous_columns)
        dataset_columns = tuple(datasets.train.exogenous_columns)
        columns = requested_columns or dataset_columns
        if columns != dataset_columns:
            raise ValueError("exogenous_columns must match the exact dataset feature order")
        for name, dataset in datasets.validation_datasets.items():
            if tuple(dataset.exogenous_columns) != columns:
                raise ValueError(f"{name} exogenous feature order differs from train")
        dimensions = GlobalRunDimensions(
            window_size=datasets.window_size,
            horizon=datasets.horizon,
            exogenous_dim=datasets.exogenous_dim,
            static_dim=datasets.static_dim,
            exogenous_columns=columns,
            static_feature_names=datasets.static_feature_names,
        )
        dimensions.validate()

        productive_training_config = hpo.best_candidate.training_config
        if train_epochs is not None:
            productive_training_config = replace(
                productive_training_config,
                epochs=int(train_epochs),
            )
        else:
            productive_training_config = replace(
                productive_training_config,
                epochs=int(schedule.pooled_train_epochs),
            )
        if batch is not None:
            productive_training_config = replace(
                productive_training_config,
                batch_size=int(batch),
            )
        productive_training_config.validate()

        session = GlobalCurriculumSession(
            self.architecture,
            hpo.best_candidate.model_config,
            datasets,
            productive_training_config,
            schedule,
        )

        # La evidencia de HPO se compromete antes del entrenamiento productivo.
        self.hpo_result = hpo
        self.datasets = datasets
        self.dimensions = dimensions
        self.static_feature_encoder = datasets.train.static_feature_encoder
        self.split_manifest = _normalize_split_manifest(split_manifest)
        self.run_metadata = _ensure_mapping(run_metadata)
        self.loaded_manifest = None
        self._curriculum_session = session
        self._dataset_factory = dataset_factory
        self.schedule_config = schedule
        self.curriculum_config = schedule
        self._capture_source_frames(dataset_factory)

        result = session.run_phases(
            ("productive_training", "pooled_continuation"),
            epoch_callback=curriculum_epoch_callback,
            batch_size_override=batch,
        )
        self.training_result = result
        self._productive_training_results = self._phase_result(
            "productive_training",
            include_pooled_continuation=True,
        )
        # Vistas legacy sólo para monitores antiguos; no forman parte del
        # contrato nuevo del notebook.
        self._warmup_results = self._productive_training_results
        self._finetune_results = {}
        self._backtest_results = {}
        self._future_results = {}
        self._df_forecasts = pd.DataFrame()
        self._df_outliers = pd.DataFrame()
        return result

    def run_hpo_and_warmup(
        self,
        dataset_factory: DatasetFactory,
        *,
        n_trials: int,
        max_epochs: int | None = None,
        batch: int | None = None,
        timeout: float | None = None,
        study_name: str | None = None,
        hpo_storage: str | None = None,
        hpo_load_if_exists: bool = False,
        split_manifest: GlobalSeriesSplit | Mapping[str, Any] | None = None,
        exogenous_columns: Sequence[str] = (),
        run_metadata: Mapping[str, Any] | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
        show_progress: bool = True,
    ) -> Mapping[str, Any]:
        """API pública: ejecuta HPO y warm-up sin exponer métodos legacy."""

        return self._warmup_all(
            dataset_factory,
            n_trials=n_trials,
            max_epochs=max_epochs,
            batch=batch,
            timeout=timeout,
            study_name=study_name,
            hpo_storage=hpo_storage,
            hpo_load_if_exists=hpo_load_if_exists,
            split_manifest=split_manifest,
            exogenous_columns=exogenous_columns,
            run_metadata=run_metadata,
            curriculum_epoch_callback=curriculum_epoch_callback,
            show_progress=show_progress,
        )

    def _warmup_all(
        self,
        dataset_factory: DatasetFactory,
        *,
        n_trials: int,
        max_epochs: int | None = None,
        batch: int | None = None,
        timeout: float | None = None,
        study_name: str | None = None,
        hpo_storage: str | None = None,
        hpo_load_if_exists: bool = False,
        split_manifest: GlobalSeriesSplit | Mapping[str, Any] | None = None,
        exogenous_columns: Sequence[str] = (),
        run_metadata: Mapping[str, Any] | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
        show_progress: bool = True,
    ) -> Mapping[str, Any]:
        """HPO global seguido únicamente por la fase de warm-up.

        Se crea una sesión curricular que mantiene modelo y optimizador vivos
        para que ``_finetune_all`` continúe exactamente desde este checkpoint.
        """

        if not callable(dataset_factory):
            raise TypeError("dataset_factory must be callable")
        # HPO and productive warm-up have independent compute budgets.
        # ``base_training_config`` keeps the proxy/HPO batch; ``max_epochs`` and
        # ``batch`` are applied only after Optuna has selected a candidate.
        hpo_base_config = self.base_training_config
        curriculum_config = self.curriculum_config
        if max_epochs is not None:
            if isinstance(max_epochs, bool) or int(max_epochs) <= 0:
                raise ValueError("max_epochs must be a positive integer")
            curriculum_config = replace(curriculum_config, warmup_epochs=int(max_epochs))
        if batch is not None and (isinstance(batch, bool) or int(batch) <= 0):
            raise ValueError("batch must be a positive integer")
        hpo_base_config.validate()
        curriculum_config.validate()

        hpo = GlobalHPOTrainer(
            self.architecture,
            base_training_config=hpo_base_config,
            hpo_config=self.hpo_config,
            candidate_factory=self.candidate_factory,
            seed=self.seed,
        ).search_and_fit(
            dataset_factory,
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name,
            storage=hpo_storage,
            load_if_exists=hpo_load_if_exists,
        )
        datasets = dataset_factory(hpo.best_candidate.window_size)
        datasets.validate()
        requested_columns = tuple(str(value) for value in exogenous_columns)
        dataset_columns = tuple(datasets.train.exogenous_columns)
        columns = requested_columns or dataset_columns
        if columns != dataset_columns:
            raise ValueError("exogenous_columns must match the exact dataset feature order")
        for name, dataset in datasets.validation_datasets.items():
            if tuple(dataset.exogenous_columns) != columns:
                raise ValueError(f"{name} exogenous feature order differs from train")
        dimensions = GlobalRunDimensions(
            window_size=datasets.window_size,
            horizon=datasets.horizon,
            exogenous_dim=datasets.exogenous_dim,
            static_dim=datasets.static_dim,
            exogenous_columns=columns,
            static_feature_names=datasets.static_feature_names,
        )
        dimensions.validate()

        productive_training_config = hpo.best_candidate.training_config
        if max_epochs is not None:
            productive_training_config = replace(
                productive_training_config, epochs=int(max_epochs)
            )
        if batch is not None:
            productive_training_config = replace(
                productive_training_config, batch_size=int(batch)
            )
        productive_training_config.validate()

        session = GlobalCurriculumSession(
            self.architecture,
            hpo.best_candidate.model_config,
            datasets,
            productive_training_config,
            curriculum_config,
        )

        # Commit HPO evidence before productive warm-up. If warm-up fails, the
        # study and best candidate remain inspectable and reusable instead of
        # disappearing with the local stack frame.
        self.hpo_result = hpo
        self.datasets = datasets
        self.dimensions = dimensions
        self.static_feature_encoder = datasets.train.static_feature_encoder
        self.split_manifest = _normalize_split_manifest(split_manifest)
        self.run_metadata = _ensure_mapping(run_metadata)
        self.loaded_manifest = None
        self._curriculum_session = session
        self._dataset_factory = dataset_factory
        self._capture_source_frames(dataset_factory)

        result = session.run_phases(
            ("warmup",),
            epoch_callback=curriculum_epoch_callback,
            batch_size_override=batch,
        )
        self.training_result = result
        self._warmup_results = self._phase_result("warmup")
        self._finetune_results = {}
        self._backtest_results = {}
        self._future_results = {}
        self._df_forecasts = pd.DataFrame()
        self._df_outliers = pd.DataFrame()
        return self._warmup_results

    def run_finetune(
        self,
        *,
        epochs: int | None = None,
        batch: int | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
        show_progress: bool = True,
    ) -> GlobalCurriculumTrainingResult:
        """API pública: continúa fine-tuning y consolidación sin reinicios."""

        return self._finetune_all(
            epochs=epochs,
            batch=batch,
            curriculum_epoch_callback=curriculum_epoch_callback,
            show_progress=show_progress,
        )

    def _finetune_all(
        self,
        *,
        epochs: int | None = None,
        batch: int | None = None,
        curriculum_epoch_callback: CurriculumEpochCallback | None = None,
        show_progress: bool = True,
    ) -> GlobalCurriculumTrainingResult:
        """Continúa fine-tuning y consolidación sin reiniciar pesos u optimizador."""

        session = self._curriculum_session
        if session is None:
            raise RuntimeError("Run run_hpo_and_warmup() before run_finetune()")
        pending_phases = tuple(stage.phase for stage in session.pending_stages)
        if "finetune" in pending_phases:
            session.run_phases(
                ("finetune",),
                epoch_callback=curriculum_epoch_callback,
                epochs_override=epochs,
                batch_size_override=batch,
            )
        if "consolidation" in tuple(stage.phase for stage in session.pending_stages):
            session.run_phases(
                ("consolidation",),
                epoch_callback=curriculum_epoch_callback,
                batch_size_override=batch,
            )
        result = session.result()
        self.training_result = result
        self._finetune_results = self._phase_result("finetune", include_consolidation=True)
        return result

    def backtest_seen(
        self,
        dataset: GlobalWindowDataset | None = None,
        *,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> GlobalValidationMetrics:
        """Evalúa ventanas futuras de identidades observadas durante training."""

        selected = dataset or self._default_dataset("validation_seen")
        return self.evaluate(selected, batch_size=batch_size, device=device)

    def backtest_unseen(
        self,
        dataset: GlobalWindowDataset | None = None,
        *,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> GlobalValidationMetrics:
        """Evalúa identidades completamente excluidas del entrenamiento."""

        selected = dataset or self._default_dataset("validation_unseen")
        return self.evaluate(selected, batch_size=batch_size, device=device)

    def evaluate(
        self,
        dataset: GlobalWindowDataset,
        *,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> GlobalValidationMetrics:
        if not isinstance(dataset, GlobalWindowDataset):
            raise TypeError("dataset must be a GlobalWindowDataset")
        resolved_device = _resolve_inference_device(device)
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self._inference_batch_size,
            shuffle=False,
            num_workers=0,
        )
        model = self.model.to(resolved_device)
        metrics = evaluate_global_model(model, loader, device=resolved_device)
        model.eval()
        return metrics

    def forecast(
        self,
        dataset: GlobalWindowDataset,
        *,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> pl.DataFrame:
        """Genera predicciones normalizadas y en escala original por ventana."""

        if not isinstance(dataset, GlobalWindowDataset):
            raise TypeError("dataset must be a GlobalWindowDataset")
        if self.dimensions is not None:
            if dataset.window_size != self.dimensions.window_size:
                raise ValueError("dataset window_size does not match the fitted artifact")
            if dataset.horizon != self.dimensions.horizon:
                raise ValueError("dataset horizon does not match the fitted artifact")
            if tuple(dataset.exogenous_columns) != self.dimensions.exogenous_columns:
                raise ValueError(
                    "dataset exogenous feature order does not match the fitted artifact"
                )
            if tuple(dataset.static_feature_names) != self.dimensions.static_feature_names:
                raise ValueError(
                    "dataset static feature order does not match the fitted artifact"
                )

        resolved_device = _resolve_inference_device(device)
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self._inference_batch_size,
            shuffle=False,
            num_workers=0,
        )
        model = self.model.to(resolved_device)
        model.eval()
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                model_inputs = {
                    name: tensor.to(resolved_device)
                    for name, tensor in batch["model_inputs"].items()
                }
                output = model(**model_inputs)
                metadata = batch["metadata"]
                prediction = output.get("y_pred")
                if not isinstance(prediction, torch.Tensor):
                    raise KeyError("Model output must contain 'y_pred'")
                prediction_scaled = prediction.detach().cpu().numpy()
                actual_scaled = batch["targets"]["y_future"].detach().cpu().numpy()
                actual_raw = batch["targets"]["y_future_raw"].detach().cpu().numpy()
                batch_count = prediction_scaled.shape[0]
                for batch_index in range(batch_count):
                    center = float(_batch_value(metadata["center"], batch_index))
                    scale = float(_batch_value(metadata["scale"], batch_index))
                    transform = str(_batch_value(metadata.get("transform", ["identity"] * batch_count), batch_index))
                    parameters = ContextScale(center=center, scale=scale, transform=transform)
                    raw_row = ContextScaler.inverse_transform(
                        prediction_scaled[batch_index], parameters
                    )
                    for horizon_index in range(prediction_scaled.shape[1]):
                        pred_scaled = float(prediction_scaled[batch_index, horizon_index, 0])
                        pred_raw = float(raw_row[horizon_index, 0])
                        rows.append(
                            {
                                CROSS_KEY_COLUMN: str(
                                    _batch_value(metadata[CROSS_KEY_COLUMN], batch_index)
                                ),
                                ACCOUNT_CURRENCY_ID_COLUMN: str(
                                    _batch_value(
                                        metadata[ACCOUNT_CURRENCY_ID_COLUMN], batch_index
                                    )
                                ),
                                SERIES_TYPE_COLUMN: str(
                                    _batch_value(metadata[SERIES_TYPE_COLUMN], batch_index)
                                ),
                                "cutoff": str(_batch_value(metadata["cutoff"], batch_index)),
                                "horizon_step": horizon_index + 1,
                                "prediction": float(pred_raw),
                                "actual": float(actual_raw[batch_index, horizon_index, 0]),
                                "prediction_scaled": pred_scaled,
                                "actual_scaled": float(
                                    actual_scaled[batch_index, horizon_index, 0]
                                ),
                                "center": center,
                                "scale": scale,
                            }
                        )
        if not rows:
            return pl.DataFrame(schema={
                CROSS_KEY_COLUMN: pl.String,
                ACCOUNT_CURRENCY_ID_COLUMN: pl.String,
                SERIES_TYPE_COLUMN: pl.String,
                "cutoff": pl.String,
                "horizon_step": pl.Int64,
                "prediction": pl.Float64,
                "actual": pl.Float64,
                "prediction_scaled": pl.Float64,
                "actual_scaled": pl.Float64,
                "center": pl.Float64,
                "scale": pl.Float64,
            }).select(FORECAST_COLUMNS)
        return pl.DataFrame(rows).select(FORECAST_COLUMNS)

    def run_backtest(
        self,
        *,
        n_mc: int = 100,
        batch_size: int | None = None,
        device: str | None = None,
        show_progress: bool = True,
    ) -> Mapping[str, Any]:
        """API pública: ejecuta el backtest MC-Dropout del modelo entrenado."""

        return self._run_backtest(
            n_mc=n_mc,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )

    def evaluate_p0_auxiliary_heads(self, dataset=None, *, batch_size=None, device=None) -> pd.DataFrame:
        if self.datasets is None:
            raise RuntimeError("GlobalManager has no fitted datasets")
        return evaluate_auxiliary_heads(self.model, dataset or self.datasets.validation_seen,
            batch_size=batch_size or self._inference_batch_size,
            device=device or self.base_training_config.device)

    def p0_patience_diagnostic(self) -> Mapping[str, Any]:
        if self.training_result is None:
            raise RuntimeError("GlobalManager has no productive training result")
        history = tuple(r for r in self.training_result.history
                        if getattr(r, "phase", "") == "productive_training") or tuple(self.training_result.history)
        return diagnose_patience(history,
            configured_patience=self.training_result.training_config.patience).to_dict()

    def _run_backtest(
        self,
        *,
        n_mc: int = 100,
        batch_size: int | None = None,
        device: str | None = None,
        show_progress: bool = True,
    ) -> Mapping[str, Any]:
        """Backtest rolling train/test por serie usando el modelo global."""

        if self.datasets is None:
            raise RuntimeError("GlobalManager has no fitted datasets")
        config = MCDropoutConfig(
            n_mc=int(n_mc),
            batch_size=batch_size or self._inference_batch_size,
            device=device or self.base_training_config.device,
        )
        results = mc_dropout_backtest(
            self.model,
            self.datasets.train,
            self.datasets.validation_seen,
            config=config,
        )
        self._backtest_results = results
        for series_id in results.get("by_series", {}):
            self._series_status[series_id] = {"stage": "backtest", "status": "ok"}
        report = results.get("run_report")
        if show_progress and report is not None:
            print(report.format_summary())
        return results

    def run_future_forecast(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        n_mc: int = 100,
        batch_size: int | None = None,
        device: str | None = None,
        series_ids: Sequence[str] | None = None,
    ) -> Mapping[str, pd.DataFrame]:
        """API pública: ejecuta forecast futuro por rango o por número de pasos.

        ``n_steps`` representa el horizonte total solicitado. El tamaño de cada
        bloque interno se toma de ``dimensions.horizon`` (rollout chunk del
        modelo persistido), por lo que ambos conceptos permanecen separados.
        """

        return self._run_forecast(
            start_date=start_date,
            end_date=end_date,
            n_steps=n_steps,
            max_steps=max_steps,
            n_mc=n_mc,
            batch_size=batch_size,
            device=device,
            series_ids=series_ids,
        )

    def _run_forecast(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        n_mc: int = 100,
        batch_size: int | None = None,
        device: str | None = None,
        series_ids: Sequence[str] | None = None,
    ) -> Mapping[str, pd.DataFrame]:
        """Forecast futuro real por ``cross_key_id`` con MC-Dropout."""

        dimensions = self._require_dimensions()
        if self._source_global_long is None or self._source_calendar is None:
            raise RuntimeError(
                "Future forecast requires a dataset_factory exposing global_long and calendar"
            )
        config = MCDropoutConfig(
            n_mc=int(n_mc),
            batch_size=batch_size or self._inference_batch_size,
            device=device or self.base_training_config.device,
        )
        results, consolidated = forecast_future_mc(
            self.model,
            self._source_global_long,
            self._source_calendar,
            window_size=dimensions.window_size,
            horizon=dimensions.horizon,
            exogenous_columns=dimensions.exogenous_columns,
            static_feature_encoder=self._require_static_feature_encoder(),
            start_date=start_date,
            end_date=end_date,
            n_steps=n_steps,
            max_steps=max_steps,
            series_ids=series_ids,
            config=config,
        )
        self._future_results = results
        self._df_forecasts = consolidated
        self._df_outliers = (
            build_train_reference_outliers(self._backtest_results)
            if self._backtest_results
            else build_legacy_series_and_outliers(self._source_global_long)[1]
        )
        for series_id in results:
            self._series_status[series_id] = {"stage": "forecast", "status": "ok"}
        return results

    def visualise(
        self,
        *,
        bt_start: str,
        bt_end: str,
        fc_start: str,
        fc_end: str,
        series_ids: Sequence[str] | None = None,
    ) -> None:
        """Genera Backtest, Forecast y Backtest+Forecast por serie."""

        if not self._backtest_results:
            raise RuntimeError("Run run_backtest() before visualise()")
        if not self._future_results:
            raise RuntimeError("Run run_future_forecast() before visualise()")
        if self._source_global_long is None:
            raise RuntimeError("Global source frame is unavailable")
        self._df_outliers = visualise_legacy_contract(
            backtest_results=self._backtest_results,
            future_results=self._future_results,
            global_long=self._source_global_long,
            bt_start=bt_start,
            bt_end=bt_end,
            fc_start=fc_start,
            fc_end=fc_end,
            series_ids=series_ids,
        )

    @property
    def temporal_alignment_report(self) -> pl.DataFrame:
        """Cobertura target/exógenas calculada sin modificar el frame fuente."""
        return self._temporal_alignment_report.clone()

    @property
    def series_status(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._series_status, orient="index")

    @property
    def backtest_results(self) -> Mapping[str, Any]:
        """Vista pública de resultados de backtest del run activo."""
        return self._backtest_results

    @property
    def future_results(self) -> Mapping[str, pd.DataFrame]:
        """Vista pública de forecasts futuros separados por serie."""
        return self._future_results

    @property
    def forecast_frame(self) -> pd.DataFrame:
        """Forecast futuro consolidado; se devuelve una copia defensiva."""
        return self._df_forecasts.copy()

    @property
    def outliers_frame(self) -> pd.DataFrame:
        """Outliers jerárquicos del run; se devuelve una copia defensiva."""
        return self._df_outliers.copy()

    def run_results(self) -> Mapping[str, Any]:
        """Contrato primario del workflow pooled estandarizado."""

        return {
            "training": self._productive_training_results,
            "backtest": self._backtest_results,
            "forecast": self._future_results,
            "df_forecasts": self._df_forecasts,
            "df_outliers": self._df_outliers,
        }

    def legacy_results(self) -> Mapping[str, Any]:
        """Contrato de resultados conservado para notebooks consumidores."""

        return {
            "warm": self._warmup_results,
            "fine": self._finetune_results,
            "backtest": self._backtest_results,
            "forecast": self._future_results,
            "df_forecasts": self._df_forecasts,
            "df_outliers": self._df_outliers,
        }

    def _capture_source_frames(self, dataset_factory: DatasetFactory) -> None:
        global_long = getattr(dataset_factory, "global_long", None)
        calendar = getattr(dataset_factory, "calendar", None)
        self._source_global_long = global_long if isinstance(global_long, pl.DataFrame) else None
        self._source_calendar = calendar if isinstance(calendar, pl.DataFrame) else None
        encoder = getattr(dataset_factory, "static_feature_encoder", None)
        if isinstance(encoder, StaticFeatureEncoder):
            self.static_feature_encoder = encoder
        alignment_report = getattr(dataset_factory, "temporal_alignment_report", None)
        self._temporal_alignment_report = (
            alignment_report.clone()
            if isinstance(alignment_report, pl.DataFrame)
            else pl.DataFrame()
        )

    def _phase_result(
        self,
        phase: str,
        *,
        include_consolidation: bool = False,
        include_pooled_continuation: bool = False,
    ) -> Mapping[str, Any]:
        training = self._require_training_result()
        phases = {str(phase)}
        if include_consolidation:
            phases.add("consolidation")
        if include_pooled_continuation:
            phases.add("pooled_continuation")
        stages = [stage for stage in training.stages if stage.stage.phase in phases]
        return {
            "architecture": self.architecture,
            "best_candidate": dict(self.best_candidate),
            "stages": [
                {
                    "name": stage.stage.name,
                    "phase": stage.stage.phase,
                    "best_epoch": stage.best_epoch,
                    "best_score": stage.best_score,
                    "start_state_digest": stage.start_state_digest,
                    "end_state_digest": stage.end_state_digest,
                    "epochs_completed": len(stage.history),
                    "validation": {
                        name: metrics.to_dict()
                        for name, metrics in stage.validation.items()
                    },
                }
                for stage in stages
            ],
        }

    def run_summary(self) -> GlobalRunSummary:
        training = self._require_training_result()
        dimensions = self._require_dimensions()
        if self.hpo_result is not None:
            best_hpo_value = float(self.hpo_result.training.best_score)
            num_hpo_trials = len(self.hpo_result.study.trials)
        else:
            manifest = self.loaded_manifest or {}
            best_hpo_value = float(manifest.get("best_hpo_value", math.nan))
            num_hpo_trials = int(manifest.get("num_hpo_trials", 0))
        return GlobalRunSummary(
            architecture=self.architecture,
            best_score=float(training.best_score),
            total_epochs=int(training.total_epochs),
            best_hpo_value=best_hpo_value,
            num_hpo_trials=num_hpo_trials,
            dimensions=dimensions,
            state_digest=state_dict_digest(training.model.state_dict()),
        )

    def save_model(self, artifact_dir: str | os.PathLike[str]) -> Path:
        """Persiste el run de forma atómica en un directorio autocontenido."""

        training = self._require_training_result()
        dimensions = self._require_dimensions()
        destination = Path(artifact_dir).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        try:
            state_path = temporary / MODEL_FILENAME
            state = {
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in training.model.state_dict().items()
                }
            }
            torch.save(state, state_path)
            checkpoint_sha256 = _sha256_file(state_path)
            state_digest_value = state_dict_digest(state["state_dict"])

            metrics = {
                name: value.to_dict()
                for name, value in training.validation.items()
            }
            history = {
                "summary": training.to_summary(),
                "epochs": [_jsonable(asdict(record)) for record in training.history],
            }
            hpo_summary = self._hpo_summary()
            split = _jsonable(self.split_manifest)
            manifest: MutableMapping[str, Any] = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "architecture": self.architecture,
                "dimensions": _jsonable(asdict(dimensions)),
                "model_config": _jsonable(dict(training.model_config)),
                "training_config": _jsonable(asdict(training.training_config)),
                "hpo_config": _jsonable(asdict(self.hpo_config)),
                "training_schedule_config": _jsonable(asdict(training.curriculum_config)),
                "best_candidate": _jsonable(self.best_candidate),
                "best_score": float(training.best_score),
                "total_epochs": int(training.total_epochs),
                "best_hpo_value": hpo_summary["best_value"],
                "num_hpo_trials": hpo_summary["num_trials"],
                "state_digest": state_digest_value,
                "checkpoint_sha256": checkpoint_sha256,
                "run_metadata": _jsonable(self.run_metadata),
                "static_feature_encoder": _jsonable(
                    self._require_static_feature_encoder().to_dict()
                ),
                "files": [
                    MODEL_FILENAME,
                    METRICS_FILENAME,
                    HISTORY_FILENAME,
                    HPO_FILENAME,
                    SPLIT_FILENAME,
                ],
            }
            _write_json(temporary / METRICS_FILENAME, metrics)
            _write_json(temporary / HISTORY_FILENAME, history)
            _write_json(temporary / HPO_FILENAME, hpo_summary)
            _write_json(temporary / SPLIT_FILENAME, split)
            _write_json(temporary / MANIFEST_FILENAME, manifest)

            if destination.exists():
                if destination.is_file():
                    raise ValueError("artifact_dir points to an existing file")
                shutil.rmtree(destination)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    @classmethod
    def load_model(
        cls,
        artifact_dir: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> "GlobalManager":
        """Reconstruye el modelo y verifica checksums antes de cargar pesos."""

        root = Path(artifact_dir).expanduser().resolve()
        manifest = _read_json(root / MANIFEST_FILENAME)
        artifact_version = str(manifest.get("artifact_schema_version", ""))
        if artifact_version not in {"1.4", ARTIFACT_SCHEMA_VERSION}:
            raise ValueError("Unsupported global artifact schema version")
        architecture = str(manifest["architecture"])
        training_config = GlobalTrainingConfig(**manifest["training_config"])
        hpo_config = GlobalHPOConfig(**manifest.get("hpo_config", {}))
        schedule_payload = manifest.get("training_schedule_config", manifest.get("curriculum_config", {}))
        schedule_config = GlobalTrainingScheduleConfig(**schedule_payload)
        manager = cls(
            architecture,
            base_training_config=training_config,
            hpo_config=hpo_config,
            schedule_config=schedule_config,
            seed=training_config.seed,
        )
        dimensions_payload = manifest["dimensions"]
        dimensions = GlobalRunDimensions(
            window_size=int(dimensions_payload["window_size"]),
            horizon=int(dimensions_payload["horizon"]),
            exogenous_dim=int(dimensions_payload["exogenous_dim"]),
            static_dim=int(dimensions_payload["static_dim"]),
            exogenous_columns=tuple(dimensions_payload.get("exogenous_columns", ())),
            static_feature_names=tuple(dimensions_payload.get("static_feature_names", ())),
        )
        dimensions.validate()

        state_path = root / MODEL_FILENAME
        if _sha256_file(state_path) != manifest["checkpoint_sha256"]:
            raise ValueError("Persisted model checkpoint checksum mismatch")
        payload = _safe_torch_load(state_path, map_location=map_location)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Persisted checkpoint does not contain a state_dict")
        if state_dict_digest(state_dict) != manifest["state_digest"]:
            raise ValueError("Persisted model state digest mismatch")

        model = build_global_model(
            architecture,
            manifest["model_config"],
            window_size=dimensions.window_size,
            horizon=dimensions.horizon,
            exogenous_dim=dimensions.exogenous_dim,
            static_dim=dimensions.static_dim,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(map_location)
        model.eval()

        metrics_payload = _read_json(root / METRICS_FILENAME)
        validation = {
            name: GlobalValidationMetrics(**value)
            for name, value in metrics_payload.items()
        }
        history_payload = _read_json(root / HISTORY_FILENAME)
        summary = history_payload["summary"]
        # Loaded artifacts retain the serializable evidence. Stage dataclasses are
        # intentionally not reconstructed because inference only needs one model.
        loaded_result = GlobalCurriculumTrainingResult(
            architecture=architecture,
            model=model,
            model_config=dict(manifest["model_config"]),
            training_config=training_config,
            curriculum_config=schedule_config,
            stages=(),
            history=(),
            validation=validation,
            best_score=float(manifest["best_score"]),
            total_epochs=int(manifest["total_epochs"]),
        )
        manager.training_result = loaded_result
        manager.dimensions = dimensions
        manager.split_manifest = _read_json(root / SPLIT_FILENAME)
        manager.run_metadata = _ensure_mapping(manifest.get("run_metadata", {}))
        manager.loaded_manifest = dict(manifest)
        manager.static_feature_encoder = StaticFeatureEncoder.from_dict(
            manifest["static_feature_encoder"]
        )
        # Keep full persisted history discoverable without pretending it is a
        # live training object.
        manager.persisted_history = history_payload
        manager.persisted_hpo_summary = _read_json(root / HPO_FILENAME)
        return manager

    def save_model_s3(
        self,
        s3_root: str = DEFAULT_FINANCIAL_GPT_S3_ROOT,
        *,
        run_id: str | None = None,
        reports_dir: str | os.PathLike[str] | None = None,
        s3_client=None,
        update_latest: bool = True,
    ) -> str:
        """Guarda un run completo bajo la raíz Financial-GPT y publica `_SUCCESS`.

        La escritura es inmutable por ``run_id``. ``latest.json`` se actualiza
        únicamente después de verificar todos los objetos y publicar el marker.
        """

        training = self._require_training_result()
        dimensions = self._require_dimensions()
        resolved_run_id = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + state_dict_digest(training.model.state_dict())[:8]
        )
        validate_component(resolved_run_id, label="run_id")
        run_uri = build_run_uri(s3_root, self.architecture, resolved_run_id)

        with tempfile.TemporaryDirectory(prefix="financial-gpt-s3-") as tmp:
            staged = Path(tmp) / "run"
            self._stage_s3_run(
                staged,
                reports_dir=reports_dir,
            )
            result = upload_atomic_run(
                staged,
                run_uri,
                s3_root=s3_root,
                architecture=self.architecture,
                run_id=resolved_run_id,
                state_digest=state_dict_digest(training.model.state_dict()),
                client=s3_client,
                update_latest=update_latest,
            )
        self.last_s3_save_result = result
        self.s3_run_uri = result.run_uri
        return result.run_uri

    @classmethod
    def load_model_s3(
        cls,
        run_uri: str,
        *,
        map_location: str | torch.device = "cpu",
        s3_client=None,
    ) -> "GlobalManager":
        """Carga un run S3 comprometido y verifica sus checksums."""

        with tempfile.TemporaryDirectory(prefix="financial-gpt-load-") as tmp:
            downloaded = download_verified_run(
                run_uri,
                Path(tmp) / "run",
                client=s3_client,
                include_prefixes=("model/", "evidence/"),
            )
            manager = cls.load_model(
                downloaded.local_root / "model",
                map_location=map_location,
            )
            manager.s3_run_uri = downloaded.run_uri
            manager.s3_success_manifest = dict(downloaded.success)
            manager.s3_checksums_manifest = dict(downloaded.checksums)
            evidence = downloaded.local_root / "evidence"
            manager.persisted_s3_evidence = _read_s3_evidence(evidence)
        return manager

    @classmethod
    def load_latest_model_s3(
        cls,
        architecture: str,
        *,
        s3_root: str = DEFAULT_FINANCIAL_GPT_S3_ROOT,
        map_location: str | torch.device = "cpu",
        s3_client=None,
    ) -> "GlobalManager":
        """Resuelve ``latest.json`` por arquitectura y carga el run comprometido."""

        run_uri = resolve_latest_run_uri(
            s3_root,
            architecture,
            client=s3_client,
        )
        return cls.load_model_s3(
            run_uri,
            map_location=map_location,
            s3_client=s3_client,
        )

    def _stage_s3_run(
        self,
        destination: Path,
        *,
        reports_dir: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Construye el layout portable model/evidence/reports antes del upload."""

        training = self._require_training_result()
        destination.mkdir(parents=True, exist_ok=False)
        model_dir = destination / "model"
        evidence_dir = destination / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(model_dir)

        _write_json(evidence_dir / "run_summary.json", self.run_summary().to_dict())
        _write_json(evidence_dir / "split_manifest.json", _jsonable(self.split_manifest))
        _write_json(evidence_dir / "best_candidate.json", _jsonable(self.best_candidate))
        _write_json(
            evidence_dir / "curriculum_history.json",
            {
                "summary": training.to_summary(),
                "stages": [
                    {
                        "name": stage.stage.name,
                        "phase": stage.stage.phase,
                        "start_state_digest": stage.start_state_digest,
                        "end_state_digest": stage.end_state_digest,
                        "best_epoch": stage.best_epoch,
                        "best_score": stage.best_score,
                        "stopped_early": stage.stopped_early,
                    }
                    for stage in training.stages
                ],
            },
        )
        _write_training_history_parquet(
            evidence_dir / "training_history.parquet",
            training,
        )
        _write_metrics_parquet(
            evidence_dir / "metrics.parquet",
            training.validation,
        )

        if reports_dir is not None:
            source = Path(reports_dir).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"reports_dir does not exist: {source}")
            shutil.copytree(source, destination / "reports")
        return destination

    @property
    def _inference_batch_size(self) -> int:
        training = self._require_training_result()
        return int(training.training_config.batch_size)

    def _default_dataset(self, name: str) -> GlobalWindowDataset:
        if self.datasets is None:
            raise RuntimeError(
                "No in-memory datasets are available; provide an explicit dataset after load"
            )
        return getattr(self.datasets, name)

    def _require_training_result(self) -> GlobalCurriculumTrainingResult:
        if self.training_result is None:
            raise RuntimeError("GlobalManager must be fitted or loaded first")
        return self.training_result

    def _require_dimensions(self) -> GlobalRunDimensions:
        if self.dimensions is None:
            raise RuntimeError("GlobalManager has no run dimensions")
        return self.dimensions

    def _require_static_feature_encoder(self) -> StaticFeatureEncoder:
        if self.static_feature_encoder is None:
            raise RuntimeError("GlobalManager has no fitted static feature encoder")
        return self.static_feature_encoder

    def _hpo_summary(self) -> Mapping[str, Any]:
        if self.hpo_result is None:
            if hasattr(self, "persisted_hpo_summary"):
                return dict(self.persisted_hpo_summary)
            raise RuntimeError("No HPO result is available")
        study = self.hpo_result.study
        trials = []
        for trial in study.trials:
            trials.append(
                {
                    "number": int(trial.number),
                    "state": trial.state.name,
                    "value": None if trial.value is None else float(trial.value),
                    "params": _jsonable(trial.params),
                    "user_attrs": _jsonable(trial.user_attrs),
                }
            )
        selected_number = self.hpo_result.selected_trial_number
        selected_params = {}
        if selected_number is not None:
            selected = next(
                (trial for trial in study.trials if int(trial.number) == int(selected_number)),
                None,
            )
            if selected is not None:
                selected_params = dict(selected.params)
        return {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "proxy_best_value": float(study.best_value),
            "proxy_best_params": _jsonable(study.best_params),
            "best_value": float(self.hpo_result.training.best_score),
            "best_params": _jsonable(selected_params),
            "selected_trial_params": _jsonable(selected_params),
            "selected_trial_number": selected_number,
            "fidelity_scores": _jsonable(self.hpo_result.fidelity_scores or {}),
            "best_candidate": _jsonable(self.hpo_result.best_candidate.to_dict()),
            "hpo_config": _jsonable(asdict(self.hpo_config)),
            "pruner": type(study.pruner).__name__,
            "num_trials": len(study.trials),
            "trials": trials,
        }


def _write_training_history_parquet(
    path: Path,
    training: GlobalCurriculumTrainingResult,
) -> None:
    rows = []
    for record in training.history:
        rows.append(
            {
                "stage_name": record.stage_name,
                "phase": record.phase,
                "global_epoch": int(record.global_epoch),
                "stage_epoch": int(record.stage_epoch),
                "train_loss": float(record.train_loss),
                "validation_objective": float(record.validation_objective),
                "learning_rate": float(record.learning_rate),
                "current_samples": int(record.current_samples),
                "replay_samples": int(record.replay_samples),
                "stratum_samples_json": json.dumps(
                    dict(record.stratum_samples),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "validation_json": json.dumps(
                    {name: metrics.to_dict() for name, metrics in record.validation.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    if rows:
        frame = pl.DataFrame(rows)
    else:
        frame = pl.DataFrame(
            schema={
                "stage_name": pl.String,
                "phase": pl.String,
                "global_epoch": pl.Int64,
                "stage_epoch": pl.Int64,
                "train_loss": pl.Float64,
                "validation_objective": pl.Float64,
                "learning_rate": pl.Float64,
                "current_samples": pl.Int64,
                "replay_samples": pl.Int64,
                "stratum_samples_json": pl.String,
                "validation_json": pl.String,
            }
        )
    frame.write_parquet(path)


def _write_metrics_parquet(
    path: Path,
    validation: Mapping[str, GlobalValidationMetrics],
) -> None:
    rows = []
    for dataset_name, metrics in validation.items():
        rows.append(
            {
                "dataset": dataset_name,
                "robust_macro_mase": float(metrics.robust_macro_mase),
                "macro_mae": float(metrics.macro_mae),
                "macro_rmse": float(metrics.macro_rmse),
                "micro_mae": float(metrics.micro_mae),
                "raw_macro_mae": float(metrics.raw_macro_mae),
                "raw_macro_rmse": float(metrics.raw_macro_rmse),
                "raw_macro_wmape": float(metrics.raw_macro_wmape),
                "raw_macro_smape": float(metrics.raw_macro_smape),
                "num_series": int(metrics.num_series),
                "num_points": int(metrics.num_points),
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _read_s3_evidence(root: Path) -> Mapping[str, Any]:
    evidence: MutableMapping[str, Any] = {}
    for name in (
        "run_summary.json",
        "split_manifest.json",
        "best_candidate.json",
        "curriculum_history.json",
    ):
        path = root / name
        if path.is_file():
            evidence[name] = _read_json(path)
    for name in ("training_history.parquet", "metrics.parquet"):
        path = root / name
        if path.is_file():
            evidence[name] = pl.read_parquet(path).to_dicts()
    return evidence


def _normalize_split_manifest(
    split: GlobalSeriesSplit | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if split is None:
        return {}
    if isinstance(split, GlobalSeriesSplit):
        split.validate()
        return dict(split.to_dict())
    if isinstance(split, Mapping):
        return _ensure_mapping(split)
    raise TypeError("split_manifest must be GlobalSeriesSplit, mapping, or None")


def _ensure_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping")
    payload = _jsonable(dict(value))
    if not isinstance(payload, Mapping):
        raise TypeError("Mapping could not be normalized")
    return dict(payload)


def _resolve_inference_device(
    requested: str | torch.device | None,
) -> torch.device:
    if requested is None or str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is not available")
    return device


def _batch_value(values: Any, index: int) -> Any:
    if isinstance(values, torch.Tensor):
        value = values[index]
        return value.item() if value.ndim == 0 else value
    if isinstance(values, np.ndarray):
        return values[index].item() if values[index].ndim == 0 else values[index]
    return values[index]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path.name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_torch_load(
    path: Path,
    *,
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch anterior a weights_only
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Persisted torch payload must be a mapping")
    return payload
