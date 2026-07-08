from __future__ import annotations

from typing import Mapping

from ._legacy import ensure_cp20_import_path

ensure_cp20_import_path()

from global_curriculum import (  # noqa: E402
    CurriculumReplaySampler,
    CurriculumStage,
    GlobalCurriculumConfig,
    GlobalCurriculumEpochRecord,
    GlobalCurriculumSession,
    GlobalCurriculumStageResult,
    GlobalCurriculumTrainer,
    GlobalCurriculumTrainingResult,
    GlobalTrainingOrderAblationResult,
    state_dict_digest,
)
from global_training import (  # noqa: E402
    GlobalCandidateConfig,
    GlobalDatasetBundle,
    GlobalEpochRecord,
    GlobalHPOConfig,
    GlobalHPOResult,
    GlobalHPOTrainer,
    GlobalTrainer,
    GlobalTrainingConfig,
    GlobalTrainingResult,
    GlobalValidationMetrics,
    NonFiniteValidationError,
    evaluate_global_model,
    global_forecast_loss,
)


def load_mac3_candidates() -> dict[str, GlobalCandidateConfig]:
    from . import config

    return config.load_candidate_configs()


def get_mac3_candidate(architecture: str) -> GlobalCandidateConfig:
    key = str(architecture).strip().lower()
    candidates = load_mac3_candidates()
    try:
        return candidates[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported MAC3 architecture={architecture!r}") from exc


def load_mac3_training_config() -> GlobalTrainingConfig:
    from . import config

    return config.load_training_config()


def build_trainer_from_candidate(
    architecture: str,
    candidate: GlobalCandidateConfig,
    *,
    training_config: GlobalTrainingConfig | None = None,
) -> GlobalTrainer:
    key = str(architecture).strip().lower()
    candidate.validate()
    resolved_training = training_config or candidate.training_config
    resolved_training.validate()
    return GlobalTrainer(
        key,
        candidate.model_config,
        training_config=resolved_training,
    )


def build_mac3_trainer(
    architecture: str,
    *,
    training_config: GlobalTrainingConfig | None = None,
) -> GlobalTrainer:
    candidate = get_mac3_candidate(architecture)
    return build_trainer_from_candidate(
        architecture,
        candidate,
        training_config=training_config,
    )


def mac3_training_facade_summary() -> Mapping[str, object]:
    candidates = load_mac3_candidates()
    training_config = load_mac3_training_config()
    return {
        "architectures": tuple(candidates),
        "selection_metric": training_config.selection_metric,
        "candidate_window_sizes": {
            architecture: candidate.window_size
            for architecture, candidate in candidates.items()
        },
        "candidate_training_epochs": {
            architecture: candidate.training_config.epochs
            for architecture, candidate in candidates.items()
        },
    }


__all__ = [
    "CurriculumReplaySampler",
    "CurriculumStage",
    "GlobalCandidateConfig",
    "GlobalCurriculumConfig",
    "GlobalCurriculumEpochRecord",
    "GlobalCurriculumSession",
    "GlobalCurriculumStageResult",
    "GlobalCurriculumTrainer",
    "GlobalCurriculumTrainingResult",
    "GlobalDatasetBundle",
    "GlobalEpochRecord",
    "GlobalHPOConfig",
    "GlobalHPOResult",
    "GlobalHPOTrainer",
    "GlobalTrainer",
    "GlobalTrainingConfig",
    "GlobalTrainingOrderAblationResult",
    "GlobalTrainingResult",
    "GlobalValidationMetrics",
    "NonFiniteValidationError",
    "build_mac3_trainer",
    "build_trainer_from_candidate",
    "evaluate_global_model",
    "get_mac3_candidate",
    "global_forecast_loss",
    "load_mac3_candidates",
    "load_mac3_training_config",
    "mac3_training_facade_summary",
    "state_dict_digest",
]
