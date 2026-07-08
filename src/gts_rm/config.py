from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._legacy import ensure_cp20_import_path
from .paths import MAC3_TEST_ROOT, REPO_ROOT

ensure_cp20_import_path()

from financial_gpt_flags import (  # noqa: E402
    FinancialGPTFeatureFlags,
    FinancialGPTStageConfig,
    LocalResidualConfig,
    PatchTokenizerConfig,
    QuantileHeadConfig,
    SelfSupervisedConfig,
)
from global_notebook import GlobalNotebookConfig  # noqa: E402
from global_training import GlobalCandidateConfig, GlobalTrainingConfig  # noqa: E402

SUPPORTED_ARCHITECTURES = ("mlp", "mlp_vae", "rnn", "rnn_bi")
CONFIG_CHECKPOINT = "CP24"


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def load_json_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a JSON object: {resolved}")
    return payload


def _require_checkpoint(payload: Mapping[str, Any], *, path: str | Path) -> None:
    checkpoint = payload.get("checkpoint")
    if checkpoint != CONFIG_CHECKPOINT:
        raise ValueError(f"Expected {CONFIG_CHECKPOINT} config in {path}, got {checkpoint!r}")


def load_stage_config(path: str | Path | None = None) -> FinancialGPTStageConfig:
    config_path = path or MAC3_TEST_ROOT / "configs" / "stage_cp20.json"
    payload = load_json_config(config_path)
    _require_checkpoint(payload, path=config_path)
    stage_payload = payload.get("stage")
    if not isinstance(stage_payload, Mapping):
        raise TypeError("stage config requires a 'stage' object")
    stage = FinancialGPTStageConfig(
        flags=FinancialGPTFeatureFlags.from_mapping(stage_payload.get("flags")),
        patch_tokenizer=PatchTokenizerConfig(**dict(stage_payload.get("patch_tokenizer") or {})),
        local_residual=LocalResidualConfig(**dict(stage_payload.get("local_residual") or {})),
        quantile_head=QuantileHeadConfig(**dict(stage_payload.get("quantile_head") or {})),
        self_supervised=SelfSupervisedConfig(**dict(stage_payload.get("self_supervised") or {})),
    )
    stage.validate()
    return stage


def load_training_config(path: str | Path | None = None) -> GlobalTrainingConfig:
    config_path = path or MAC3_TEST_ROOT / "configs" / "training_smoke.json"
    payload = load_json_config(config_path)
    _require_checkpoint(payload, path=config_path)
    training_payload = payload.get("training_config")
    if not isinstance(training_payload, Mapping):
        raise TypeError("training config requires a 'training_config' object")
    config = GlobalTrainingConfig(**dict(training_payload))
    config.validate()
    return config


def load_candidate_configs(path: str | Path | None = None) -> dict[str, GlobalCandidateConfig]:
    config_path = path or MAC3_TEST_ROOT / "configs" / "candidates_smoke.json"
    payload = load_json_config(config_path)
    _require_checkpoint(payload, path=config_path)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, Mapping):
        raise TypeError("candidate config requires a 'candidates' object")
    candidates: dict[str, GlobalCandidateConfig] = {}
    for architecture in SUPPORTED_ARCHITECTURES:
        candidate_payload = raw_candidates.get(architecture)
        if not isinstance(candidate_payload, Mapping):
            raise ValueError(f"Missing candidate config for architecture {architecture!r}")
        training_payload = candidate_payload.get("training_config")
        if not isinstance(training_payload, Mapping):
            raise TypeError(f"Candidate {architecture!r} requires training_config")
        candidate = GlobalCandidateConfig(
            window_size=int(candidate_payload["window_size"]),
            model_config=dict(candidate_payload["model_config"]),
            training_config=GlobalTrainingConfig(**dict(training_payload)),
        )
        candidate.validate()
        candidates[architecture] = candidate
    extra = sorted(set(raw_candidates) - set(SUPPORTED_ARCHITECTURES))
    if extra:
        raise ValueError(f"Unexpected candidate architectures: {extra}")
    return candidates


def load_notebook_configs(path: str | Path | None = None) -> dict[str, GlobalNotebookConfig]:
    config_path = path or MAC3_TEST_ROOT / "configs" / "notebooks_mac3.json"
    payload = load_json_config(config_path)
    _require_checkpoint(payload, path=config_path)
    raw_notebooks = payload.get("notebooks")
    if not isinstance(raw_notebooks, Mapping):
        raise TypeError("notebook config requires a 'notebooks' object")
    notebooks: dict[str, GlobalNotebookConfig] = {}
    for architecture in SUPPORTED_ARCHITECTURES:
        notebook_payload = raw_notebooks.get(architecture)
        if not isinstance(notebook_payload, Mapping):
            raise ValueError(f"Missing notebook config for architecture {architecture!r}")
        config = GlobalNotebookConfig(**dict(notebook_payload))
        config.validate()
        notebooks[architecture] = config
    extra = sorted(set(raw_notebooks) - set(SUPPORTED_ARCHITECTURES))
    if extra:
        raise ValueError(f"Unexpected notebook architectures: {extra}")
    return notebooks


def load_smoke_config(architecture: str) -> dict[str, Any]:
    key = str(architecture).strip().lower()
    if key not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture={architecture!r}; expected {SUPPORTED_ARCHITECTURES}")
    payload = load_json_config(MAC3_TEST_ROOT / "configs" / f"smoke_global_{key}.json")
    if payload.get("architecture") != key:
        raise ValueError(f"Smoke config architecture mismatch for {key!r}")
    return payload


@dataclass(frozen=True)
class MAC3ConfigBundle:
    stage: FinancialGPTStageConfig
    training: GlobalTrainingConfig
    candidates: Mapping[str, GlobalCandidateConfig]
    notebooks: Mapping[str, GlobalNotebookConfig]
    smokes: Mapping[str, Mapping[str, Any]]

    def validate(self) -> None:
        self.stage.validate()
        self.training.validate()
        if tuple(self.candidates) != SUPPORTED_ARCHITECTURES:
            raise ValueError("candidate configs must cover all supported architectures")
        if tuple(self.notebooks) != SUPPORTED_ARCHITECTURES:
            raise ValueError("notebook configs must cover all supported architectures")
        if tuple(self.smokes) != SUPPORTED_ARCHITECTURES:
            raise ValueError("smoke configs must cover all supported architectures")
        for architecture, candidate in self.candidates.items():
            candidate.validate()
            if self.smokes[architecture]["model_config"] != candidate.model_config:
                raise ValueError(f"Smoke and candidate model_config differ for {architecture!r}")
            if self.notebooks[architecture].architecture != architecture:
                raise ValueError(f"Notebook config architecture mismatch for {architecture!r}")


def load_mac3_config_bundle() -> MAC3ConfigBundle:
    bundle = MAC3ConfigBundle(
        stage=load_stage_config(),
        training=load_training_config(),
        candidates=load_candidate_configs(),
        notebooks=load_notebook_configs(),
        smokes={architecture: load_smoke_config(architecture) for architecture in SUPPORTED_ARCHITECTURES},
    )
    bundle.validate()
    return bundle


__all__ = [
    "CONFIG_CHECKPOINT",
    "SUPPORTED_ARCHITECTURES",
    "FinancialGPTFeatureFlags",
    "FinancialGPTStageConfig",
    "GlobalCandidateConfig",
    "GlobalNotebookConfig",
    "GlobalTrainingConfig",
    "LocalResidualConfig",
    "MAC3ConfigBundle",
    "PatchTokenizerConfig",
    "QuantileHeadConfig",
    "SelfSupervisedConfig",
    "load_candidate_configs",
    "load_json_config",
    "load_mac3_config_bundle",
    "load_notebook_configs",
    "load_smoke_config",
    "load_stage_config",
    "load_training_config",
    "resolve_repo_path",
]
