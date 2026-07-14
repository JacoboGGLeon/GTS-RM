from __future__ import annotations

from pathlib import Path

import nbformat
import optuna
import pytest
from pydantic import ValidationError

from global_surface_config import (
    AuxiliaryHeadsConfig,
    GlobalActiveConfiguration,
    InferenceConfig,
    ModalityEncoderDefaults,
    ModalityEncoderHPOSpace,
    ModelFeatureConfig,
    ResidualDecoderConfig,
    TrainingBudgetConfig,
)
from global_training import GlobalHPOConfig, GlobalTrainingConfig, suggest_global_candidate

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "code_03_GLOBAL_MLP_E_D.ipynb",
    "code_03_GLOBAL_MLP_VaE_D.ipynb",
    "code_03_GLOBAL_RNN_E_D.ipynb",
    "code_03_GLOBAL_RNNBi_E_D.ipynb",
)
DEAD_NOTEBOOK_CONTROLS = (
    "USE_CALENDAR_ENCODER",
    "USE_PATCH_TOKENIZER",
    "USE_QUANTILE_HEAD",
    "USE_SELF_SUPERVISED_PRETRAINING",
    "USE_HPO",
    "LATENT_DIM",
    "HPO_MODALITY_ENCODER_ARCHITECTURE",
)


def active_config() -> GlobalActiveConfiguration:
    return GlobalActiveConfiguration(
        features=ModelFeatureConfig(),
        modality_defaults=ModalityEncoderDefaults(),
        modality_hpo=ModalityEncoderHPOSpace(),
        residual=ResidualDecoderConfig(),
        auxiliary=AuxiliaryHeadsConfig(),
        budget=TrainingBudgetConfig(),
        inference=InferenceConfig(),
    )


def test_active_surface_is_strict_serializable_and_cross_validated() -> None:
    config = active_config()
    assert config.schema_version == "22.3.2b"
    assert config.modality_defaults.future_dim == 32
    assert config.modality_hpo.future_dim_choices == (16, 32, 64, 128)
    assert config.budget.hpo_trials == 80
    assert config.budget.hpo_fidelity_epochs == 12
    assert config.training_kwargs()["future_encoder_dim"] == 32
    assert config.to_dict()["modality_hpo"]["enabled"] is True

    with pytest.raises(ValidationError):
        ModelFeatureConfig(unknown=True)
    with pytest.raises(ValidationError):
        GlobalActiveConfiguration(
            features=ModelFeatureConfig(use_modality_specific_encoders=False),
            modality_defaults=ModalityEncoderDefaults(),
            modality_hpo=ModalityEncoderHPOSpace(enabled=True),
            residual=ResidualDecoderConfig(),
            auxiliary=AuxiliaryHeadsConfig(),
            budget=TrainingBudgetConfig(),
            inference=InferenceConfig(),
        )


def test_hpo_space_rejects_invalid_choices_and_budget() -> None:
    with pytest.raises(ValidationError):
        ModalityEncoderHPOSpace(target_dim_choices=(16, 16))
    with pytest.raises(ValidationError):
        TrainingBudgetConfig(hpo_trials=4, hpo_finalists=5)
    with pytest.raises(ValidationError):
        TrainingBudgetConfig(hpo_epochs=8, hpo_fidelity_epochs=5)


def test_suggest_global_candidate_uses_explicit_modality_space() -> None:
    space = ModalityEncoderHPOSpace(
        target_dim_choices=(11,),
        historical_dim_choices=(12,),
        future_dim_choices=(13,),
        static_dim_choices=(14,),
        fusion_hidden_size_choices=(15,),
        target_layers={"minimum": 2, "maximum": 2},
        historical_layers={"minimum": 2, "maximum": 2},
        future_layers={"minimum": 2, "maximum": 2},
        static_layers={"minimum": 1, "maximum": 1},
        fusion_layers={"minimum": 2, "maximum": 2},
        dropout={"minimum": 0.1, "maximum": 0.1},
        activations=("silu",),
    )
    hpo_config = GlobalHPOConfig(
        modality_encoder_hpo_space=space.model_dump(mode="python")
    )
    trial = optuna.create_study(direction="minimize").ask()
    candidate = suggest_global_candidate(
        trial,
        "mlp",
        GlobalTrainingConfig(use_modality_specific_encoders=True),
        hpo_config=hpo_config,
    )

    assert candidate.model_config["target_encoder_dim"] == 11
    assert candidate.model_config["historical_encoder_dim"] == 12
    assert candidate.model_config["future_encoder_dim"] == 13
    assert candidate.model_config["static_encoder_dim"] == 14
    assert candidate.model_config["fusion_hidden_size"] == 15
    assert candidate.model_config["modality_encoder_dropout_rate"] == 0.1
    assert candidate.model_config["modality_encoder_activation"] == "silu"


def test_notebooks_expose_only_active_controls_and_explicit_hpo_space() -> None:
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        for dead_control in DEAD_NOTEBOOK_CONTROLS:
            assert dead_control not in code, (name, dead_control)
        assert "MODALITY_ENCODER_DEFAULTS = {" in code, name
        assert "MODALITY_ENCODER_HPO_SPACE = {" in code, name
        assert "GlobalActiveConfiguration(" in code, name
        assert "surface=active_config" in code, name
        assert "HPO_TRIALS = 80" in code, name
        assert "HPO_EPOCHS = 5" in code, name
        assert "HPO_FINALISTS = 8" in code, name
        assert "HPO_FIDELITY_EPOCHS = 12" in code, name
        assert '"training_methodology_checkpoint": "22.3.2b"' in code, name
        assert all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in notebook.cells
            if cell.cell_type == "code"
        ), name
