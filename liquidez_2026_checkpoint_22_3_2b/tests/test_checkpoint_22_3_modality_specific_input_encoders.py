from __future__ import annotations

import copy
from pathlib import Path

import nbformat
import optuna
import pytest
import torch

from global_models import build_global_model, list_global_models
from global_training import GlobalTrainingConfig, suggest_global_candidate
from gtrm_config import GTRMModelConfig

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "code_03_GLOBAL_MLP_E_D.ipynb",
    "code_03_GLOBAL_MLP_VaE_D.ipynb",
    "code_03_GLOBAL_RNN_E_D.ipynb",
    "code_03_GLOBAL_RNNBi_E_D.ipynb",
)


def modality_cfg() -> dict[str, object]:
    return {
        "use_modality_specific_encoders": True,
        "latent_dim": 12,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "enc_hidden_size": 16,
        "enc_num_layers": 1,
        "dec_hidden_size": 16,
        "dec_num_layers": 1,
        "rnn_hidden_size": 16,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "target_encoder_dim": 9,
        "historical_encoder_dim": 8,
        "future_encoder_dim": 7,
        "static_encoder_dim": 6,
        "fusion_hidden_size": 18,
        "target_encoder_num_layers": 1,
        "historical_encoder_num_layers": 1,
        "future_encoder_num_layers": 1,
        "static_encoder_num_layers": 1,
        "fusion_num_layers": 1,
        "modality_encoder_dropout_rate": 0.0,
        "modality_encoder_activation": "gelu",
        "use_auxiliary_autoencoder": False,
        "use_local_residual_decoder": True,
        "local_residual_hidden_size": 8,
        "local_residual_num_layers": 1,
        "use_event_head": True,
        "use_magnitude_head": True,
        "use_direction_head": True,
        "auxiliary_head_hidden_size": 8,
        "auxiliary_head_num_layers": 1,
    }


def inputs(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(223)
    values = {
        "y_context": torch.randn(3, 5, 1),
        "x_history": torch.randn(3, 5, 2),
        "x_future": torch.randn(3, 4, 2),
        "x_static": torch.randn(3, 5),
    }
    if requires_grad:
        values = {name: value.requires_grad_() for name, value in values.items()}
    return values


def test_gtrm_config_exposes_stage_23_flag_and_rejects_it_in_stage1() -> None:
    config = GTRMModelConfig(
        architecture="mlp",
        use_modality_specific_encoders=True,
    )
    config.validate(stage=2)
    assert config.stage_flags()["use_modality_specific_encoders"] is True
    with pytest.raises(ValueError, match="Stage 2.3"):
        config.validate(stage=1)


def test_all_architectures_encode_each_modality_and_preserve_contract() -> None:
    for architecture in list_global_models():
        batch = inputs(requires_grad=True)
        model = build_global_model(
            architecture,
            modality_cfg(),
            window_size=5,
            horizon=4,
            exogenous_dim=2,
            static_dim=5,
        )
        output = model(**batch)
        extras = output["extras"]
        assert tuple(output["y_pred"].shape) == (3, 4, 1), architecture
        assert tuple(extras["history_embedding"].shape) == (3, 12), architecture
        assert tuple(extras["target_embedding"].shape) == (3, 9), architecture
        assert tuple(extras["historical_exogenous_embedding"].shape) == (3, 8), architecture
        assert tuple(extras["future_exogenous_embedding"].shape) == (3, 4, 7), architecture
        assert tuple(extras["static_context_embedding"].shape) == (3, 6), architecture
        assert extras["use_modality_specific_encoders"] is True
        assert model.representation_contract()["use_modality_specific_encoders"] is True

        output["y_pred"].square().mean().backward()
        for name, value in batch.items():
            assert value.grad is not None, (architecture, name)
            assert torch.isfinite(value.grad).all(), (architecture, name)
            assert float(value.grad.abs().sum()) > 0.0, (architecture, name)


def test_modalities_are_encoded_independently_before_fusion() -> None:
    model = build_global_model(
        "mlp",
        modality_cfg(),
        window_size=5,
        horizon=4,
        exogenous_dim=2,
        static_dim=5,
    ).eval()
    baseline = inputs()
    changed = copy.deepcopy(baseline)
    changed["y_context"] = changed["y_context"] + 3.0

    first = model(**baseline)["extras"]
    second = model(**changed)["extras"]
    assert not torch.allclose(first["target_embedding"], second["target_embedding"])
    torch.testing.assert_close(
        first["historical_exogenous_embedding"],
        second["historical_exogenous_embedding"],
    )
    torch.testing.assert_close(
        first["future_exogenous_embedding"],
        second["future_exogenous_embedding"],
    )
    torch.testing.assert_close(
        first["static_context_embedding"],
        second["static_context_embedding"],
    )
    assert not torch.allclose(first["history_embedding"], second["history_embedding"])


def test_hpo_tunes_every_modality_encoder_and_fusion() -> None:
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    candidate = suggest_global_candidate(
        trial,
        "rnn",
        GlobalTrainingConfig(
            use_modality_specific_encoders=True,
            hpo_modality_encoder_architecture=True,
        ),
    )
    expected = {
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
        "modality_encoder_dropout_rate",
        "modality_encoder_activation",
    }
    assert expected.issubset(candidate.model_config)
    assert expected.issubset(trial.params)
    assert candidate.model_config["use_modality_specific_encoders"] is True


def test_legacy_joint_encoder_remains_available_for_ablation() -> None:
    config = modality_cfg()
    config["use_modality_specific_encoders"] = False
    model = build_global_model(
        "mlp",
        config,
        window_size=5,
        horizon=4,
        exogenous_dim=2,
        static_dim=5,
    )
    output = model(**inputs())
    assert output["extras"]["use_modality_specific_encoders"] is False
    assert "target_embedding" not in output["extras"]
    assert model.representation_contract()["use_modality_specific_encoders"] is False


def test_all_global_notebooks_enable_and_serialize_stage_23_contract() -> None:
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "USE_MODALITY_SPECIFIC_ENCODERS = True" in source, name
        assert "MODALITY_ENCODER_HPO_SPACE = {" in source, name
        assert '"enabled": True' in source, name
        assert ("use_modality_specific_encoders=(" in source or "use_modality_specific_encoders=feature_config.use_modality_specific_encoders" in source), name
        assert '"training_methodology_checkpoint": "22.3.2b"' in source, name
        assert all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in notebook.cells
            if cell.cell_type == "code"
        ), name


def test_modality_specific_state_dict_roundtrips_strictly() -> None:
    for architecture in list_global_models():
        config = modality_cfg()
        original = build_global_model(
            architecture,
            config,
            window_size=5,
            horizon=4,
            exogenous_dim=2,
            static_dim=5,
        ).eval()
        restored = build_global_model(
            architecture,
            config,
            window_size=5,
            horizon=4,
            exogenous_dim=2,
            static_dim=5,
        ).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        sample = inputs()
        torch.testing.assert_close(
            original(**sample)["y_pred"],
            restored(**sample)["y_pred"],
        )
