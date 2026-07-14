from __future__ import annotations

import optuna
import torch

from global_models import build_global_model
from global_surface_config import ModalityEncoderHPOSpace
from global_training import GlobalHPOConfig, GlobalTrainingConfig, suggest_global_candidate


def _model() -> torch.nn.Module:
    return build_global_model(
        "mlp",
        {
            "latent_dim": 8,
            "enc_hidden_size": 16,
            "enc_num_layers": 1,
            "dec_hidden_size": 16,
            "dec_num_layers": 1,
            "dropout_rate": 0.0,
            "use_auxiliary_autoencoder": False,
            "use_local_residual_decoder": True,
            "local_residual_autoregressive": True,
            "local_residual_hidden_size": 8,
            "local_residual_num_layers": 1,
            "local_residual_dropout_rate": 0.0,
        },
        window_size=5,
        horizon=4,
        exogenous_dim=2,
        static_dim=3,
    )


@torch.no_grad()
def test_autoregressive_residual_is_causal_across_horizon() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    embedding = torch.randn(2, 8)
    future = torch.randn(2, 4, 6)
    global_prediction = torch.randn(2, 4, 1)

    _, original = model._apply_local_residual(global_prediction, embedding, future)
    changed_future = future.clone()
    changed_future[:, 3, :] += 100.0
    _, changed = model._apply_local_residual(global_prediction, embedding, changed_future)

    assert original["local_residual_mode"] == "autoregressive"
    assert torch.allclose(
        original["delta_local"][:, :3], changed["delta_local"][:, :3]
    )


@torch.no_grad()
def test_previous_global_prediction_feeds_next_residual_step() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    embedding = torch.randn(2, 8)
    future = torch.randn(2, 4, 6)
    global_prediction = torch.randn(2, 4, 1)
    changed_global = global_prediction.clone()
    changed_global[:, 0, :] += 10.0

    _, original = model._apply_local_residual(global_prediction, embedding, future)
    _, changed = model._apply_local_residual(changed_global, embedding, future)

    assert torch.allclose(original["delta_local"][:, 0], changed["delta_local"][:, 0])
    assert not torch.allclose(original["delta_local"][:, 1], changed["delta_local"][:, 1])


def test_default_hpo_couples_temporal_encoders_and_omits_low_value_axes() -> None:
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    training = GlobalTrainingConfig(
        use_modality_specific_encoders=True,
        hpo_modality_encoder_architecture=True,
        use_auxiliary_autoencoder=True,
        hpo_auxiliary_loss_weights=False,
    )
    candidate = suggest_global_candidate(
        trial,
        "mlp",
        training,
        hpo_config=GlobalHPOConfig(
            modality_encoder_hpo_space=ModalityEncoderHPOSpace().model_dump(mode="python")
        ),
    )

    config = candidate.model_config
    assert config["target_encoder_dim"] == config["historical_encoder_dim"]
    assert config["historical_encoder_dim"] == config["future_encoder_dim"]
    assert config["target_encoder_num_layers"] == config["historical_encoder_num_layers"]
    assert config["historical_encoder_num_layers"] == config["future_encoder_num_layers"]
    assert "weight_decay" not in trial.params
    assert "beta_ae" not in trial.params
    assert "ae_hidden_size" not in trial.params
    assert "event_loss_share_raw" not in trial.params
    assert set(trial.params["activation"] for _ in range(1)) <= {"gelu", "silu"}
