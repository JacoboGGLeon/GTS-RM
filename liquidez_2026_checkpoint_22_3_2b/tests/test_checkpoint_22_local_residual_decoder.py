from __future__ import annotations

import pytest
import torch

from global_contracts import GLOBAL_COMPONENT_FIELD, LOCAL_RESIDUAL_FIELD
from global_models import GLOBAL_OUTPUT_FIELD, build_global_model, validate_global_model_output
from gtrm_config import GTRMModelConfig


def _model_cfg(*, use_local: bool) -> dict[str, object]:
    return {
        "latent_dim": 5,
        "enc_hidden_size": 8,
        "enc_num_layers": 1,
        "dec_hidden_size": 8,
        "dec_num_layers": 1,
        "rnn_hidden_size": 6,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "use_auxiliary_autoencoder": False,
        "use_local_residual_decoder": use_local,
        "local_residual_lambda": 0.05,
        "global_aux_alpha": 0.2,
        "local_residual_hidden_size": 7,
        "local_residual_num_layers": 1,
        "local_residual_dropout_rate": 0.0,
    }


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "y_context": torch.randn(batch, 4, 1),
        "x_history": torch.randn(batch, 4, 2),
        "x_future": torch.randn(batch, 3, 2),
        "x_static": torch.randn(batch, 3),
    }


@torch.no_grad()
def test_local_residual_decoder_outputs_final_equals_global_plus_delta() -> None:
    model = build_global_model(
        "rnn",
        _model_cfg(use_local=True),
        window_size=4,
        horizon=3,
        exogenous_dim=2,
        static_dim=3,
    )
    output = model(**_inputs())
    embedding = validate_global_model_output(output, batch_size=2, horizon=3, latent_dim=5)
    extras = output["extras"]
    y_global = extras[GLOBAL_COMPONENT_FIELD]
    delta_local = extras[LOCAL_RESIDUAL_FIELD]
    assert embedding.shape == (2, 5)
    assert y_global.shape == output[GLOBAL_OUTPUT_FIELD].shape
    assert delta_local.shape == output[GLOBAL_OUTPUT_FIELD].shape
    assert torch.allclose(output[GLOBAL_OUTPUT_FIELD], y_global + delta_local)
    assert extras["use_local_residual_decoder"] is True


@torch.no_grad()
def test_global_only_path_does_not_emit_local_tensors() -> None:
    model = build_global_model(
        "mlp",
        _model_cfg(use_local=False),
        window_size=4,
        horizon=3,
        exogenous_dim=2,
        static_dim=3,
    )
    output = model(**_inputs())
    extras = output["extras"]
    assert extras["use_local_residual_decoder"] is False
    assert GLOBAL_COMPONENT_FIELD not in extras
    assert LOCAL_RESIDUAL_FIELD not in extras


def test_global_forecast_loss_includes_residual_and_global_auxiliary_terms() -> None:
    pytest.importorskip("polars")
    from global_training import global_forecast_loss

    target = torch.zeros(1, 2, 1)
    y_global = torch.ones(1, 2, 1) * 2.0
    delta_local = torch.ones(1, 2, 1) * -1.0
    y_pred = y_global + delta_local
    output = {
        GLOBAL_OUTPUT_FIELD: y_pred,
        "extras": {
            GLOBAL_COMPONENT_FIELD: y_global,
            LOCAL_RESIDUAL_FIELD: delta_local,
            "local_residual_lambda": 0.5,
            "global_aux_alpha": 0.25,
        },
    }
    # MAE final = 1.0; residual penalty = 0.5*1.0; global aux = 0.25*2.0
    loss = global_forecast_loss(output, target, loss="mae")
    assert torch.allclose(loss, torch.tensor(2.0))


def test_gtrm_stage2_allows_local_residual_but_rejects_future_heads() -> None:
    cfg = GTRMModelConfig(use_local_residual_decoder=True)
    cfg.validate(stage=2)
    bad = GTRMModelConfig(use_local_residual_decoder=True, use_quantile_head=True)
    try:
        bad.validate(stage=2)
    except ValueError as exc:
        assert "Stage 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Stage 2 must still reject quantile head")


def test_training_config_carries_local_residual_regularization() -> None:
    pytest.importorskip("polars")
    from global_training import GlobalTrainingConfig

    cfg = GlobalTrainingConfig(
        use_local_residual_decoder=True,
        local_residual_lambda=0.01,
        global_aux_alpha=0.2,
        local_residual_hidden_size=16,
        local_residual_num_layers=1,
    )
    cfg.validate()
