from __future__ import annotations

from gtrm_config import GTRMModelConfig
from gtrm_representation import gtrm_stage1_manifest


def test_gtrm_model_config_centralizes_all_stage_flags() -> None:
    cfg = GTRMModelConfig(architecture="RNN", use_static_context=True)
    cfg.validate(stage=1)
    assert cfg.normalized_architecture() == "rnn"
    assert cfg.stage_flags() == {
        "use_static_context": True,
        "use_modality_specific_encoders": False,
        "use_patch_tokenizer": False,
        "use_local_residual_decoder": False,
        "use_event_head": False,
        "use_magnitude_head": False,
        "use_direction_head": False,
        "use_quantile_head": False,
        "use_self_supervised_pretraining": False,
    }
    assert cfg.dataset_kwargs() == {
        "use_static_context": True,
        "event_threshold": 1.0,
        "magnitude_transform": "asinh",
    }
    assert "static_context" in cfg.model_label_suffix()


def test_gtrm_model_config_rejects_future_stage_flags_for_stage1() -> None:
    cfg = GTRMModelConfig(use_local_residual_decoder=True)
    try:
        cfg.validate(stage=1)
    except ValueError as exc:
        assert "Stage 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Stage 1 must reject local residual activation")


def test_gtrm_model_config_can_be_built_from_notebook_globals() -> None:
    cfg = GTRMModelConfig.from_notebook_globals(
        {
            "ARCHITECTURE": "mlp",
            "USE_STATIC_CONTEXT": False,
            "USE_PATCH_TOKENIZER": False,
            "USE_LOCAL_RESIDUAL_DECODER": False,
            "USE_QUANTILE_HEAD": False,
            "USE_SELF_SUPERVISED_PRETRAINING": False,
            "USE_HPO": True,
            "LOSS_FUNCTION": "huber",
        }
    )
    cfg.validate(stage=1)
    assert cfg.normalized_architecture() == "mlp"
    assert cfg.use_static_context is False
    assert cfg.loss_type == "huber"


def test_stage1_manifest_surfaces_single_model_config() -> None:
    cfg = GTRMModelConfig(architecture="rnn", use_static_context=True)
    manifest = gtrm_stage1_manifest(model_config=cfg)
    assert manifest["checkpoint"] == 21
    assert manifest["subcheckpoint"] == "21.1"
    assert manifest["model_config"]["architecture"] == "rnn"
    assert manifest["model_config"]["stage_flags"]["use_static_context"] is True
    assert manifest["flags"] == manifest["model_config"]["stage_flags"]
