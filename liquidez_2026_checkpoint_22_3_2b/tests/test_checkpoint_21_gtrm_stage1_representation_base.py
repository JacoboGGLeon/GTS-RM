from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import torch
from torch.utils.data import DataLoader

from global_contracts import (
    DEFAULT_GTRM_STAGE1_FLAGS,
    HISTORY_EMBEDDING_FIELD,
    MODEL_INPUT_FIELDS,
    default_gtrm_stage1_flags,
    validate_gtrm_stage_flags,
)
from global_data import GlobalWindowDataset
from global_models import (
    GLOBAL_OUTPUT_FIELD,
    build_global_model,
    get_history_embedding,
    list_global_models,
    validate_global_model_output,
)
from gtrm_representation import (
    GTRMStage1Config,
    collect_history_embeddings,
    gtrm_stage1_manifest,
)


def _frame(series_count: int = 4, length: int = 8) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    currencies = ("MXN", "USD")
    for number in range(series_count):
        series_type = "saldo" if number % 2 == 0 else "variacion"
        currency = currencies[number % len(currencies)]
        account = f"ACC{number:02d}{currency}"
        for offset in range(length):
            rows.append(
                {
                    "fecha": origin + timedelta(days=offset),
                    "account_currency_id": account,
                    "divisa": currency,
                    "cross_key_id": f"{account}_{series_type}",
                    "tipo_serie": series_type,
                    "series_age_step": offset + 1,
                    "target": float(10 + number + offset),
                    "difficulty_score": float(number % 3) / 3.0,
                    "nivel_curriculum": 1 + (number % 2),
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


def _model_cfg() -> dict[str, object]:
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
    }


def test_stage1_flags_are_explicit_and_future_heads_start_disabled() -> None:
    assert dict(DEFAULT_GTRM_STAGE1_FLAGS)["use_static_context"] is True
    flags = default_gtrm_stage1_flags()
    assert flags == {
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
    assert validate_gtrm_stage_flags({"use_static_context": False})["use_static_context"] is False
    GTRMStage1Config().validate_stage1_only()
    try:
        GTRMStage1Config(use_local_residual_decoder=True).validate_stage1_only()
    except ValueError as exc:
        assert "Stage 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Stage 1 must reject local residual activation")


def test_static_context_can_be_disabled_without_changing_forward_contract() -> None:
    dataset = GlobalWindowDataset(
        _frame(),
        window_size=3,
        horizon=2,
        use_static_context=False,
    )
    sample = dataset[0]
    assert tuple(sample["model_inputs"]) == MODEL_INPUT_FIELDS
    assert dataset.static_dim == 1
    assert dataset.static_feature_names == ("static_context_disabled",)
    assert torch.allclose(sample["model_inputs"]["x_static"], torch.zeros(1))


@torch.no_grad()
def test_all_global_architectures_return_valid_stage1_history_embedding() -> None:
    dataset = GlobalWindowDataset(_frame(), window_size=3, horizon=2)
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) for key, value in sample["model_inputs"].items()}
    for architecture in list_global_models():
        model = build_global_model(
            architecture,
            _model_cfg(),
            window_size=dataset.window_size,
            horizon=dataset.horizon,
            exogenous_dim=len(dataset.exogenous_columns),
            static_dim=dataset.static_dim,
        )
        output = model(**batch)
        embedding = validate_global_model_output(
            output,
            batch_size=1,
            horizon=dataset.horizon,
            latent_dim=5,
        )
        assert output[GLOBAL_OUTPUT_FIELD].shape == (1, dataset.horizon, 1)
        assert get_history_embedding(output).shape == embedding.shape
        assert output["extras"][HISTORY_EMBEDDING_FIELD].shape == (1, 5)
        contract = model.representation_contract()
        assert contract["latent_field"] == HISTORY_EMBEDDING_FIELD
        assert contract["series_identity_in_forward"] is False


def test_collect_history_embeddings_exports_metadata_and_latent_columns() -> None:
    dataset = GlobalWindowDataset(_frame(), window_size=3, horizon=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = build_global_model(
        "mlp",
        _model_cfg(),
        window_size=dataset.window_size,
        horizon=dataset.horizon,
        exogenous_dim=len(dataset.exogenous_columns),
        static_dim=dataset.static_dim,
    )
    frame = collect_history_embeddings(model, loader, max_batches=1)
    assert len(frame) == 2
    assert "cross_key_id" in frame.columns
    assert "cutoff" in frame.columns
    latent_columns = [c for c in frame.columns if c.startswith(f"{HISTORY_EMBEDDING_FIELD}_")]
    assert len(latent_columns) == 5
    assert set(frame["embedding_dim"]) == {5}


def test_gtrm_stage1_manifest_names_acceptance_metrics_and_inputs() -> None:
    manifest = gtrm_stage1_manifest(use_static_context=True)
    assert manifest["checkpoint"] == 21
    assert manifest["model_inputs"] == MODEL_INPUT_FIELDS
    assert manifest["latent_field"] == HISTORY_EMBEDDING_FIELD
    assert manifest["flags"]["use_static_context"] is True
    assert "robust_macro_mase" in manifest["acceptance_metrics"]
    assert "percent_series_improved" in manifest["acceptance_metrics"]
