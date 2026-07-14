from __future__ import annotations

import inspect
import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from global_contracts import FORBIDDEN_MODEL_INPUT_FIELDS, MODEL_INPUT_FIELDS
from global_data import GlobalWindowDataset
from global_models import (
    GLOBAL_MODEL_REGISTRY,
    GlobalMLPEncoderDecoder,
    GlobalMLPVAEEncoderDecoder,
    GlobalRNNBiEncoderDecoder,
    GlobalRNNEncoderDecoder,
    build_global_model,
    list_global_models,
    validate_global_forward_contract,
)


ROOT = Path(__file__).resolve().parents[1]


BASE_CFG = {
    "latent_dim": 12,
    "enc_hidden_size": 24,
    "enc_num_layers": 2,
    "dec_hidden_size": 20,
    "dec_num_layers": 2,
    "rnn_hidden_size": 16,
    "rnn_num_layers": 2,
    "decoder_num_layers": 1,
    "dropout_rate": 0.0,
    "activation": "gelu",
    "beta": 0.25,
}


def sample_inputs(
    *,
    batch_size: int = 4,
    window_size: int = 6,
    horizon: int = 3,
    exogenous_dim: int = 2,
    static_dim: int = 5,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "y_context": torch.randn(batch_size, window_size, 1),
        "x_history": torch.randn(batch_size, window_size, exogenous_dim),
        "x_future": torch.randn(batch_size, horizon, exogenous_dim),
        "x_static": torch.randn(batch_size, static_dim),
    }


def sample_global_long() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for series_number in range(3):
        account_currency_id = f"ACC{series_number:02d}_MXN"
        series_type = "saldo" if series_number % 2 == 0 else "variacion"
        cross_key_id = f"{account_currency_id}_{series_type}"
        for day in range(9):
            rows.append(
                {
                    "fecha": date(2026, 1, 1) + timedelta(days=day),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(series_number * 100 + day),
                    "difficulty_score": 0.5,
                    "nivel_curriculum": 1,
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


class TestCheckpoint3GlobalArchitectures(unittest.TestCase):
    def test_registry_matches_contract_and_contains_four_distinct_classes(self) -> None:
        self.assertEqual(list_global_models(), ("mlp", "mlp_vae", "rnn", "rnn_bi"))
        self.assertEqual(len(GLOBAL_MODEL_REGISTRY), 4)
        self.assertEqual(len(set(GLOBAL_MODEL_REGISTRY.values())), 4)

    def test_forward_contract_is_exact_and_has_no_identifiers(self) -> None:
        for model_class in GLOBAL_MODEL_REGISTRY.values():
            fields = validate_global_forward_contract(model_class)
            self.assertEqual(fields, MODEL_INPUT_FIELDS)
            self.assertTrue(set(fields).isdisjoint(FORBIDDEN_MODEL_INPUT_FIELDS))
            constructor_fields = set(inspect.signature(model_class.__init__).parameters)
            self.assertTrue(constructor_fields.isdisjoint(FORBIDDEN_MODEL_INPUT_FIELDS))

    def test_all_architectures_produce_direct_multi_horizon_output(self) -> None:
        inputs = sample_inputs()
        for name in list_global_models():
            model = build_global_model(
                name,
                BASE_CFG,
                window_size=6,
                horizon=3,
                exogenous_dim=2,
                static_dim=5,
            ).eval()
            output = model(**inputs)
            self.assertEqual(tuple(output["y_pred"].shape), (4, 3, 1), name)
            self.assertEqual(
                tuple(output["extras"]["history_embedding"].shape),
                (4, 12),
                name,
            )
            self.assertTrue(torch.isfinite(output["y_pred"]).all(), name)

    def test_dataset_dataloader_batch_flows_into_every_architecture(self) -> None:
        dataset = GlobalWindowDataset(sample_global_long(), window_size=4, horizon=2)
        batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))
        batch_ids = set(batch["metadata"]["cross_key_id"])
        self.assertEqual(len(batch_ids), 1)
        self.assertTrue(batch_ids.issubset(set(dataset.series_ids)))
        for name in list_global_models():
            model = build_global_model(
                name,
                BASE_CFG,
                window_size=4,
                horizon=2,
                exogenous_dim=0,
                static_dim=dataset.static_dim,
            ).eval()
            output = model(**batch["model_inputs"])
            self.assertEqual(tuple(output["y_pred"].shape), (3, 2, 1), name)

    def test_models_accept_zero_exogenous_features(self) -> None:
        inputs = sample_inputs(exogenous_dim=0)
        for name in list_global_models():
            model = build_global_model(
                name,
                BASE_CFG,
                window_size=6,
                horizon=3,
                exogenous_dim=0,
                static_dim=5,
            ).eval()
            prediction = model(**inputs)["y_pred"]
            self.assertEqual(tuple(prediction.shape), (4, 3, 1), name)

    def test_context_mask_is_removed_and_observed_values_remain_informative(self) -> None:
        inputs = sample_inputs(batch_size=2)
        self.assertNotIn("context_mask", inputs)
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["y_context"][:, 1, :] = 1_000_000.0

        for name in list_global_models():
            torch.manual_seed(19)
            model = build_global_model(
                name,
                BASE_CFG,
                window_size=6,
                horizon=3,
                exogenous_dim=2,
                static_dim=5,
            ).eval()
            original_prediction = model(**inputs)["y_pred"]
            changed_prediction = model(**changed)["y_pred"]
            self.assertFalse(torch.allclose(original_prediction, changed_prediction), name)

    def test_future_covariates_receive_gradient(self) -> None:
        inputs = sample_inputs(batch_size=2)
        for name in list_global_models():
            model_inputs = {key: value.clone() for key, value in inputs.items()}
            model_inputs["x_future"].requires_grad_(True)
            model = build_global_model(
                name,
                BASE_CFG,
                window_size=6,
                horizon=3,
                exogenous_dim=2,
                static_dim=5,
            )
            loss = model(**model_inputs)["y_pred"].square().mean()
            loss.backward()
            gradient = model_inputs["x_future"].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_vae_is_deterministic_in_eval_and_exposes_finite_kl(self) -> None:
        inputs = sample_inputs()
        model = GlobalMLPVAEEncoderDecoder(
            BASE_CFG,
            window_size=6,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        ).eval()
        first = model(**inputs)
        second = model(**inputs)
        torch.testing.assert_close(first["y_pred"], second["y_pred"])
        torch.testing.assert_close(
            first["extras"]["history_embedding"], first["extras"]["mu"]
        )
        self.assertEqual(first["losses"]["kl"].ndim, 0)
        self.assertTrue(torch.isfinite(first["losses"]["kl"]))
        self.assertGreaterEqual(float(first["losses"]["kl"].detach()), 0.0)
        torch.testing.assert_close(
            first["losses"]["weighted_kl"],
            BASE_CFG["beta"] * first["losses"]["kl"],
        )

    def test_bidirectionality_is_limited_to_historical_encoder(self) -> None:
        unidirectional = GlobalRNNEncoderDecoder(
            BASE_CFG,
            window_size=6,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        bidirectional = GlobalRNNBiEncoderDecoder(
            BASE_CFG,
            window_size=6,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        self.assertFalse(unidirectional.encoder.bidirectional)
        self.assertTrue(bidirectional.encoder.bidirectional)
        self.assertFalse(unidirectional.decoder.bidirectional)
        self.assertFalse(bidirectional.decoder.bidirectional)

    def test_shape_mismatches_fail_explicitly(self) -> None:
        model = GlobalMLPEncoderDecoder(
            BASE_CFG,
            window_size=6,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        inputs = sample_inputs()
        inputs["x_future"] = torch.randn(4, 4, 2)
        with self.assertRaisesRegex(ValueError, "x_future shape mismatch"):
            model(**inputs)

    def test_checkpoint_is_architecture_only_and_local_models_stay_separate(self) -> None:
        module = (ROOT / "global_models.py").read_text(encoding="utf-8")
        self.assertNotIn("optuna", module.lower())
        self.assertNotIn("DataLoader", module)
        self.assertNotIn("optimizer", module.lower())

        local_models = (ROOT / "models.py").read_text(encoding="utf-8")
        self.assertNotIn("GlobalMLPEncoderDecoder", local_models)
        self.assertIn("MultiHeadMLPModel", local_models)
        self.assertIn("RNNSeq2SeqModel", local_models)

        for notebook_name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            notebook_source = (ROOT / notebook_name).read_text(encoding="utf-8")
            self.assertNotIn("global_models", notebook_source)


if __name__ == "__main__":
    unittest.main()
