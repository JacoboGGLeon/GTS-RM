from __future__ import annotations

import unittest
from pathlib import Path

import optuna
import torch

from global_models import RECONSTRUCTION_FIELD, build_global_model, list_global_models
from global_training import GlobalTrainingConfig, global_forecast_loss, suggest_global_candidate

ROOT = Path(__file__).resolve().parents[1]


def base_config() -> dict[str, object]:
    return {
        "latent_dim": 8,
        "enc_hidden_size": 12,
        "enc_num_layers": 1,
        "dec_hidden_size": 10,
        "dec_num_layers": 1,
        "rnn_hidden_size": 10,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "beta_ae": 0.2,
        "ae_hidden_size": 12,
        "ae_num_layers": 1,
        "beta_kl": 0.01,
    }


def inputs(batch: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(16)
    return {
        "y_context": torch.randn(batch, 5, 1),
        "x_history": torch.randn(batch, 5, 2),
        "x_future": torch.randn(batch, 3, 2),
        "x_static": torch.ones(batch, 5),
    }


class TestCheckpoint16AgnosticAutoencoderRegularization(unittest.TestCase):
    def test_all_architectures_have_one_forecast_head_and_one_auxiliary_decoder(self) -> None:
        batch = inputs()
        for architecture in list_global_models():
            model = build_global_model(
                architecture,
                base_config(),
                window_size=5,
                horizon=3,
                exogenous_dim=2,
                static_dim=5,
            )
            output = model(**batch)
            self.assertNotIn("task_predictions", output, architecture)
            self.assertEqual(tuple(output["y_pred"].shape), (3, 3, 1), architecture)
            self.assertEqual(
                tuple(output[RECONSTRUCTION_FIELD].shape), (3, 5, 1), architecture
            )
            self.assertIn("weighted_reconstruction", output["losses"], architecture)
            self.assertTrue(torch.isfinite(output["losses"]["reconstruction"]), architecture)

    def test_reconstruction_regularizer_is_weighted_by_beta(self) -> None:
        model = build_global_model(
            "mlp",
            base_config(),
            window_size=5,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        batch = inputs()
        output = model(**batch)
        target = torch.zeros(3, 3, 1)
        total = global_forecast_loss(output, target, loss="mse")
        forecast_only = torch.mean(output["y_pred"].square())
        expected = forecast_only + output["losses"]["weighted_reconstruction"]
        torch.testing.assert_close(total, expected)

    def test_beta_zero_disables_regularization_without_removing_head(self) -> None:
        cfg = base_config()
        cfg["beta_ae"] = 0.0
        model = build_global_model(
            "rnn",
            cfg,
            window_size=5,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        output = model(**inputs())
        self.assertIn(RECONSTRUCTION_FIELD, output)
        self.assertEqual(float(output["losses"]["weighted_reconstruction"].detach()), 0.0)

    def test_auxiliary_loss_backpropagates_into_shared_encoder(self) -> None:
        model = build_global_model(
            "mlp",
            base_config(),
            window_size=5,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        batch = inputs()
        output = model(**batch)
        output["losses"]["weighted_reconstruction"].backward()
        encoder_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("encoder.") and parameter.requires_grad
        ]
        self.assertTrue(encoder_gradients)
        self.assertTrue(any(g is not None and torch.any(g != 0) for g in encoder_gradients))

    def test_hpo_tunes_autoencoder_capacity_and_importance(self) -> None:
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        candidate = suggest_global_candidate(trial, "mlp", GlobalTrainingConfig())
        self.assertIn("beta_ae", candidate.model_config)
        self.assertIn("ae_hidden_size", candidate.model_config)
        self.assertIn("ae_num_layers", candidate.model_config)
        self.assertNotIn("task_heads", candidate.model_config)

    def test_vae_keeps_kl_and_autoencoder_regularization_separate(self) -> None:
        model = build_global_model(
            "mlp_vae",
            base_config(),
            window_size=5,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        output = model(**inputs())
        self.assertIn("weighted_kl", output["losses"])
        self.assertIn("weighted_reconstruction", output["losses"])
        self.assertAlmostEqual(float(output["extras"]["beta_kl"]), 0.01)
        self.assertAlmostEqual(float(output["extras"]["beta_ae"]), 0.2)

    def test_series_vocabulary_is_open_and_scaler_ignores_type(self) -> None:
        import numpy as np
        from global_contracts import canonical_cross_key
        from global_data import ContextScaler

        self.assertEqual(
            canonical_cross_key("OBJ_01", "liquidez contractual"),
            "OBJ_01_liquidez_contractual",
        )
        values = np.asarray([-10.0, 0.0, 25.0])
        scaler = ContextScaler()
        self.assertEqual(
            scaler.fit(values, series_type="saldo"),
            scaler.fit(values, series_type="cualquier_serie"),
        )

    def test_schema_version_is_bumped_for_incompatible_weights(self) -> None:
        from global_manager import ARTIFACT_SCHEMA_VERSION
        self.assertEqual(ARTIFACT_SCHEMA_VERSION, "1.5")


if __name__ == "__main__":
    unittest.main()
