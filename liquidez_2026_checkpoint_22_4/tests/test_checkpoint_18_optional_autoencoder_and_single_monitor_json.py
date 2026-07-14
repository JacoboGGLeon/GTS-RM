from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from datetime import date

import optuna
import polars as pl
import torch

from financial_gpt_monitor import FinancialGPTMonitorResult
from global_models import RECONSTRUCTION_FIELD, build_global_model, list_global_models
from global_training import GlobalTrainingConfig, global_forecast_loss, suggest_global_candidate


class TestOptionalAutoencoderAndSingleMonitorJson(unittest.TestCase):
    def _inputs(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(18)
        return {
            "y_context": torch.randn(4, 5, 1),
            "x_history": torch.randn(4, 5, 3),
            "x_future": torch.randn(4, 2, 3),
            "x_static": torch.ones(4, 4),
        }

    def _base_model_config(self, enabled: bool) -> dict[str, object]:
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
            "use_auxiliary_autoencoder": enabled,
        }

    def test_autoencoder_flag_disables_head_and_loss_for_all_architectures(self) -> None:
        target = torch.zeros(4, 2, 1)
        for architecture in list_global_models():
            model = build_global_model(
                architecture,
                self._base_model_config(False),
                window_size=5,
                horizon=2,
                exogenous_dim=3,
                static_dim=4,
            )
            output = model(**self._inputs())
            self.assertNotIn(RECONSTRUCTION_FIELD, output, architecture)
            self.assertNotIn("weighted_reconstruction", output.get("losses", {}), architecture)
            self.assertFalse(output["extras"]["use_auxiliary_autoencoder"], architecture)
            self.assertEqual(float(output["extras"]["beta_ae"]), 0.0, architecture)
            total = global_forecast_loss(output, target, loss="mse")
            self.assertTrue(torch.isfinite(total), architecture)
            if architecture == "mlp_vae":
                self.assertIn("weighted_kl", output["losses"])

    def test_autoencoder_flag_keeps_current_behavior_when_enabled(self) -> None:
        for architecture in list_global_models():
            model = build_global_model(
                architecture,
                self._base_model_config(True),
                window_size=5,
                horizon=2,
                exogenous_dim=3,
                static_dim=4,
            )
            output = model(**self._inputs())
            self.assertIn(RECONSTRUCTION_FIELD, output, architecture)
            self.assertIn("weighted_reconstruction", output["losses"], architecture)
            self.assertTrue(output["extras"]["use_auxiliary_autoencoder"], architecture)

    def test_hpo_omits_autoencoder_search_space_when_disabled(self) -> None:
        study = optuna.create_study(direction="minimize")
        candidate = suggest_global_candidate(
            study.ask(),
            "rnn",
            GlobalTrainingConfig(use_auxiliary_autoencoder=False),
        )
        self.assertFalse(candidate.model_config["use_auxiliary_autoencoder"])
        self.assertNotIn("beta_ae", candidate.model_config)
        self.assertNotIn("ae_hidden_size", candidate.model_config)
        self.assertNotIn("ae_num_layers", candidate.model_config)

    def test_monitor_writes_exactly_one_json_document(self) -> None:
        result = FinancialGPTMonitorResult(
            run_inventory=pl.DataFrame(
                [{"candidate_id": "GLOBAL_RNN_E_D", "family": "global"}]
            ),
            comparison_coverage=pl.DataFrame(
                [{"cross_key_id": "A_saldo", "comparison_points": 50}]
            ),
            metrics_by_series=pl.DataFrame(
                [{"cross_key_id": "A_saldo", "candidate_id": "GLOBAL_RNN_E_D", "MAE": 1.0}]
            ),
            winners_by_series=pl.DataFrame(
                [{"cross_key_id": "A_saldo", "winner_candidate": "GLOBAL_RNN_E_D"}]
            ),
            winner_counts=pl.DataFrame(
                [{"winner_candidate": "GLOBAL_RNN_E_D", "num_series": 1}]
            ),
            ensemble_forecast=pl.DataFrame(
                [{"date": date(2026, 6, 1), "cross_key_id": "A_saldo", "pred_orig": 1.0}]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = result.write(directory)
            self.assertEqual(output.name, "financial_gpt_monitor.json")
            self.assertEqual([path.name for path in Path(directory).iterdir()], [output.name])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.1")
            self.assertEqual(payload["summary"]["series_compared"], 1)
            self.assertEqual(len(payload["metrics_by_series"]), 1)
            self.assertEqual(len(payload["ensemble_forecast"]), 1)


if __name__ == "__main__":
    unittest.main()
