from __future__ import annotations

import json
import math
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from global_data import ContextScaler, GlobalWindowDataset
from global_models import (
    GlobalForecastModel,
    GlobalModelDimensions,
    build_global_model,
)
from global_monitoring import MCDropoutConfig, mc_dropout_backtest
from global_training import (
    DEFAULT_OBJECTIVE_METRIC,
    GlobalTrainingConfig,
    SUPPORTED_GLOBAL_LOSSES,
    global_forecast_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def frame(series_type: str, base: float) -> pl.DataFrame:
    account = f"ACC_MXN"
    key = f"{account}_{series_type}"
    rows = []
    for i in range(10):
        value = base + i if series_type == "saldo" else ((-1) ** i) * (base + i)
        rows.append({
            "fecha": date(2026, 1, 1) + timedelta(days=i),
            "account_currency_id": account,
            "cross_key_id": key,
            "tipo_serie": series_type,
            "target": float(value),
            "difficulty_score": 0.7,
            "nivel_curriculum": 1,
            "grupo": "Grupo_2",
        })
    return pl.DataFrame(rows)


class ZeroTaskModel(GlobalForecastModel):
    def __init__(self, window_size: int, horizon: int) -> None:
        super().__init__(GlobalModelDimensions(window_size, horizon, 0, 7))

    def forward(self, y_context, x_history, x_future, x_static):
        batch = y_context.shape[0]
        prediction = torch.zeros(
            (batch, self.dimensions.horizon, 1),
            dtype=y_context.dtype,
            device=y_context.device,
        )
        return {"y_pred": prediction, "extras": {}}


class TestCheckpoint15TrainingAndEvaluationIntegrity(unittest.TestCase):
    def test_every_series_uses_linear_causal_scale_and_is_reversible(self) -> None:
        scaler = ContextScaler()
        values = np.array([0.0, 1e8, -1e8, 5e7], dtype=float)
        params = scaler.fit(values, series_type="liquidez")
        transformed = scaler.transform(values, params)
        restored = scaler.inverse_transform(transformed, params)
        self.assertEqual(params.transform, "linear_context_scale")
        self.assertEqual(params.center, 0.0)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-3)

    def test_zero_context_has_a_stable_linear_scale_floor(self) -> None:
        scaler = ContextScaler()
        params = scaler.fit(np.zeros(8), series_type="contratos")
        transformed = scaler.transform(np.asarray([1e8, -1e8]), params)
        self.assertGreaterEqual(params.scale, 1.0)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(params.scale, 1.0)

    def test_global_models_use_one_series_agnostic_forecast_head(self) -> None:
        cfg = {
            "latent_dim": 8,
            "enc_hidden_size": 12,
            "enc_num_layers": 1,
            "dec_hidden_size": 10,
            "dec_num_layers": 1,
            "dropout_rate": 0.0,
            "activation": "gelu",
            "beta_ae": 0.1,
            "ae_hidden_size": 8,
            "ae_num_layers": 1,
        }
        model = build_global_model(
            "mlp", cfg, window_size=3, horizon=2, exogenous_dim=0
        ).eval()
        inputs = {
            "y_context": torch.randn(2, 3, 1),
            "x_history": torch.empty(2, 3, 0),
            "x_future": torch.empty(2, 2, 0),
            "x_static": torch.ones(2, 1),
        }
        output = model(**inputs)
        self.assertNotIn("task_predictions", output)
        self.assertEqual(tuple(output["y_pred"].shape), (2, 2, 1))
        self.assertEqual(tuple(output["context_reconstruction"].shape), (2, 3, 1))

    def test_all_original_losses_are_supported(self) -> None:
        target = torch.tensor([[[1.0]], [[-2.0]]])
        saldo = torch.tensor([[[0.5]], [[0.0]]], requires_grad=True)
        output = {"y_pred": saldo, "extras": {}}
        self.assertEqual(
            SUPPORTED_GLOBAL_LOSSES,
            ("rmse", "mae", "mse", "smape", "wmape", "log_cosh", "huber"),
        )
        for name in SUPPORTED_GLOBAL_LOSSES:
            value = global_forecast_loss(
                output,
                target,
                loss=name,
            )
            self.assertEqual(value.ndim, 0, name)
            self.assertTrue(torch.isfinite(value), name)

    def test_selection_metric_is_shared_by_productive_training(self) -> None:
        config = GlobalTrainingConfig()
        self.assertEqual(DEFAULT_OBJECTIVE_METRIC, "robust_macro_mase")
        self.assertEqual(config.selection_metric, "robust_macro_mase")

    def test_backtest_metrics_use_test_only_and_report_interval_quality(self) -> None:
        train = GlobalWindowDataset(frame("saldo", 1.0), window_size=3, horizon=1)
        test = GlobalWindowDataset(frame("saldo", 1000.0), window_size=3, horizon=1)
        result = mc_dropout_backtest(
            ZeroTaskModel(window_size=3, horizon=1),
            train,
            test,
            config=MCDropoutConfig(n_mc=2, batch_size=4, device="cpu"),
        )
        metrics = result["df_regression_metrics"].iloc[0]
        self.assertEqual(metrics["evaluation_scope"], "test_only")
        self.assertIn("PICP", metrics.index)
        self.assertIn("MPIW", metrics.index)
        self.assertIn("Winkler", metrics.index)
        test_rows = result["df_regression"].loc[
            ~result["df_regression"]["isTrain"].astype(bool)
        ]
        manual_mae = float(np.mean(np.abs(
            test_rows["actual_orig"].to_numpy() - test_rows["pred_orig"].to_numpy()
        )))
        self.assertAlmostEqual(float(metrics["MAE"]), manual_mae, places=10)
        self.assertIn("ACC_MXN_saldo", result["train_bounds"])

    def test_four_global_notebooks_expose_original_style_configuration(self) -> None:
        names = (
            "code_03_GLOBAL_MLP_E_D.ipynb",
            "code_03_GLOBAL_MLP_VaE_D.ipynb",
            "code_03_GLOBAL_RNN_E_D.ipynb",
            "code_03_GLOBAL_RNNBi_E_D.ipynb",
        )
        required = (
            "N_MONTE_CARLO",
            "HPO_TRIALS",
            "HPO_EPOCHS",
            "WARM_EPOCHS",
            "WARM_BATCH",
            "FINE_EPOCHS",
            "FINE_BATCH",
            "LOSS_FUNCTION",
            "SELECTION_METRIC",
            "SHOW_PLOTS",
        )
        for name in names:
            notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
            )
            for token in required:
                self.assertIn(token, source, f"{token} missing from {name}")
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
