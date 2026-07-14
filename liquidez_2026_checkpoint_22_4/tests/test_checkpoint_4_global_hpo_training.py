from __future__ import annotations

import math
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import optuna
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from global_data import GlobalWindowDataset
from global_models import GlobalForecastModel, GlobalModelDimensions
from global_training import (
    GlobalCandidateConfig,
    GlobalDatasetBundle,
    GlobalHPOConfig,
    GlobalHPOTrainer,
    GlobalTrainer,
    GlobalTrainingConfig,
    evaluate_global_model,
    global_forecast_loss,
    suggest_global_candidate,
    validation_objective,
)


ROOT = Path(__file__).resolve().parents[1]


SMALL_MODEL_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "mlp": {
        "latent_dim": 4,
        "enc_hidden_size": 8,
        "enc_num_layers": 1,
        "dec_hidden_size": 8,
        "dec_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
    },
    "mlp_vae": {
        "latent_dim": 4,
        "enc_hidden_size": 8,
        "enc_num_layers": 1,
        "dec_hidden_size": 8,
        "dec_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "beta": 0.01,
    },
    "rnn": {
        "latent_dim": 4,
        "rnn_hidden_size": 6,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "tanh",
    },
    "rnn_bi": {
        "latent_dim": 4,
        "rnn_hidden_size": 6,
        "rnn_num_layers": 1,
        "decoder_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "tanh",
    },
}


def make_rows(
    series_numbers: range,
    *,
    start_day: int,
    length: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    origin = date(2026, 1, 1)
    for series_number in series_numbers:
        account_currency_id = f"ACC{series_number:02d}_MXN"
        series_type = "saldo" if series_number % 2 == 0 else "variacion"
        cross_key_id = f"{account_currency_id}_{series_type}"
        for offset in range(length):
            current_day = start_day + offset
            rows.append(
                {
                    "fecha": origin + timedelta(days=current_day),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(series_number * 10 + current_day + 1),
                    "difficulty_score": 0.4 + series_number * 0.01,
                    "nivel_curriculum": 1 + series_number % 3,
                    "grupo": "Grupo_2",
                }
            )
    return rows


def make_calendar(*, total_days: int = 40) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=day) for day in range(total_days)],
            "es_quincena": [float(day % 15 == 14) for day in range(total_days)],
            "dia_habil": [float(day % 7 < 5) for day in range(total_days)],
        }
    )


def make_bundle(window_size: int = 3, horizon: int = 2) -> GlobalDatasetBundle:
    calendar = make_calendar()
    exogenous_columns = ("es_quincena", "dia_habil")
    train = GlobalWindowDataset(
        pl.DataFrame(make_rows(range(4), start_day=0, length=10)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_seen = GlobalWindowDataset(
        pl.DataFrame(make_rows(range(4), start_day=12, length=8)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_unseen = GlobalWindowDataset(
        pl.DataFrame(make_rows(range(4, 6), start_day=12, length=8)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    return GlobalDatasetBundle(train, validation_seen, validation_unseen)


def tiny_training_config(**changes: Any) -> GlobalTrainingConfig:
    values: dict[str, Any] = {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "loss": "mae",
        "patience": 1,
        "scheduler_patience": 1,
        "samples_per_epoch": 8,
        "num_workers": 0,
        "seed": 13,
        "device": "cpu",
    }
    values.update(changes)
    return GlobalTrainingConfig(**values)


class ZeroForecastModel(GlobalForecastModel):
    def __init__(self) -> None:
        super().__init__(GlobalModelDimensions(window_size=1, horizon=1, exogenous_dim=0))

    def forward(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> dict[str, object]:
        return {"y_pred": torch.zeros(y_context.shape[0], 1, 1)}


class ErrorDataset(Dataset[Mapping[str, object]]):
    def __init__(self) -> None:
        self.rows = [("LONG", 0.0)] * 10 + [("SHORT", 10.0)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        series_id, target = self.rows[index]
        target_tensor = torch.tensor([[target]], dtype=torch.float32)
        return {
            "model_inputs": {
                "y_context": torch.zeros(1, 1),
                "x_history": torch.empty(1, 0),
                "x_future": torch.empty(1, 0),
                "x_static": torch.ones(1),
            },
            "targets": {
                "y_future": target_tensor,
                "y_future_raw": target_tensor,
            },
            "metadata": {
                "cross_key_id": series_id,
                "center": 0.0,
                "scale": 1.0,
            },
        }


class TestCheckpoint4GlobalHPOTraining(unittest.TestCase):
    def test_dataset_bundle_enforces_seen_and_unseen_identity_contract(self) -> None:
        bundle = make_bundle()
        bundle.validate()
        self.assertTrue(
            set(bundle.validation_seen.series_ids).issubset(set(bundle.train.series_ids))
        )
        self.assertTrue(
            set(bundle.validation_unseen.series_ids).isdisjoint(set(bundle.train.series_ids))
        )

        invalid = GlobalDatasetBundle(
            bundle.train,
            bundle.validation_seen,
            bundle.train,
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            invalid.validate()

    def test_all_four_architectures_train_one_shared_model(self) -> None:
        bundle = make_bundle()
        for architecture, model_config in SMALL_MODEL_CONFIGS.items():
            result = GlobalTrainer(
                architecture,
                model_config,
                tiny_training_config(),
            ).fit(bundle)
            self.assertEqual(result.architecture, architecture)
            self.assertEqual(len(result.history), 1)
            self.assertEqual(result.best_epoch, 1)
            self.assertTrue(math.isfinite(result.best_score))
            self.assertEqual(set(result.validation), {"validation_seen", "validation_unseen"})
            self.assertEqual(result.validation["validation_seen"].num_series, 4)
            self.assertEqual(result.validation["validation_unseen"].num_series, 2)
            self.assertFalse(hasattr(result, "models"))
            self.assertFalse(isinstance(result.model, dict))

    def test_training_is_reproducible_with_same_seed(self) -> None:
        bundle = make_bundle()
        config = tiny_training_config(seed=99)
        first = GlobalTrainer("mlp", SMALL_MODEL_CONFIGS["mlp"], config).fit(bundle)
        second = GlobalTrainer("mlp", SMALL_MODEL_CONFIGS["mlp"], config).fit(bundle)
        self.assertAlmostEqual(first.best_score, second.best_score, places=7)
        for name, tensor in first.model.state_dict().items():
            torch.testing.assert_close(tensor, second.model.state_dict()[name])

    def test_macro_metric_does_not_let_long_series_dominate(self) -> None:
        loader = DataLoader(ErrorDataset(), batch_size=4, shuffle=False)
        metrics = evaluate_global_model(ZeroForecastModel(), loader, device="cpu")
        self.assertAlmostEqual(metrics.macro_mae, 5.0, places=6)
        self.assertAlmostEqual(metrics.micro_mae, 10.0 / 11.0, places=6)
        self.assertNotAlmostEqual(metrics.macro_mae, metrics.micro_mae, places=4)
        self.assertEqual(metrics.num_series, 2)

    def test_validation_objective_weights_seen_and_unseen_equally(self) -> None:
        bundle = make_bundle()
        result = GlobalTrainer(
            "mlp", SMALL_MODEL_CONFIGS["mlp"], tiny_training_config()
        ).fit(bundle)
        seen = result.validation["validation_seen"].robust_macro_mase
        unseen = result.validation["validation_unseen"].robust_macro_mase
        self.assertAlmostEqual(
            validation_objective(result.validation),
            (seen + unseen) / 2.0,
            places=7,
        )

    def test_vae_weighted_kl_is_added_once(self) -> None:
        prediction = torch.tensor([[[2.0]]])
        target = torch.tensor([[[1.0]]])
        weighted_kl = torch.tensor(0.25)
        output = {
            "y_pred": prediction,
            "losses": {"kl": torch.tensor(10.0), "weighted_kl": weighted_kl},
        }
        loss = global_forecast_loss(output, target, loss="mae")
        self.assertAlmostEqual(float(loss), 1.25, places=6)

    def test_single_optuna_study_returns_one_final_global_model(self) -> None:
        base_config = tiny_training_config(epochs=1, samples_per_epoch=6)

        def candidate_factory(
            trial: optuna.Trial,
            architecture: str,
            base: GlobalTrainingConfig,
        ) -> GlobalCandidateConfig:
            learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-3])
            return GlobalCandidateConfig(
                window_size=3,
                model_config=SMALL_MODEL_CONFIGS[architecture],
                training_config=GlobalTrainingConfig(
                    **{**base.__dict__, "learning_rate": learning_rate}
                ),
            )

        result = GlobalHPOTrainer(
            "mlp",
            base_training_config=base_config,
            hpo_config=GlobalHPOConfig(
                epochs=1,
                windows_per_series_per_epoch=1,
                validation_windows_per_series=1,
                finalists=1,
                fidelity_epochs=1,
                fidelity_windows_per_series_per_epoch=1,
            ),
            candidate_factory=candidate_factory,
            seed=21,
        ).search_and_fit(make_bundle, n_trials=2, study_name="checkpoint_4_test")

        self.assertEqual(result.architecture, "mlp")
        self.assertEqual(len(result.study.trials), 2)
        self.assertEqual(result.study.study_name, "checkpoint_4_test")
        self.assertEqual(result.best_candidate.window_size, 3)
        self.assertFalse(isinstance(result.training.model, dict))
        self.assertTrue(math.isfinite(result.training.best_score))

    def test_default_search_space_covers_each_architecture(self) -> None:
        common_params = {
            "window_size": 5,
            "latent_dim": 32,
            "dropout_rate": 0.1,
            "activation": "gelu",
            "learning_rate": 1e-3,
        }
        for architecture in SMALL_MODEL_CONFIGS:
            params = dict(common_params)
            if architecture in {"mlp", "mlp_vae"}:
                params.update(
                    {
                        "mlp_hidden_size": 64,
                        "mlp_num_layers": 1,
                    }
                )
                if architecture == "mlp_vae":
                    params["beta_kl"] = 0.01
            else:
                params.update(
                    {
                        "rnn_hidden_size": 64,
                        "rnn_num_layers": 1,
                    }
                )
            candidate = suggest_global_candidate(
                optuna.trial.FixedTrial(params),
                architecture,
                tiny_training_config(),
            )
            candidate.validate()
            self.assertEqual(candidate.window_size, 5)
            self.assertIn("latent_dim", candidate.model_config)

    def test_checkpoint_four_contains_no_curriculum_or_runtime_or_notebook_logic(self) -> None:
        source = (ROOT / "global_training.py").read_text(encoding="utf-8")
        self.assertNotIn("nivel_curriculum", source)
        self.assertNotIn("difficulty_score", source)
        self.assertNotIn("boto3", source)
        self.assertNotIn("s3://", source)
        self.assertNotIn("papermill", source)

        for notebook_name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            notebook = (ROOT / notebook_name).read_text(encoding="utf-8")
            self.assertNotIn("global_training", notebook)


if __name__ == "__main__":
    unittest.main()
