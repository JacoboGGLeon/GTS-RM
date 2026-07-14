from __future__ import annotations

import math
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import torch

from global_curriculum import (
    CurriculumReplaySampler,
    GlobalCurriculumConfig,
    GlobalCurriculumTrainer,
    state_dict_digest,
)
from global_data import GlobalWindowDataset
from global_training import GlobalDatasetBundle, GlobalTrainingConfig


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
    series_levels: Mapping[int, int],
    *,
    start_day: int,
    length: int,
) -> list[dict[str, object]]:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for series_number, level in series_levels.items():
        account_currency_id = f"ACC{series_number:02d}_MXN"
        series_type = "saldo" if series_number % 2 == 0 else "variacion"
        cross_key_id = f"{account_currency_id}_{series_type}"
        for offset in range(length):
            day_number = start_day + offset
            rows.append(
                {
                    "fecha": origin + timedelta(days=day_number),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(series_number * 10 + day_number + 1),
                    "difficulty_score": float(level) / 3.0,
                    "nivel_curriculum": level,
                    "grupo": "Grupo_2",
                }
            )
    return rows


def make_calendar(total_days: int = 50) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "fecha": [origin + timedelta(days=day) for day in range(total_days)],
            "es_quincena": [float(day % 15 == 14) for day in range(total_days)],
            "dia_habil": [float(day % 7 < 5) for day in range(total_days)],
        }
    )


def make_bundle(
    *,
    levels: Mapping[int, int] | None = None,
    unseen_levels: Mapping[int, int] | None = None,
    window_size: int = 3,
    horizon: int = 2,
) -> GlobalDatasetBundle:
    levels = levels or {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
    unseen_levels = unseen_levels or {6: 1, 7: 2, 8: 3}
    calendar = make_calendar()
    exogenous_columns = ("es_quincena", "dia_habil")
    train = GlobalWindowDataset(
        pl.DataFrame(make_rows(levels, start_day=0, length=10)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_seen = GlobalWindowDataset(
        pl.DataFrame(make_rows(levels, start_day=14, length=8)),
        window_size=window_size,
        horizon=horizon,
        exogenous=calendar,
        exogenous_columns=exogenous_columns,
    )
    validation_unseen = GlobalWindowDataset(
        pl.DataFrame(make_rows(unseen_levels, start_day=14, length=8)),
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
        "patience": 2,
        "scheduler_patience": 1,
        "samples_per_epoch": 12,
        "num_workers": 0,
        "seed": 17,
        "device": "cpu",
    }
    values.update(changes)
    return GlobalTrainingConfig(**values)


def tiny_curriculum_config(**changes: Any) -> GlobalCurriculumConfig:
    values: dict[str, Any] = {
        "warmup_epochs": 1,
        "fine_tune_epochs_per_level": 1,
        "consolidation_epochs": 1,
        "replay_fraction": 0.35,
        "fine_tune_lr_factor": 0.2,
        "consolidation_lr_factor": 0.1,
    }
    values.update(changes)
    return GlobalCurriculumConfig(**values)


class TestCheckpoint5CurriculumLearningWithReplay(unittest.TestCase):
    def test_dataset_exposes_curriculum_only_as_series_metadata(self) -> None:
        dataset = make_bundle().train
        self.assertEqual(set(dataset.series_curriculum_levels.values()), {1, 2, 3})
        sample = dataset[0]
        self.assertIn("nivel_curriculum", sample["metadata"])
        self.assertNotIn("nivel_curriculum", sample["model_inputs"])
        self.assertNotIn("cross_key_id", sample["model_inputs"])

    def test_curriculum_metadata_must_be_constant_within_each_series(self) -> None:
        rows = make_rows({0: 1}, start_day=0, length=8)
        rows[-1]["nivel_curriculum"] = 2
        with self.assertRaisesRegex(ValueError, "time-varying metadata"):
            GlobalWindowDataset(
                pl.DataFrame(rows),
                window_size=3,
                horizon=2,
                exogenous=make_calendar(),
                exogenous_columns=("es_quincena", "dia_habil"),
            )

    def test_curriculum_builds_one_ordered_path(self) -> None:
        stages = tiny_curriculum_config().build_stages([3, 1, 2])
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "warmup_level_1",
                "finetune_level_2",
                "finetune_level_3",
                "consolidation_all_levels",
            ],
        )
        self.assertEqual(stages[1].replay_levels, (1,))
        self.assertEqual(stages[2].replay_levels, (1, 2))
        self.assertEqual(stages[0].learning_rate_factor, 1.0)
        self.assertLess(stages[1].learning_rate_factor, 1.0)

    def test_replay_sampler_is_series_balanced_and_respects_level_pools(self) -> None:
        dataset = make_bundle().train
        sampler = CurriculumReplaySampler(
            dataset,
            current_levels=(3,),
            replay_levels=(1, 2),
            replay_fraction=0.35,
            num_samples=200,
            seed=23,
        )
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 200)
        self.assertGreater(sampler.last_draw_counts["current"], 0)
        self.assertGreater(sampler.last_draw_counts["replay"], 0)
        self.assertEqual(sum(sampler.last_draw_counts.values()), 200)
        observed_levels = {
            int(dataset[index]["metadata"]["nivel_curriculum"])
            for index in indices
        }
        self.assertEqual(observed_levels, {1, 2, 3})

        same = CurriculumReplaySampler(
            dataset,
            current_levels=(3,),
            replay_levels=(1, 2),
            replay_fraction=0.35,
            num_samples=200,
            seed=23,
        )
        self.assertEqual(indices, list(iter(same)))

    def test_warmup_and_finetuning_keep_one_continuous_model(self) -> None:
        result = GlobalCurriculumTrainer(
            "mlp",
            SMALL_MODEL_CONFIGS["mlp"],
            tiny_training_config(),
            tiny_curriculum_config(),
        ).fit(make_bundle())

        self.assertEqual(len(result.stages), 4)
        self.assertEqual(result.total_epochs, 4)
        self.assertFalse(isinstance(result.model, dict))
        self.assertEqual(state_dict_digest(result.model.state_dict()), result.stages[-1].end_state_digest)
        for previous, current in zip(result.stages, result.stages[1:]):
            self.assertEqual(previous.end_state_digest, current.start_state_digest)

        warmup, level_two, level_three, consolidation = result.stages
        self.assertEqual(warmup.stage.phase, "warmup")
        self.assertEqual(warmup.history[0].replay_samples, 0)
        self.assertGreater(level_two.history[0].replay_samples, 0)
        self.assertGreater(level_three.history[0].replay_samples, 0)
        self.assertEqual(consolidation.history[0].replay_samples, 0)
        self.assertAlmostEqual(warmup.history[0].learning_rate, 1e-3, places=8)
        self.assertAlmostEqual(level_two.history[0].learning_rate, 2e-4, places=8)
        self.assertAlmostEqual(level_three.history[0].learning_rate, 2e-4, places=8)
        self.assertAlmostEqual(consolidation.history[0].learning_rate, 1e-4, places=8)
        self.assertTrue(math.isfinite(result.best_score))
        self.assertEqual(set(result.validation), {"validation_seen", "validation_unseen"})

    def test_training_is_reproducible_with_replay(self) -> None:
        bundle = make_bundle()
        first = GlobalCurriculumTrainer(
            "mlp",
            SMALL_MODEL_CONFIGS["mlp"],
            tiny_training_config(seed=91),
            tiny_curriculum_config(),
        ).fit(bundle)
        second = GlobalCurriculumTrainer(
            "mlp",
            SMALL_MODEL_CONFIGS["mlp"],
            tiny_training_config(seed=91),
            tiny_curriculum_config(),
        ).fit(bundle)
        self.assertAlmostEqual(first.best_score, second.best_score, places=7)
        for name, tensor in first.model.state_dict().items():
            torch.testing.assert_close(tensor, second.model.state_dict()[name])

    def test_all_four_architectures_support_curriculum_contract(self) -> None:
        bundle = make_bundle(
            levels={0: 1, 1: 1},
            unseen_levels={6: 1},
        )
        curriculum = tiny_curriculum_config(consolidation_epochs=0)
        training = tiny_training_config(samples_per_epoch=4, batch_size=2)
        for architecture, model_config in SMALL_MODEL_CONFIGS.items():
            result = GlobalCurriculumTrainer(
                architecture,
                model_config,
                training,
                curriculum,
            ).fit(bundle)
            self.assertEqual(result.architecture, architecture)
            self.assertEqual(len(result.stages), 1)
            self.assertEqual(result.stages[0].stage.phase, "warmup")
            self.assertTrue(math.isfinite(result.best_score))

    def test_checkpoint_five_does_not_add_manager_notebook_or_s3_logic(self) -> None:
        source = (ROOT / "global_curriculum.py").read_text(encoding="utf-8")
        self.assertNotIn("boto3", source)
        self.assertNotIn("s3://", source)
        self.assertNotIn("papermill", source)
        self.assertNotIn("class GlobalManager", source)

        for notebook_name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            notebook = (ROOT / notebook_name).read_text(encoding="utf-8")
            self.assertNotIn("global_curriculum", notebook)


if __name__ == "__main__":
    unittest.main()
