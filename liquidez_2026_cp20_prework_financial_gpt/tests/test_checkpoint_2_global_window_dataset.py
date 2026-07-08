from __future__ import annotations

import unittest
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from global_contracts import CROSS_KEY_COLUMN, MODEL_INPUT_FIELDS
from global_data import (
    ContextScale,
    ContextScaler,
    GlobalSeriesSplit,
    GlobalWindowDataset,
    SeriesBalancedSampler,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_global_long(lengths: tuple[int, ...] = (8, 8, 8, 8, 8, 8)) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for series_number, length in enumerate(lengths):
        series_type = "saldo" if series_number % 2 == 0 else "variacion"
        account_currency_id = f"ACC{series_number:02d}_MXN"
        cross_key_id = f"{account_currency_id}_{series_type}"
        for day in range(length):
            rows.append(
                {
                    "fecha": date(2026, 1, 1) + timedelta(days=day),
                    "account_currency_id": account_currency_id,
                    "cross_key_id": cross_key_id,
                    "tipo_serie": series_type,
                    "target": float(series_number * 100 + day),
                    "difficulty_score": 0.5 + series_number * 0.01,
                    "nivel_curriculum": 1 + series_number,
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


def sample_calendar(days: int = 12) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fecha": [f"2026-01-{day + 1:02d}" for day in range(days)],
            "es_quincena": [1.0 if day in (14, 29) else 0.0 for day in range(days)],
            "dia_semana": [float(day % 7) for day in range(days)],
        }
    ).with_columns(pl.col("fecha").str.to_date())


class TestCheckpoint2GlobalWindowDataset(unittest.TestCase):
    def test_context_scaler_uses_only_context_and_is_reversible(self) -> None:
        scaler = ContextScaler(min_scale=1e-6)
        context = np.asarray([10.0, 20.0, 30.0, 40.0])
        parameters = scaler.fit(context)

        future = np.asarray([1_000_000.0])
        transformed = scaler.transform(future, parameters)
        restored = scaler.inverse_transform(transformed, parameters)

        self.assertEqual(parameters.center, 0.0)
        self.assertAlmostEqual(parameters.scale, float(np.mean(np.abs(context))))
        np.testing.assert_allclose(restored, future)
        self.assertLess(parameters.center, 1_000.0)

    def test_constant_context_has_finite_minimum_scale(self) -> None:
        parameters = ContextScaler(min_scale=0.25).fit(np.ones(5) * 7.0)
        self.assertEqual(parameters.center, 0.0)
        self.assertAlmostEqual(parameters.scale, 7.0)

    def test_global_series_split_is_deterministic_and_unseen_is_disjoint(self) -> None:
        frame = sample_global_long()
        first = GlobalSeriesSplit.create(
            frame,
            validation_unseen_fraction=0.2,
            test_unseen_fraction=0.2,
            seed=7,
        )
        second = GlobalSeriesSplit.create(
            frame,
            validation_unseen_fraction=0.2,
            test_unseen_fraction=0.2,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first.validation_seen_series), set(first.train_series))
        self.assertTrue(set(first.train_series).isdisjoint(first.validation_unseen_series))
        self.assertTrue(set(first.train_series).isdisjoint(first.test_unseen_series))
        self.assertTrue(
            set(first.validation_unseen_series).isdisjoint(first.test_unseen_series)
        )

    def test_split_filter_returns_only_requested_series(self) -> None:
        frame = sample_global_long()
        split = GlobalSeriesSplit.create(frame, seed=11)
        filtered = split.filter_frame(frame, "test_unseen")
        self.assertEqual(
            set(filtered.get_column(CROSS_KEY_COLUMN).unique().to_list()),
            set(split.test_unseen_series),
        )

    def test_window_shapes_alignment_and_metadata_isolation(self) -> None:
        frame = sample_global_long(lengths=(8, 8, 8))
        dataset = GlobalWindowDataset(
            frame,
            window_size=3,
            horizon=2,
            exogenous=sample_calendar(),
            exogenous_columns=("es_quincena", "dia_semana"),
        )
        sample = dataset[0]
        inputs = sample["model_inputs"]
        targets = sample["targets"]
        metadata = sample["metadata"]

        self.assertEqual(tuple(inputs), MODEL_INPUT_FIELDS)
        self.assertEqual(tuple(inputs["y_context"].shape), (3, 1))
        self.assertEqual(tuple(inputs["x_history"].shape), (3, 2))
        self.assertEqual(tuple(inputs["x_future"].shape), (2, 2))
        self.assertEqual(tuple(inputs["x_static"].shape), (dataset.static_dim,))
        self.assertNotIn("context_mask", inputs)
        self.assertEqual(tuple(targets["y_future"].shape), (2, 1))
        self.assertNotIn(CROSS_KEY_COLUMN, inputs)
        self.assertIn(CROSS_KEY_COLUMN, metadata)

        np.testing.assert_allclose(
            inputs["x_history"][:, 1].numpy(),
            np.asarray([0.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(
            inputs["x_future"][:, 1].numpy(),
            np.asarray([3.0, 4.0]),
        )

    def test_window_target_is_scaled_with_context_statistics_only(self) -> None:
        frame = sample_global_long(lengths=(7, 7, 7))
        dataset = GlobalWindowDataset(frame, window_size=3, horizon=2)
        sample = dataset[0]
        raw_future = sample["targets"]["y_future_raw"].numpy()
        scaled_future = sample["targets"]["y_future"].numpy()
        metadata = sample["metadata"]
        restored = ContextScaler.inverse_transform(
            scaled_future,
            ContextScale(
                center=float(metadata["center"]),
                scale=float(metadata["scale"]),
                transform=str(metadata.get("transform", "identity")),
            ),
        )
        np.testing.assert_allclose(restored, raw_future, rtol=1e-5, atol=1e-5)

    def test_each_window_belongs_to_one_series(self) -> None:
        frame = sample_global_long(lengths=(6, 9, 12))
        dataset = GlobalWindowDataset(frame, window_size=3, horizon=2)
        self.assertEqual(set(dataset.series_ids), set(frame[CROSS_KEY_COLUMN].unique()))
        for index in range(len(dataset)):
            sample = dataset[index]
            self.assertIn(sample["metadata"][CROSS_KEY_COLUMN], dataset.series_ids)

    def test_balanced_sampler_does_not_favor_long_series(self) -> None:
        frame = sample_global_long(lengths=(6, 30, 60))
        dataset = GlobalWindowDataset(frame, window_size=3, horizon=2)
        sampler = SeriesBalancedSampler(dataset, num_samples=6000, seed=123)

        counts = Counter(
            dataset[index]["metadata"][CROSS_KEY_COLUMN]
            for index in sampler
        )
        proportions = np.asarray(list(counts.values()), dtype=float) / 6000.0
        self.assertEqual(len(counts), 3)
        self.assertTrue(np.all(np.abs(proportions - (1.0 / 3.0)) < 0.04))

    def test_sampler_is_reproducible_per_epoch(self) -> None:
        dataset = GlobalWindowDataset(sample_global_long(lengths=(8, 8, 8)), window_size=3, horizon=2)
        sampler = SeriesBalancedSampler(dataset, num_samples=20, seed=99)
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(first, second)
        sampler.set_epoch(1)
        third = list(iter(sampler))
        self.assertNotEqual(first, third)

    def test_missing_calendar_dates_are_aligned_and_reported(self) -> None:
        frame = sample_global_long(lengths=(8, 8, 8))
        incomplete_calendar = sample_calendar(days=5)
        dataset = GlobalWindowDataset(
            frame,
            window_size=3,
            horizon=2,
            exogenous=incomplete_calendar,
            exogenous_columns=("dia_semana",),
        )
        self.assertEqual(len(dataset), 3)
        self.assertTrue((dataset.alignment_report["excluded_rows"] == 3).all())
        self.assertTrue((dataset.alignment_report["coverage_ratio"] == 0.625).all())

    def test_series_filter_limits_dataset_without_identity_as_feature(self) -> None:
        frame = sample_global_long(lengths=(8, 8, 8, 8, 8, 8))
        split = GlobalSeriesSplit.create(frame, seed=42)
        dataset = GlobalWindowDataset(
            frame,
            window_size=3,
            horizon=2,
            series_ids=split.train_series,
        )
        self.assertEqual(set(dataset.series_ids), set(split.train_series))
        self.assertTrue(
            all(
                CROSS_KEY_COLUMN not in dataset[index]["model_inputs"]
                for index in range(len(dataset))
            )
        )

    def test_checkpoint_does_not_add_models_or_modify_local_training(self) -> None:
        module = (ROOT / "global_data.py").read_text(encoding="utf-8")
        self.assertNotIn("torch.nn", module)
        self.assertNotIn("optuna", module)

        scientist_source = (ROOT / "scientist.py").read_text(encoding="utf-8")
        manager_source = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertIn("self.models", scientist_source)
        self.assertIn("for serie", manager_source)


if __name__ == "__main__":
    unittest.main()
