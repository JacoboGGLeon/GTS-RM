from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import torch

from financial_gpt_monitor import (
    PRIMARY_WINNER_METRIC,
    _causal_mase_scale_from_backtest,
    _rank_candidates,
)
from global_curriculum import GlobalCurriculumConfig
from global_data import GlobalBalancedSampler, GlobalWindowDataset, robust_mase_scale
from global_models import build_global_model
from global_training import (
    DEFAULT_OBJECTIVE_METRIC,
    GlobalCandidateConfig,
    GlobalDatasetBundle,
    GlobalEpochRecord,
    GlobalHPOConfig,
    GlobalHPOTrainer,
    GlobalTrainer,
    GlobalTrainingConfig,
    GlobalTrainingResult,
    GlobalValidationMetrics,
)

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_NOTEBOOKS = (
    "code_03_GLOBAL_MLP_E_D.ipynb",
    "code_03_GLOBAL_MLP_VaE_D.ipynb",
    "code_03_GLOBAL_RNN_E_D.ipynb",
    "code_03_GLOBAL_RNNBi_E_D.ipynb",
)


def _frame(series_count: int = 8, length: int = 10) -> pl.DataFrame:
    origin = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    for number in range(series_count):
        series_type = "saldo" if number % 2 == 0 else "variacion"
        level = 1 + ((number // 2) % 2)
        account = f"ACC{number:02d}_MXP"
        for offset in range(length):
            rows.append(
                {
                    "fecha": origin + timedelta(days=offset),
                    "account_currency_id": account,
                    "cross_key_id": f"{account}_{series_type}",
                    "tipo_serie": series_type,
                    "target": float(10 + number + offset),
                    "difficulty_score": float(level) / 2.0,
                    "nivel_curriculum": level,
                    "grupo": "Grupo_2",
                }
            )
    return pl.DataFrame(rows)


def _bundle() -> GlobalDatasetBundle:
    frame = _frame(series_count=6, length=10)
    ids = frame["cross_key_id"].unique().sort().to_list()
    train_ids, unseen_ids = ids[:4], ids[4:]
    train = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(train_ids)),
        window_size=3,
        horizon=2,
    )
    seen = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(train_ids)),
        window_size=3,
        horizon=2,
    )
    unseen = GlobalWindowDataset(
        frame.filter(pl.col("cross_key_id").is_in(unseen_ids)),
        window_size=3,
        horizon=2,
    )
    return GlobalDatasetBundle(train, seen, unseen)


def _metrics(value: float) -> GlobalValidationMetrics:
    return GlobalValidationMetrics(
        macro_mae=value,
        macro_rmse=value,
        micro_mae=value,
        raw_macro_mae=value,
        raw_macro_rmse=value,
        raw_macro_wmape=value,
        raw_macro_smape=value,
        num_series=1,
        num_points=1,
        per_series={"S": {"robust_mase": value}},
        robust_macro_mase=value,
    )


def test_robust_mase_scale_is_causal_and_has_a_level_floor() -> None:
    values = np.asarray([100.0, 110.0, 120.0, 130.0])
    assert robust_mase_scale(values) == 10.0
    constant = np.asarray([1_000.0, 1_000.0, 1_000.0])
    assert robust_mase_scale(constant) == 10.0


def test_monitor_mase_denominator_uses_only_training_history() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "cross_key_id": ["S_saldo"] * 5,
            "actual_orig": [10.0, 20.0, 30.0, 1.0e9, -1.0e9],
            "pred_orig": [10.0] * 5,
            "isTrain": [True, True, True, False, False],
        }
    )
    scale = _causal_mase_scale_from_backtest(
        frame,
        series_id="S_saldo",
        comparison_start=dates[3],
    )
    assert scale == 10.0


def test_balanced_sampler_controls_type_level_series_and_window_dominance() -> None:
    dataset = GlobalWindowDataset(_frame(), window_size=3, horizon=2)
    sampler = GlobalBalancedSampler(dataset, num_samples=20_000, seed=20)
    indices = list(iter(sampler))
    samples = [dataset[index]["metadata"] for index in indices]
    type_counts: dict[str, int] = {}
    level_counts: dict[int, int] = {}
    series_counts: dict[str, int] = {}
    for metadata in samples:
        series_id = str(metadata["cross_key_id"])
        series_type = str(metadata["tipo_serie"])
        level = int(metadata["nivel_curriculum"])
        type_counts[series_type] = type_counts.get(series_type, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1
        series_counts[series_id] = series_counts.get(series_id, 0) + 1
    assert 0.48 <= type_counts["saldo"] / len(indices) <= 0.52
    assert 0.48 <= level_counts[1] / len(indices) <= 0.52
    assert max(series_counts.values()) / min(series_counts.values()) < 1.15


def test_hpo_uses_proxy_for_screening_and_medium_fidelity_for_selection() -> None:
    bundle = _bundle()
    base = GlobalTrainingConfig(
        epochs=5,
        batch_size=4,
        samples_per_epoch=8,
        selection_metric=DEFAULT_OBJECTIVE_METRIC,
        device="cpu",
    )

    def factory(window_size: int) -> GlobalDatasetBundle:
        assert window_size == 3
        return bundle

    def candidate_factory(trial, architecture, base_config):
        latent = 4 if trial.number == 0 else 8
        return GlobalCandidateConfig(
            window_size=3,
            model_config={
                "latent_dim": latent,
                "enc_hidden_size": 8,
                "enc_num_layers": 1,
                "dec_hidden_size": 8,
                "dec_num_layers": 1,
                "dropout_rate": 0.0,
                "activation": "gelu",
                "use_auxiliary_autoencoder": False,
            },
            training_config=base_config,
        )

    def fake_fit(self, datasets, *, epoch_callback=None, **kwargs):
        latent = int(self.model_config["latent_dim"])
        proxy = self.training_config.epochs == 1
        score = 1.0 if proxy else (3.0 if latent == 4 else 0.5)
        validation = {
            "validation_seen": _metrics(score),
            "validation_unseen": _metrics(score),
        }
        record = GlobalEpochRecord(
            epoch=1,
            train_loss=score,
            validation_objective=score,
            learning_rate=1e-3,
            validation=validation,
        )
        if epoch_callback is not None:
            epoch_callback(record)
        model = build_global_model(
            "mlp",
            self.model_config,
            window_size=3,
            horizon=2,
            exogenous_dim=0,
            static_dim=bundle.static_dim,
        )
        return GlobalTrainingResult(
            architecture="mlp",
            model=model,
            model_config=self.model_config,
            training_config=self.training_config,
            history=(record,),
            best_epoch=1,
            best_score=score,
            validation=validation,
            stopped_early=False,
        )

    trainer = GlobalHPOTrainer(
        "mlp",
        base_training_config=base,
        hpo_config=GlobalHPOConfig(
            epochs=1,
            windows_per_series_per_epoch=1,
            validation_windows_per_series=3,
            finalists=2,
            fidelity_epochs=2,
            fidelity_windows_per_series_per_epoch=2,
        ),
        candidate_factory=candidate_factory,
        seed=20,
    )
    with patch.object(GlobalTrainer, "fit", new=fake_fit):
        result = trainer.search_and_fit(factory, n_trials=2)

    assert result.study.best_trial.number == 0
    assert result.selected_trial_number == 1
    assert result.best_candidate.model_config["latent_dim"] == 8
    assert result.fidelity_scores == {0: 3.0, 1: 0.5}


def test_curriculum_and_shuffled_orders_have_equal_budgets() -> None:
    levels = (1, 2, 3)
    base = GlobalCurriculumConfig(
        warmup_epochs=2,
        fine_tune_epochs_per_level=4,
        consolidation_epochs=3,
        replay_fraction=0.25,
    )
    curriculum = base.build_stages(levels)
    shuffled = GlobalCurriculumConfig(
        **{**base.__dict__, "training_order": "shuffled"}
    ).build_stages(levels)
    assert sum(stage.epochs for stage in curriculum) == sum(stage.epochs for stage in shuffled)
    assert [stage.learning_rate_factor for stage in curriculum] == [
        stage.learning_rate_factor for stage in shuffled
    ]
    assert all(stage.current_levels == levels for stage in shuffled)
    assert all(stage.replay_fraction == 0.0 for stage in shuffled)


def test_monitor_selection_uses_only_causal_mase_without_redundant_rank_score() -> None:
    frame = pl.DataFrame(
        {
            "cross_key_id": ["S_saldo", "S_saldo"],
            "candidate_id": ["LOW_MAE", "LOW_MASE"],
            "MAE": [1.0, 100.0],
            "RMSE": [1.0, 100.0],
            "WMAPE": [1.0, 100.0],
            "EVS": [1.0, -1.0],
            "R2": [1.0, -1.0],
            "MASE": [2.0, 1.0],
        }
    )
    ranked = _rank_candidates(frame, (PRIMARY_WINNER_METRIC,))
    winner = ranked.sort("selection_rank").row(0, named=True)
    assert winner["candidate_id"] == "LOW_MASE"
    assert winner["selection_metric"] == "MASE"
    assert "rank_score" not in ranked.columns


def test_four_notebooks_expose_checkpoint_20_contract() -> None:
    assert DEFAULT_OBJECTIVE_METRIC == "robust_macro_mase"
    assert GlobalHPOConfig().validation_windows_per_series == 3
    for name in GLOBAL_NOTEBOOKS:
        notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            'SELECTION_METRIC = "robust_macro_mase"',
            "HPO_VALIDATION_WINDOWS_PER_SERIES = 3",
            "HPO_FINALISTS = 5",
            "HPO_FIDELITY_EPOCHS = 8",
            'TRAINING_ORDER = "curriculum"',
            "objective_metric=SELECTION_METRIC",
        ):
            assert token in code
        assert all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(ROOT / name), "exec")

    monitor_notebook = json.loads(
        (ROOT / "monitor_codigo_03_FINANCIAL_GPT.ipynb").read_text(encoding="utf-8")
    )
    monitor_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in monitor_notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'WINNER_METRICS = ["MASE"]' in monitor_code
    assert 'WINNER_METRICS = ["MAE", "RMSE", "WMAPE", "EVS", "R2"]' not in monitor_code
