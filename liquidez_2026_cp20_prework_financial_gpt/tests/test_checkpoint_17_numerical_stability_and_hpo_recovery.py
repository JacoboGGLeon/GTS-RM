from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from global_data import ContextScale, ContextScaler
from global_manager import GlobalManager
from global_training import GlobalTrainingConfig, evaluate_global_model

ROOT = Path(__file__).resolve().parents[1]


class HugeFiniteForecast(torch.nn.Module):
    def eval(self):
        super().eval()
        return self

    def forward(self, **kwargs):
        batch = kwargs["y_context"].shape[0]
        return {"y_pred": torch.full((batch, 2, 1), 1.0e6)}


def validation_batch() -> dict:
    return {
        "model_inputs": {
            "y_context": torch.zeros(1, 3, 1),
            "x_history": torch.zeros(1, 3, 0),
            "x_future": torch.zeros(1, 2, 0),
            "x_static": torch.ones(1, 1),
        },
        "targets": {
            "y_future": torch.zeros(1, 2, 1),
            "y_future_raw": torch.zeros(1, 2, 1),
        },
        "metadata": {
            "center": [0.0],
            "scale": [1.0],
            "transform": ["linear_context_scale"],
            "cross_key_id": ["SERIE_A"],
        },
    }


def test_inverse_transform_is_finite_and_reports_clipping() -> None:
    raw, diagnostics = ContextScaler.inverse_transform_with_diagnostics(
        np.asarray([1.0e308, -1.0e308]),
        ContextScale(center=0.0, scale=1.0, transform="linear_context_scale"),
    )
    assert np.all(np.isfinite(raw))
    assert diagnostics["clipped_values"] == 2
    assert diagnostics["nonfinite_inputs"] == 0


def test_raw_smape_remains_finite_for_extreme_finite_predictions() -> None:
    report = evaluate_global_model(HugeFiniteForecast(), [validation_batch()], device="cpu")
    assert math.isfinite(report.raw_macro_smape)
    assert 0.0 <= report.raw_macro_smape <= 200.0
    assert report.num_clipped_predictions == 0


def test_training_config_exposes_generic_nonfinite_recovery() -> None:
    config = GlobalTrainingConfig(nonfinite_max_retries=2, nonfinite_lr_factor=0.25)
    config.validate()
    assert config.nonfinite_max_retries == 2
    assert config.nonfinite_lr_factor == 0.25
    with pytest.raises(ValueError):
        GlobalTrainingConfig(nonfinite_lr_factor=1.0).validate()


def test_manager_commits_hpo_before_productive_warmup_failure() -> None:
    manager = GlobalManager("mlp")
    candidate = SimpleNamespace(
        window_size=3,
        model_config={},
        training_config=GlobalTrainingConfig(epochs=1, batch_size=2),
    )
    hpo_result = SimpleNamespace(best_candidate=candidate)

    train = SimpleNamespace(
        exogenous_columns=(),
        static_feature_names=("legacy_static",),
        static_feature_encoder=None,
        series_ids=("A",),
    )
    validation = SimpleNamespace(exogenous_columns=())
    bundle = SimpleNamespace(
        train=train,
        validation_datasets={"validation_seen": validation, "validation_unseen": validation},
        window_size=3,
        horizon=2,
        exogenous_dim=0,
        static_dim=1,
        static_feature_names=("legacy_static",),
        validate=Mock(),
    )
    factory = Mock(return_value=bundle)
    session = Mock()
    session.run_phases.side_effect = RuntimeError("synthetic warmup failure")

    with patch("global_manager.GlobalHPOTrainer.search_and_fit", return_value=hpo_result), patch(
        "global_manager.GlobalCurriculumSession", return_value=session
    ):
        with pytest.raises(RuntimeError, match="synthetic warmup failure"):
            manager._warmup_all(factory, n_trials=1)

    assert manager.hpo_result is hpo_result
    assert manager.datasets is bundle
    assert manager._curriculum_session is session


def test_global_notebooks_persist_optuna_and_expose_recovery_controls() -> None:
    notebooks = sorted(ROOT.glob("code_03_GLOBAL_*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        assert "optuna_study.db" in source
        assert "hpo_storage=HPO_STORAGE_URI" in source
        assert "NONFINITE_MAX_RETRIES" in source
        assert "NONFINITE_LR_FACTOR" in source
        assert "retries={record.recovery_retries}" in source
