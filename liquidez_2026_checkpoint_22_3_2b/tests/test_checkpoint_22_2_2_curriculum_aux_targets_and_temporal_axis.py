import importlib.util

import pytest

if importlib.util.find_spec("polars") is None:
    pytest.skip("polars is required by the global dataset/curriculum modules", allow_module_level=True)

import warnings
from unittest.mock import Mock

import pandas as pd
import torch

from global_curriculum import _train_one_epoch
from temporal_axis import TemporalAxis


class TinyAuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **model_inputs):
        y_context = model_inputs["y_context"]
        batch = y_context.shape[0]
        horizon = 1
        y_pred = self.bias.expand(batch, horizon, 1)
        return {
            "y_pred": y_pred,
            "extras": {
                "event_logits": torch.zeros(batch, horizon, 1, device=y_context.device) + self.bias,
                "magnitude_pred": torch.zeros(batch, horizon, 1, device=y_context.device) + self.bias,
                "direction_logits": torch.zeros(batch, horizon, 3, device=y_context.device) + self.bias,
                "event_loss_weight": 0.1,
                "magnitude_loss_weight": 0.1,
                "direction_loss_weight": 0.1,
            },
        }


def test_curriculum_train_one_epoch_passes_auxiliary_targets_to_loss():
    batch = {
        "model_inputs": {
            "y_context": torch.zeros(2, 3, 1),
            "x_history": torch.zeros(2, 3, 0),
            "x_future": torch.zeros(2, 1, 0),
            "x_static": torch.zeros(2, 0),
        },
        "targets": {
            "y_future": torch.zeros(2, 1, 1),
            "event_target": torch.zeros(2, 1, 1),
            "magnitude_target": torch.zeros(2, 1, 1),
            "direction_target": torch.ones(2, 1, dtype=torch.long),
        },
    }
    model = TinyAuxModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = Mock(loss="mse", huber_delta=1.0, grad_clip_norm=None)

    value = _train_one_epoch(model, [batch], optimizer, cfg, torch.device("cpu"))

    assert value >= 0.0


def test_temporal_axis_to_polars_defragments_fragmented_frame():
    frame = pd.DataFrame({"fecha": pd.date_range("2026-01-01", periods=4)})
    for idx in range(80):
        frame[f"feature_{idx}"] = idx
    axis = TemporalAxis(
        frame,
        timestamp_column="fecha",
        feature_columns=[f"feature_{idx}" for idx in range(80)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        output = axis.to_polars()

    assert output.height == 4
    assert not [w for w in caught if issubclass(w.category, pd.errors.PerformanceWarning)]
