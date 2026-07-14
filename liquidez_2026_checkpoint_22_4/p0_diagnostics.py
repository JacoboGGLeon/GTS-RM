"""P0 diagnostics required before autoregressive residual refinement."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from global_contracts import SERIES_TYPE_COLUMN
from global_models import (
    DIRECTION_LOGITS_FIELD, EVENT_LOGITS_FIELD, MAGNITUDE_PRED_FIELD,
    GlobalForecastModel,
)


def _divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _binary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = target.astype(bool).ravel(), prediction.astype(bool).ravel()
    tp, fp = int(np.sum(target & prediction)), int(np.sum(~target & prediction))
    fn, tn = int(np.sum(target & ~prediction)), int(np.sum(~target & ~prediction))
    precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
    return {"accuracy": _divide(tp + tn, target.size), "precision": precision,
            "recall": recall, "f1": _divide(2 * precision * recall, precision + recall)}


def auxiliary_head_metrics(*, event_target: np.ndarray, event_probability: np.ndarray,
                           magnitude_target: np.ndarray, magnitude_prediction: np.ndarray,
                           direction_target: np.ndarray, direction_prediction: np.ndarray,
                           event_threshold: float = 0.5) -> Mapping[str, Mapping[str, float]]:
    event = _binary(event_target, event_probability >= event_threshold)
    error = np.asarray(magnitude_prediction) - np.asarray(magnitude_target)
    f1 = [_binary(np.asarray(direction_target) == k, np.asarray(direction_prediction) == k)["f1"]
          for k in (0, 1, 2)]
    return {"event": event,
            "magnitude": {"mae": float(np.mean(np.abs(error))),
                          "rmse": float(np.sqrt(np.mean(error ** 2)))},
            "direction": {"accuracy": float(np.mean(np.asarray(direction_target) == np.asarray(direction_prediction))),
                          "macro_f1": float(np.mean(f1))}}


def evaluate_auxiliary_heads(model: GlobalForecastModel, dataset: Any, *, batch_size: int = 256,
                             device: str | torch.device = "cpu") -> pd.DataFrame:
    """Evaluate event/magnitude/direction heads by horizon and series type."""
    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    model = model.to(device).eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            output = model(**{k: v.to(device) for k, v in batch["model_inputs"].items()})
            extras = output.get("extras", {})
            required = (EVENT_LOGITS_FIELD, MAGNITUDE_PRED_FIELD, DIRECTION_LOGITS_FIELD)
            if any(extras.get(key) is None for key in required):
                raise ValueError("P0 requires all three auxiliary heads to be enabled")
            event = torch.sigmoid(extras[EVENT_LOGITS_FIELD]).cpu().numpy()
            magnitude = extras[MAGNITUDE_PRED_FIELD].cpu().numpy()
            direction = extras[DIRECTION_LOGITS_FIELD].argmax(-1).cpu().numpy()
            targets, types = batch["targets"], list(batch["metadata"][SERIES_TYPE_COLUMN])
            for i, series_type in enumerate(types):
                for h in range(event.shape[1]):
                    rows.append({"series_type": str(series_type), "horizon_step": h + 1,
                                 "event_target": float(targets["event_target"][i, h, 0]),
                                 "event_probability": float(event[i, h, 0]),
                                 "magnitude_target": float(targets["magnitude_target"][i, h, 0]),
                                 "magnitude_prediction": float(magnitude[i, h, 0]),
                                 "direction_target": int(targets["direction_target"][i, h]),
                                 "direction_prediction": int(direction[i, h])})
    raw, report = pd.DataFrame(rows), []
    for (series_type, horizon), frame in raw.groupby(["series_type", "horizon_step"]):
        m = auxiliary_head_metrics(event_target=frame.event_target.to_numpy(),
            event_probability=frame.event_probability.to_numpy(),
            magnitude_target=frame.magnitude_target.to_numpy(),
            magnitude_prediction=frame.magnitude_prediction.to_numpy(),
            direction_target=frame.direction_target.to_numpy(),
            direction_prediction=frame.direction_prediction.to_numpy())
        report.append({"series_type": series_type, "horizon_step": int(horizon),
                       "event_f1": m["event"]["f1"], "event_precision": m["event"]["precision"],
                       "event_recall": m["event"]["recall"], "magnitude_mae": m["magnitude"]["mae"],
                       "magnitude_rmse": m["magnitude"]["rmse"],
                       "direction_accuracy": m["direction"]["accuracy"],
                       "direction_macro_f1": m["direction"]["macro_f1"], "support": len(frame)})
    return pd.DataFrame(report).sort_values(["series_type", "horizon_step"]).reset_index(drop=True)


def interval_calibration_by_horizon(frame: pd.DataFrame, *, nominal_coverage: float = 0.95,
                                    evaluation_only: bool = True) -> pd.DataFrame:
    required = {"actual_orig", "lower_ci", "upper_ci", "horizon_step", "tipo_serie"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Missing interval columns: {missing}")
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be in (0, 1)")
    work = frame.loc[~frame.isTrain.astype(bool)].copy() if evaluation_only and "isTrain" in frame else frame.copy()
    actual, lower, upper = (work[c].to_numpy(float) for c in ("actual_orig", "lower_ci", "upper_ci"))
    work["covered"], work["width"] = (actual >= lower) & (actual <= upper), upper - lower
    score, alpha = (upper - lower).copy(), 1 - nominal_coverage
    score[actual < lower] += 2 / alpha * (lower[actual < lower] - actual[actual < lower])
    score[actual > upper] += 2 / alpha * (actual[actual > upper] - upper[actual > upper])
    work["interval_score"] = score
    out = work.groupby(["tipo_serie", "horizon_step"], as_index=False).agg(
        empirical_coverage=("covered", "mean"), mean_interval_width=("width", "mean"),
        interval_score=("interval_score", "mean"), support=("covered", "size"))
    out["nominal_coverage"], out["coverage_gap"] = nominal_coverage, out.empirical_coverage - nominal_coverage
    return out


def compare_residual_runs(enabled: Mapping[str, float], disabled: Mapping[str, float], *, metric: str) -> Mapping[str, Any]:
    a, b = float(enabled[metric]), float(disabled[metric])
    return {"metric": metric, "residual_enabled": a, "residual_disabled": b,
            "delta_enabled_minus_disabled": a - b, "relative_improvement": _divide(b - a, b),
            "interpretation": "independent_seed_matched_training_ablation"}


@dataclass(frozen=True)
class PatienceDiagnostic:
    epochs_run: int
    best_epoch: int
    trailing_epochs: int
    objective_noise_mad: float
    recommended_patience: int
    stopped_near_best: bool
    def to_dict(self) -> Mapping[str, Any]: return self.__dict__.copy()


def diagnose_patience(history: Iterable[Any], *, configured_patience: int) -> PatienceDiagnostic:
    records = list(history)
    values = np.asarray([float(getattr(r, "validation_objective", r.get("validation_objective") if isinstance(r, dict) else np.nan)) for r in records])
    if not len(values) or not np.all(np.isfinite(values)): raise ValueError("history needs finite objectives")
    best, diff = int(np.argmin(values)), np.diff(values)
    median = float(np.median(diff)) if len(diff) else 0.0
    mad = float(np.median(np.abs(diff - median))) if len(diff) else 0.0
    improvements = np.abs(diff[diff < 0])
    typical = float(np.median(improvements)) if len(improvements) else np.finfo(float).eps
    recommended = max(configured_patience, int(math.ceil(4 + 2 * min(_divide(mad, typical), 6))))
    trailing = len(values) - best - 1
    return PatienceDiagnostic(len(values), best + 1, trailing, mad, recommended,
                              trailing <= max(1, configured_patience // 2))
