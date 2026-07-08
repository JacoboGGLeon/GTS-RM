from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from gts_rm import MAC3_TEST_ROOT, config, models


def config_path_for(architecture: str) -> Path:
    key = str(architecture).strip().lower()
    return MAC3_TEST_ROOT / "configs" / f"smoke_global_{key}.json"


def _synthetic_batch(config_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    batch_size = int(config_payload.get("batch_size", 2))
    window_size = int(config_payload["window_size"])
    horizon = int(config_payload["horizon"])
    exogenous_dim = int(config_payload["exogenous_dim"])
    static_dim = int(config_payload["static_dim"])

    y_context = torch.linspace(-1.0, 1.0, steps=batch_size * window_size).reshape(
        batch_size, window_size, 1
    )
    x_history = torch.zeros(batch_size, window_size, exogenous_dim)
    x_future = torch.zeros(batch_size, horizon, exogenous_dim)
    x_static = torch.zeros(batch_size, static_dim)
    if static_dim:
        x_static[:, 0] = 1.0

    return {
        "y_context": y_context,
        "x_history": x_history,
        "x_future": x_future,
        "x_static": x_static,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_global_smoke(
    architecture: str,
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    if config_path is None:
        config_payload = config.load_smoke_config(architecture)
        resolved_config_path = config_path_for(architecture)
    else:
        config_payload = config.load_json_config(config_path)
        resolved_config_path = Path(config_path)
    if config_payload["architecture"] != architecture:
        raise ValueError(
            f"Smoke config architecture mismatch: expected {architecture!r}, "
            f"got {config_payload['architecture']!r}"
        )

    torch.manual_seed(int(config_payload.get("seed", 7)))
    model = models.build_global_model(
        architecture,
        config_payload["model_config"],
        window_size=int(config_payload["window_size"]),
        horizon=int(config_payload["horizon"]),
        exogenous_dim=int(config_payload["exogenous_dim"]),
        static_dim=int(config_payload["static_dim"]),
    )
    model.eval()

    inputs = _synthetic_batch(config_payload)
    with torch.no_grad():
        output = model(**inputs)

    prediction = output[models.GLOBAL_OUTPUT_FIELD]
    history_embedding = output["extras"][models.GLOBAL_LATENT_FIELD]
    expected_shape = tuple(int(value) for value in config_payload["expected_output_shape"])
    actual_shape = tuple(int(value) for value in prediction.shape)
    finite_prediction = bool(torch.isfinite(prediction).all().item())
    finite_embedding = bool(torch.isfinite(history_embedding).all().item())
    ok = actual_shape == expected_shape and finite_prediction and finite_embedding

    now = datetime.now(timezone.utc).isoformat()
    name = str(config_payload["name"])
    root = Path(output_root) if output_root is not None else MAC3_TEST_ROOT
    report_path = root / "reports" / f"{name}.json"
    run_record_path = root / "runs" / f"{name}_run.json"
    report = {
        "name": name,
        "checkpoint": config_payload.get("checkpoint", "CP24"),
        "ok": ok,
        "created_at_utc": now,
        "architecture": architecture,
        "expected_output_shape": list(expected_shape),
        "actual_output_shape": list(actual_shape),
        "history_embedding_shape": [int(value) for value in history_embedding.shape],
        "finite_prediction": finite_prediction,
        "finite_history_embedding": finite_embedding,
        "facade_modules": ["gts_rm.config", "gts_rm.models"],
    }
    run_record = {
        "name": name,
        "checkpoint": config_payload.get("checkpoint", "CP24"),
        "created_at_utc": now,
        "config_path": str(resolved_config_path),
        "report_path": str(report_path),
    }
    _write_json(report_path, report)
    _write_json(run_record_path, run_record)
    return report


def run_all_global_smokes(
    *,
    output_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        architecture: run_global_smoke(architecture, output_root=output_root)
        for architecture in models.list_global_models()
    }
