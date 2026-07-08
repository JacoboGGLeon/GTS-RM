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

from gts_rm import MAC3_TEST_ROOT, models


CONFIG_PATH = MAC3_TEST_ROOT / "configs" / "smoke_global_mlp.json"


def _load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def _synthetic_batch(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    batch_size = int(config.get("batch_size", 2))
    window_size = int(config["window_size"])
    horizon = int(config["horizon"])
    exogenous_dim = int(config["exogenous_dim"])
    static_dim = int(config["static_dim"])

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


def run_smoke(
    *,
    config_path: str | Path = CONFIG_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_config(Path(config_path))
    torch.manual_seed(int(config.get("seed", 7)))

    model = models.build_global_model(
        str(config["architecture"]),
        config["model_config"],
        window_size=int(config["window_size"]),
        horizon=int(config["horizon"]),
        exogenous_dim=int(config["exogenous_dim"]),
        static_dim=int(config["static_dim"]),
    )
    model.eval()

    inputs = _synthetic_batch(config)
    with torch.no_grad():
        output = model(**inputs)

    prediction = output[models.GLOBAL_OUTPUT_FIELD]
    history_embedding = output["extras"][models.GLOBAL_LATENT_FIELD]
    expected_shape = tuple(int(value) for value in config["expected_output_shape"])
    actual_shape = tuple(int(value) for value in prediction.shape)
    finite_prediction = bool(torch.isfinite(prediction).all().item())
    finite_embedding = bool(torch.isfinite(history_embedding).all().item())
    ok = actual_shape == expected_shape and finite_prediction and finite_embedding

    now = datetime.now(timezone.utc).isoformat()
    report = {
        "name": config["name"],
        "checkpoint": "CP23",
        "ok": ok,
        "created_at_utc": now,
        "architecture": config["architecture"],
        "expected_output_shape": list(expected_shape),
        "actual_output_shape": list(actual_shape),
        "history_embedding_shape": [int(value) for value in history_embedding.shape],
        "finite_prediction": finite_prediction,
        "finite_history_embedding": finite_embedding,
        "facade_modules": ["gts_rm.models"],
    }

    root = Path(output_root) if output_root is not None else MAC3_TEST_ROOT
    run_record = {
        "name": "smoke_global_mlp",
        "checkpoint": "CP23",
        "created_at_utc": now,
        "config_path": str(Path(config_path)),
        "report_path": str(root / "reports" / "smoke_global_mlp.json"),
    }
    _write_json(root / "reports" / "smoke_global_mlp.json", report)
    _write_json(root / "runs" / "smoke_global_mlp_run.json", run_record)
    return report


if __name__ == "__main__":
    result = run_smoke()
    raise SystemExit(0 if result["ok"] else 1)
