"""Utilidades de representación para GTRM Stage 1.

Checkpoint 21 cierra la base global de representación: toda arquitectura global
produce ``y_pred`` y un ``history_embedding`` causal. Este módulo sólo extrae y
valida embeddings; no agrega residual local, cuantiles, patching ni SSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    DATE_COLUMN,
    DEFAULT_GTRM_STAGE1_FLAGS,
    GTRM_STAGE_FLAGS,
    HISTORY_EMBEDDING_FIELD,
    MODEL_INPUT_FIELDS,
    SERIES_TYPE_COLUMN,
    default_gtrm_stage1_flags,
    validate_gtrm_stage_flags,
)
from global_models import GlobalForecastModel, get_history_embedding, validate_global_model_output
from gtrm_config import GTRMModelConfig


@dataclass(frozen=True)
class GTRMStage1Config:
    """Flags explícitas para la base global de representación."""

    use_static_context: bool = True
    use_patch_tokenizer: bool = False
    use_local_residual_decoder: bool = False
    use_quantile_head: bool = False
    use_self_supervised_pretraining: bool = False

    def as_dict(self) -> dict[str, bool]:
        return validate_gtrm_stage_flags({
            "use_static_context": self.use_static_context,
            "use_patch_tokenizer": self.use_patch_tokenizer,
            "use_local_residual_decoder": self.use_local_residual_decoder,
            "use_quantile_head": self.use_quantile_head,
            "use_self_supervised_pretraining": self.use_self_supervised_pretraining,
        })

    def validate_stage1_only(self) -> None:
        flags = self.as_dict()
        if flags["use_patch_tokenizer"]:
            raise ValueError("use_patch_tokenizer belongs to a later GTRM stage")
        if flags["use_local_residual_decoder"]:
            raise ValueError("use_local_residual_decoder belongs to GTRM Stage 2")
        if flags["use_quantile_head"]:
            raise ValueError("use_quantile_head belongs to GTRM Stage 3")
        if flags["use_self_supervised_pretraining"]:
            raise ValueError("use_self_supervised_pretraining belongs to GTRM Stage 4")


def gtrm_stage1_manifest(
    *,
    use_static_context: bool = True,
    model_config: GTRMModelConfig | None = None,
) -> Mapping[str, object]:
    """Manifiesto operativo de Checkpoint 21/21.1."""

    cfg = model_config or GTRMModelConfig(use_static_context=use_static_context)
    cfg.validate(stage=1)
    flags = cfg.stage_flags()
    return {
        "checkpoint": 21,
        "subcheckpoint": "21.1",
        "name": "GTRM Stage 1 - Global Representation Base + Config Architecture",
        "model_config": cfg.as_dict(),
        "model_inputs": MODEL_INPUT_FIELDS,
        "latent_field": HISTORY_EMBEDDING_FIELD,
        "flags": flags,
        "future_stage_flags": tuple(
            key for key in GTRM_STAGE_FLAGS if key != "use_static_context"
        ),
        "default_flags": dict(DEFAULT_GTRM_STAGE1_FLAGS),
        "series_identity_in_forward": False,
        "hard_ids_forbidden_in_forward": (
            CROSS_KEY_COLUMN,
            ACCOUNT_CURRENCY_ID_COLUMN,
            CURRENCY_COLUMN,
            SERIES_TYPE_COLUMN,
        ),
        "acceptance_metrics": (
            "robust_macro_mase",
            "raw_macro_wmape",
            "p90_series_error",
            "percent_series_improved",
        ),
    }


def _to_device(model_inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in model_inputs.items()}


def _metadata_values(metadata: Mapping[str, Any], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(batch_size):
        row: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                item = value[i].item() if value.ndim > 0 else value.item()
            elif isinstance(value, np.ndarray):
                item = value[i].item() if value.ndim > 0 else value.item()
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                item = value[i]
            else:
                item = value
            row[str(key)] = item
        rows.append(row)
    return rows


def collect_history_embeddings(
    model: GlobalForecastModel,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Ejecuta inferencia y devuelve embeddings + metadata por ventana.

    La función asume batches con la forma generada por ``GlobalWindowDataset``:
    ``model_inputs`` contiene los cuatro tensores canónicos y ``metadata`` trae
    trazabilidad sólo para análisis posterior. Ningún identificador se pasa al
    ``forward``.
    """

    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= int(max_batches):
                break
            model_inputs = batch["model_inputs"]
            if tuple(model_inputs) != MODEL_INPUT_FIELDS:
                raise RuntimeError("Batch model_inputs violate the canonical GTRM order")
            model_inputs = _to_device(model_inputs, device_obj)
            output = model(**model_inputs)
            embedding = validate_global_model_output(
                output,
                batch_size=model_inputs["y_context"].shape[0],
                horizon=model_inputs["x_future"].shape[1],
            )
            embedding_np = embedding.detach().cpu().numpy()
            meta_rows = _metadata_values(batch.get("metadata", {}), embedding_np.shape[0])
            for meta, vector in zip(meta_rows, embedding_np):
                record = dict(meta)
                for idx, value in enumerate(vector):
                    record[f"{HISTORY_EMBEDDING_FIELD}_{idx:03d}"] = float(value)
                record["embedding_dim"] = int(vector.shape[0])
                rows.append(record)
    return pd.DataFrame(rows)
