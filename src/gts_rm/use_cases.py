from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

FACADE_MODULES = [
    "gts_rm.config",
    "gts_rm.data",
    "gts_rm.models",
    "gts_rm.training",
    "gts_rm.evaluation",
    "gts_rm.artifacts",
]


@dataclass(frozen=True)
class UseCaseContract:
    name: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def contract_path(self) -> Path:
        return REPO_ROOT / str(self.manifest["contract"])

    @property
    def cp20_bundle_path(self) -> Path:
        return REPO_ROOT / str(self.manifest["cp20_bundle"])

    @property
    def frozen_contract_path(self) -> Path:
        return REPO_ROOT / str(self.manifest["frozen_contract"])

    def required_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = [
            self.root,
            self.manifest_path,
            self.contract_path,
            self.cp20_bundle_path,
            self.frozen_contract_path,
        ]
        for path_value in self.manifest.get("directories", {}).values():
            paths.append(REPO_ROOT / str(path_value))
        for path_value in self.manifest.get("configs", {}).values():
            paths.append(REPO_ROOT / str(path_value))
        for workflow in self.manifest.get("workflows", {}).values():
            paths.append(REPO_ROOT / str(workflow["path"]))
            paths.append(REPO_ROOT / str(workflow["config"]))
        return tuple(paths)

    def validate(self) -> None:
        if self.manifest.get("kind") != "use_case":
            raise ValueError("manifest kind must be 'use_case'")
        if self.manifest.get("release_first") is not True:
            raise ValueError("release_first must be true")
        if self.manifest.get("tutorials_deferred") is not True:
            raise ValueError("tutorials_deferred must be true")
        if self.manifest.get("entry_package") != "gts_rm":
            raise ValueError("entry_package must be gts_rm")
        locked = self.manifest.get("locked_cp20_contract") or {}
        if locked.get("model_inputs") != ["y_context", "x_history", "x_future", "x_static"]:
            raise ValueError("locked CP20 model_inputs changed")
        if locked.get("output") != "y_pred":
            raise ValueError("locked CP20 output changed")
        if locked.get("latent") != "history_embedding":
            raise ValueError("locked CP20 latent changed")
        if locked.get("architectures") != ["mlp", "mlp_vae", "rnn", "rnn_bi"]:
            raise ValueError("locked CP20 architectures changed")
        facade = self.manifest.get("library_facade") or {}
        if facade.get("modules") != FACADE_MODULES:
            raise ValueError("library facade modules changed")
        if facade.get("migration_mode") != "wrapper_first":
            raise ValueError("library facade migration_mode must be wrapper_first")
        smoke = (self.manifest.get("workflows") or {}).get("smoke_global_mlp") or {}
        if smoke.get("uses_facade_modules") != ["gts_rm.models"]:
            raise ValueError("smoke workflow must use the gts_rm.models facade")
        missing = [path for path in self.required_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing use-case contract paths: {missing}")


def load_use_case(name: str = "MAC3_TEST") -> UseCaseContract:
    root = REPO_ROOT / name
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    contract = UseCaseContract(
        name=str(manifest.get("name") or name),
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    contract.validate()
    return contract


__all__ = ["FACADE_MODULES", "UseCaseContract", "load_use_case"]
