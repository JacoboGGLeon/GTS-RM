"""Feature-flag scaffold for the next Financial-GPT checkpoints.

This module is intentionally lightweight and CP20-compatible.  It does not
change runtime behavior by itself; Codex should wire it into models, training,
notebooks and monitors one checkpoint at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class FinancialGPTFeatureFlags:
    """Central switchboard for incremental Financial-GPT variants.

    Defaults preserve Checkpoint 20 behavior.
    """

    use_causal_scaler: bool = True
    use_observed_mask: bool = False
    use_context_mask: bool = False
    use_patch_tokenizer: bool = False
    use_calendar_encoder: bool = True
    use_calendar_future: bool = True
    use_static_context: bool = True
    use_local_residual_decoder: bool = False
    use_quantile_head: bool = False
    use_self_supervised_pretraining: bool = False
    use_auxiliary_autoencoder: bool = True

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if not self.use_causal_scaler:
            raise ValueError(
                "use_causal_scaler=False is not supported yet; CP20 requires a causal inverse transform."
            )
        if not self.use_calendar_future:
            raise ValueError(
                "use_calendar_future=False is not supported yet for direct multi-horizon forecasting."
            )

    def to_dict(self) -> Mapping[str, bool]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "FinancialGPTFeatureFlags":
        if payload is None:
            flags = cls()
        else:
            known = {field: payload[field] for field in cls.__dataclass_fields__ if field in payload}
            flags = cls(**known)  # type: ignore[arg-type]
        flags.validate()
        return flags


@dataclass(frozen=True)
class PatchTokenizerConfig:
    enabled: bool = False
    patch_size: int = 5
    patch_stride: int = 5
    patch_dim: int = 32

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        for name in ("patch_size", "patch_stride", "patch_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class LocalResidualConfig:
    enabled: bool = False
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    residual_lambda: float = 0.01
    global_aux_alpha: float = 0.2

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        for name in ("hidden_dim", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.residual_lambda) < 0.0:
            raise ValueError("residual_lambda must be non-negative")
        if float(self.global_aux_alpha) < 0.0:
            raise ValueError("global_aux_alpha must be non-negative")


@dataclass(frozen=True)
class QuantileHeadConfig:
    enabled: bool = False
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    quantile_weight: float = 1.0
    crossing_penalty: float = 0.01

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")
        previous = 0.0
        for q in self.quantiles:
            value = float(q)
            if not 0.0 < value < 1.0:
                raise ValueError("all quantiles must be in the open interval (0, 1)")
            if value <= previous:
                raise ValueError("quantiles must be strictly increasing")
            previous = value
        if float(self.quantile_weight) < 0.0:
            raise ValueError("quantile_weight must be non-negative")
        if float(self.crossing_penalty) < 0.0:
            raise ValueError("crossing_penalty must be non-negative")


@dataclass(frozen=True)
class SelfSupervisedConfig:
    enabled: bool = False
    task: str = "masked_reconstruction"
    mask_ratio: float = 0.15
    mask_strategy: str = "random"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if self.task != "masked_reconstruction":
            raise ValueError("Only masked_reconstruction is supported by the scaffold")
        if not 0.0 < float(self.mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be in the open interval (0, 1)")
        if self.mask_strategy not in {"random", "block", "patch"}:
            raise ValueError("mask_strategy must be one of: random, block, patch")


@dataclass(frozen=True)
class FinancialGPTStageConfig:
    """Combined scaffold config for upcoming checkpoints."""

    flags: FinancialGPTFeatureFlags = FinancialGPTFeatureFlags()
    patch_tokenizer: PatchTokenizerConfig = PatchTokenizerConfig()
    local_residual: LocalResidualConfig = LocalResidualConfig()
    quantile_head: QuantileHeadConfig = QuantileHeadConfig()
    self_supervised: SelfSupervisedConfig = SelfSupervisedConfig()

    def validate(self) -> None:
        self.flags.validate()
        self.patch_tokenizer.validate()
        self.local_residual.validate()
        self.quantile_head.validate()
        self.self_supervised.validate()
        if self.patch_tokenizer.enabled != self.flags.use_patch_tokenizer:
            raise ValueError("patch_tokenizer.enabled must match flags.use_patch_tokenizer")
        if self.local_residual.enabled != self.flags.use_local_residual_decoder:
            raise ValueError("local_residual.enabled must match flags.use_local_residual_decoder")
        if self.quantile_head.enabled != self.flags.use_quantile_head:
            raise ValueError("quantile_head.enabled must match flags.use_quantile_head")
        if self.self_supervised.enabled != self.flags.use_self_supervised_pretraining:
            raise ValueError("self_supervised.enabled must match flags.use_self_supervised_pretraining")

    def to_dict(self) -> Mapping[str, object]:
        self.validate()
        return asdict(self)
