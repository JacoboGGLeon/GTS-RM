"""Configuración modular única para GTRM.

Checkpoint 21.1 centraliza las banderas del modelo en un solo objeto. La idea
es que notebooks, dataset, modelos y futuros heads compartan la misma fuente de
verdad, en lugar de repartir flags sueltas por el proyecto.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from global_contracts import SUPPORTED_ARCHITECTURES, validate_gtrm_stage_flags


@dataclass(frozen=True)
class GTRMModelConfig:
    """Configuración canónica de arquitectura para GTRM.

    Stage 1 sólo activa ``use_static_context``. Las demás banderas existen como
    hooks explícitos para etapas posteriores, pero empiezan apagadas para que el
    Global Representation Base siga siendo auditable y comparable con
    Checkpoint 20/21.
    """

    # Backbone
    architecture: str = "rnn"

    # Inputs / encoders
    use_static_context: bool = True
    use_calendar_encoder: bool = True
    use_modality_specific_encoders: bool = False
    use_patch_tokenizer: bool = False

    # Heads
    use_local_residual_decoder: bool = False
    use_quantile_head: bool = False
    use_event_head: bool = False
    use_magnitude_head: bool = False
    use_direction_head: bool = False
    event_threshold: float = 1.0
    magnitude_transform: str = "asinh"

    # Training / objectives
    use_self_supervised_pretraining: bool = False
    loss_type: str = "huber"
    use_hpo: bool = True

    # Representation size hint. The concrete architecture may still override it
    # through the HPO/model cfg, but exposing it here keeps the config complete.
    latent_dim: int | None = None

    def normalized_architecture(self) -> str:
        architecture = str(self.architecture).strip().lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture={self.architecture!r}; "
                f"expected {SUPPORTED_ARCHITECTURES}"
            )
        return architecture

    def stage_flags(self) -> dict[str, bool]:
        """Devuelve sólo las flags modulares del GTRM."""

        return validate_gtrm_stage_flags(
            {
                "use_static_context": self.use_static_context,
                "use_modality_specific_encoders": self.use_modality_specific_encoders,
                "use_patch_tokenizer": self.use_patch_tokenizer,
                "use_local_residual_decoder": self.use_local_residual_decoder,
                "use_quantile_head": self.use_quantile_head,
                "use_self_supervised_pretraining": self.use_self_supervised_pretraining,
                "use_event_head": self.use_event_head,
                "use_magnitude_head": self.use_magnitude_head,
                "use_direction_head": self.use_direction_head,
            }
        )

    def validate(self, *, stage: int = 1) -> None:
        self.normalized_architecture()
        self.stage_flags()
        if isinstance(stage, bool) or int(stage) <= 0:
            raise ValueError("stage must be a positive integer")
        if not isinstance(self.use_calendar_encoder, bool):
            raise TypeError("use_calendar_encoder must be a boolean")
        if not isinstance(self.use_modality_specific_encoders, bool):
            raise TypeError("use_modality_specific_encoders must be a boolean")
        if not isinstance(self.use_hpo, bool):
            raise TypeError("use_hpo must be a boolean")
        if self.latent_dim is not None and int(self.latent_dim) <= 0:
            raise ValueError("latent_dim must be positive when provided")
        if not str(self.loss_type).strip():
            raise ValueError("loss_type must not be empty")
        if float(self.event_threshold) <= 0.0:
            raise ValueError("event_threshold must be positive")
        if str(self.magnitude_transform).strip().lower() not in {"asinh", "log1p", "abs", "none", "identity"}:
            raise ValueError("magnitude_transform must be one of: asinh, log1p, abs")
        if int(stage) <= 1:
            self.validate_stage1_only()
        elif int(stage) == 2:
            self.validate_stage2_only()

    def validate_stage1_only(self) -> None:
        flags = self.stage_flags()
        if flags["use_modality_specific_encoders"]:
            raise ValueError("use_modality_specific_encoders belongs to GTRM Stage 2.3")
        if flags["use_patch_tokenizer"]:
            raise ValueError("use_patch_tokenizer belongs to a later GTRM stage")
        if flags["use_local_residual_decoder"]:
            raise ValueError("use_local_residual_decoder belongs to GTRM Stage 2")
        if flags["use_quantile_head"]:
            raise ValueError("use_quantile_head belongs to GTRM Stage 3")
        if flags["use_self_supervised_pretraining"]:
            raise ValueError("use_self_supervised_pretraining belongs to GTRM Stage 4")
        if flags["use_event_head"] or flags["use_magnitude_head"] or flags["use_direction_head"]:
            raise ValueError("agnostic auxiliary heads belong to GTRM Stage 2.2")

    def validate_stage2_only(self) -> None:
        flags = self.stage_flags()
        if flags["use_patch_tokenizer"]:
            raise ValueError("use_patch_tokenizer belongs to GTRM Stage 4/5")
        if flags["use_quantile_head"]:
            raise ValueError("use_quantile_head belongs to GTRM Stage 3")
        if flags["use_self_supervised_pretraining"]:
            raise ValueError("use_self_supervised_pretraining belongs to GTRM Stage 4")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["architecture"] = self.normalized_architecture()
        payload["stage_flags"] = self.stage_flags()
        return payload

    def dataset_kwargs(self) -> dict[str, bool]:
        """Argumentos que el dataset necesita hoy."""

        return {
            "use_static_context": bool(self.use_static_context),
            "event_threshold": float(self.event_threshold),
            "magnitude_transform": str(self.magnitude_transform).strip().lower(),
        }

    def model_label_suffix(self) -> str:
        """Etiqueta compacta y estable para reportes de ablation."""

        flags = self.stage_flags()
        enabled = [key.replace("use_", "") for key, value in flags.items() if value]
        return "GTRM_" + "_".join(enabled or ["global_only"])

    @classmethod
    def from_notebook_globals(cls, values: Mapping[str, Any]) -> "GTRMModelConfig":
        """Construye config desde variables de notebook en mayúsculas.

        Esto permite que la celda ``#@title Configuración general`` sea el lugar
        visible donde el usuario controla las flags.
        """

        return cls(
            architecture=str(values.get("ARCHITECTURE", "rnn")),
            use_static_context=bool(values.get("USE_STATIC_CONTEXT", True)),
            use_calendar_encoder=bool(values.get("USE_CALENDAR_ENCODER", True)),
            use_modality_specific_encoders=bool(
                values.get("USE_MODALITY_SPECIFIC_ENCODERS", False)
            ),
            use_patch_tokenizer=bool(values.get("USE_PATCH_TOKENIZER", False)),
            use_local_residual_decoder=bool(values.get("USE_LOCAL_RESIDUAL_DECODER", False)),
            use_quantile_head=bool(values.get("USE_QUANTILE_HEAD", False)),
            use_event_head=bool(values.get("USE_EVENT_HEAD", False)),
            use_magnitude_head=bool(values.get("USE_MAGNITUDE_HEAD", False)),
            use_direction_head=bool(values.get("USE_DIRECTION_HEAD", False)),
            event_threshold=float(values.get("EVENT_THRESHOLD", 1.0)),
            magnitude_transform=str(values.get("MAGNITUDE_TRANSFORM", "asinh")),
            use_self_supervised_pretraining=bool(
                values.get("USE_SELF_SUPERVISED_PRETRAINING", False)
            ),
            loss_type=str(values.get("LOSS_FUNCTION", values.get("LOSS_TYPE", "huber"))),
            use_hpo=bool(values.get("USE_HPO", True)),
            latent_dim=values.get("LATENT_DIM"),
        )
