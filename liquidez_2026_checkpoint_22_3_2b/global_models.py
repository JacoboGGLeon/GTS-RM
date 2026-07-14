"""Arquitecturas globales de Deep Representation Learning para forecasting.

Checkpoint 3 define únicamente cuatro modelos globales bajo un contrato común:

- ``GlobalMLPEncoderDecoder``;
- ``GlobalMLPVAEEncoderDecoder``;
- ``GlobalRNNEncoderDecoder``;
- ``GlobalRNNBiEncoderDecoder``.

Todos comparten pesos entre series y reciben contexto escalado linealmente,
covariables temporales del calendario y un vector ``x_static`` no identificador
(tipo, divisa, escala contextual y edad causal). Checkpoint 22.3 permite
codificar por separado target histórico, exógenas históricas, exógenas futuras
y contexto estático antes de fusionar ``history_embedding``. Generan el
horizonte completo con una única cabeza de forecasting. Un decoder autoencoder
auxiliar puede reconstruir el contexto escalado desde el espacio latente. Los
identificadores contables permanecen fuera de ``forward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import inspect
from typing import Any, Dict, Final, Mapping, Sequence, Tuple, Type

import torch
from torch import nn

from global_contracts import (
    GLOBAL_COMPONENT_FIELD,
    GLOBAL_OUTPUT_FIELD,
    HISTORY_EMBEDDING_FIELD,
    LOCAL_RESIDUAL_FIELD,
    MODEL_INPUT_FIELDS,
    RECONSTRUCTION_FIELD,
    SUPPORTED_ARCHITECTURES,
)


GlobalModelOutput = Dict[str, Any]
GLOBAL_LATENT_FIELD: Final[str] = HISTORY_EMBEDDING_FIELD
EVENT_LOGITS_FIELD: Final[str] = "event_logits"
MAGNITUDE_PRED_FIELD: Final[str] = "magnitude_pred"
DIRECTION_LOGITS_FIELD: Final[str] = "direction_logits"


_ACTIVATIONS: Final[Mapping[str, Type[nn.Module]]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _global_output(
    prediction: torch.Tensor,
    *,
    history_embedding: torch.Tensor,
    reconstruction: torch.Tensor | None = None,
    extras: Mapping[str, Any] | None = None,
    losses: Mapping[str, torch.Tensor] | None = None,
) -> GlobalModelOutput:
    """Build the series-agnostic forecast output and auxiliary reconstruction.

    The forecasting head is unique for every series.  The auxiliary decoder
    reconstructs the normalized historical target from the same latent space
    and is used only as a training regularizer.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("prediction must have shape [batch, horizon, 1]")
    output: GlobalModelOutput = {
        GLOBAL_OUTPUT_FIELD: prediction,
        "extras": {GLOBAL_LATENT_FIELD: history_embedding, **dict(extras or {})},
    }
    if reconstruction is not None:
        if reconstruction.ndim != 3 or reconstruction.shape[-1] != 1:
            raise ValueError("reconstruction must have shape [batch, window_size, 1]")
        output[RECONSTRUCTION_FIELD] = reconstruction
    if losses:
        output["losses"] = dict(losses)
    return output


def get_history_embedding(output: Mapping[str, Any]) -> torch.Tensor:
    """Extrae el ``history_embedding`` como artefacto explícito de GTRM.

    Checkpoint 21 formaliza que toda arquitectura global debe devolver una
    representación histórica causal en ``output["extras"]["history_embedding"]``.
    """

    if not isinstance(output, Mapping):
        raise TypeError("output must be a mapping returned by a global model")
    extras = output.get("extras")
    if not isinstance(extras, Mapping):
        raise KeyError("global model output must contain an extras mapping")
    embedding = extras.get(GLOBAL_LATENT_FIELD)
    if not isinstance(embedding, torch.Tensor):
        raise KeyError(f"extras must contain tensor {GLOBAL_LATENT_FIELD!r}")
    if embedding.ndim != 2:
        raise ValueError("history_embedding must have shape [batch, latent_dim]")
    if not torch.isfinite(embedding).all():
        raise ValueError("history_embedding contains non-finite values")
    return embedding


def validate_global_model_output(
    output: Mapping[str, Any],
    *,
    batch_size: int | None = None,
    horizon: int | None = None,
    latent_dim: int | None = None,
) -> torch.Tensor:
    """Valida el contrato de salida del GTRM Stage 1 y devuelve el embedding."""

    prediction = output.get(GLOBAL_OUTPUT_FIELD)
    if not isinstance(prediction, torch.Tensor):
        raise KeyError(f"global model output must contain tensor {GLOBAL_OUTPUT_FIELD!r}")
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("y_pred must have shape [batch, horizon, 1]")
    if batch_size is not None and prediction.shape[0] != int(batch_size):
        raise ValueError("y_pred batch dimension does not match the requested batch_size")
    if horizon is not None and prediction.shape[1] != int(horizon):
        raise ValueError("y_pred horizon dimension does not match the requested horizon")
    if not torch.isfinite(prediction).all():
        raise ValueError("y_pred contains non-finite values")

    embedding = get_history_embedding(output)
    if batch_size is not None and embedding.shape[0] != int(batch_size):
        raise ValueError("history_embedding batch dimension does not match batch_size")
    if latent_dim is not None and embedding.shape[1] != int(latent_dim):
        raise ValueError("history_embedding dimension does not match latent_dim")
    return embedding


@dataclass(frozen=True)
class GlobalModelDimensions:
    """Dimensiones estáticas compartidas por dataset y arquitectura."""

    window_size: int
    horizon: int
    exogenous_dim: int
    static_dim: int = 1

    def validate(self) -> None:
        if isinstance(self.window_size, bool) or int(self.window_size) <= 0:
            raise ValueError("window_size must be a positive integer")
        if isinstance(self.horizon, bool) or int(self.horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        if isinstance(self.exogenous_dim, bool) or int(self.exogenous_dim) < 0:
            raise ValueError("exogenous_dim must be a non-negative integer")
        if isinstance(self.static_dim, bool) or int(self.static_dim) <= 0:
            raise ValueError("static_dim must be a positive integer")


class GlobalForecastModel(nn.Module, ABC):
    """Base estricta para modelos globales con salida directa multi-horizonte."""

    def __init__(self, dimensions: GlobalModelDimensions) -> None:
        super().__init__()
        dimensions.validate()
        self.dimensions = dimensions

    def representation_contract(self) -> Mapping[str, object]:
        """Describe la representación causal sin exponer identidad de serie."""

        use_modalities = bool(getattr(self, "use_modality_specific_encoders", False))
        contract: dict[str, object] = {
            "stage": (
                "GTRM_STAGE_2_3_MODALITY_SPECIFIC_INPUT_ENCODERS"
                if use_modalities
                else "GTRM_STAGE_1_GLOBAL_REPRESENTATION_BASE"
            ),
            "model_inputs": MODEL_INPUT_FIELDS,
            "output_field": GLOBAL_OUTPUT_FIELD,
            "latent_field": GLOBAL_LATENT_FIELD,
            "history_embedding_shape": ("batch", "latent_dim"),
            "forecast_shape": ("batch", self.dimensions.horizon, 1),
            "causal": True,
            "series_identity_in_forward": False,
            "static_context_is_semantic_not_identity": True,
            "agnostic_auxiliary_heads": ("event", "magnitude", "direction"),
            "use_modality_specific_encoders": use_modalities,
        }
        if use_modalities:
            contract["modality_encoders"] = {
                "y_context": "target_encoder",
                "x_history": "historical_exogenous_encoder",
                "x_future": "future_exogenous_encoder_per_horizon",
                "x_static": "static_context_encoder",
                "fusion": "history_fusion_encoder",
            }
        return contract

    @abstractmethod
    def forward(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> GlobalModelOutput:
        """Pronostica el horizonte sin recibir identidad de serie."""

    def _validate_and_prepare(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (y_context, x_history, x_future, x_static)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("All global model inputs must be torch.Tensor instances")

        if y_context.ndim != 3 or y_context.shape[-1] != 1:
            raise ValueError("y_context must have shape [batch, window_size, 1]")
        if x_history.ndim != 3:
            raise ValueError("x_history must have shape [batch, window_size, exogenous_dim]")
        if x_future.ndim != 3:
            raise ValueError("x_future must have shape [batch, horizon, exogenous_dim]")
        if x_static.ndim != 2:
            raise ValueError("x_static must have shape [batch, static_dim]")

        batch_size = y_context.shape[0]
        expected_history = (
            batch_size,
            self.dimensions.window_size,
            self.dimensions.exogenous_dim,
        )
        expected_future = (
            batch_size,
            self.dimensions.horizon,
            self.dimensions.exogenous_dim,
        )
        if tuple(y_context.shape[1:]) != (self.dimensions.window_size, 1):
            raise ValueError(
                "y_context window does not match model dimensions: "
                f"expected {self.dimensions.window_size}, got {y_context.shape[1]}"
            )
        if tuple(x_history.shape) != expected_history:
            raise ValueError(
                f"x_history shape mismatch: expected {expected_history}, "
                f"got {tuple(x_history.shape)}"
            )
        if tuple(x_future.shape) != expected_future:
            raise ValueError(
                f"x_future shape mismatch: expected {expected_future}, "
                f"got {tuple(x_future.shape)}"
            )
        expected_static = (batch_size, self.dimensions.static_dim)
        if tuple(x_static.shape) != expected_static:
            raise ValueError(
                f"x_static shape mismatch: expected {expected_static}, got {tuple(x_static.shape)}"
            )

        device = y_context.device
        dtype = y_context.dtype
        if not dtype.is_floating_point:
            raise TypeError("y_context must use a floating-point dtype")
        for name, value in (
            ("x_history", x_history),
            ("x_future", x_future),
            ("x_static", x_static),
        ):
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as y_context")
            if value.dtype != dtype:
                raise TypeError(f"{name} must use the same dtype as y_context")

        return y_context, x_history, x_future, x_static

    def _history_features(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((y_context, x_history), dim=-1)

    def _horizon_positions(self, reference: torch.Tensor) -> torch.Tensor:
        batch_size = reference.shape[0]
        positions = torch.linspace(
            0.0,
            1.0,
            steps=self.dimensions.horizon,
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, self.dimensions.horizon, 1)
        return positions.expand(batch_size, -1, -1)

    def _future_features(self, x_future: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        positions = self._horizon_positions(x_future)
        repeated_static = x_static.unsqueeze(1).expand(-1, self.dimensions.horizon, -1)
        return torch.cat((x_future, positions, repeated_static), dim=-1)

    def _configure_local_residual_decoder(
        self,
        cfg: Mapping[str, Any],
        *,
        latent_dim: int,
        activation: object,
        dropout: float,
        future_context_dim: int | None = None,
    ) -> None:
        """Configure the Stage 2 local residual head.

        The residual head is deliberately small and series-id free. It receives
        the same causal representation used by the global decoder plus known
        future covariates/static semantic context, and predicts only a delta:

            y_pred = y_global + delta_local

        Regularization is applied in ``global_forecast_loss`` so the head cannot
        silently absorb the full explanation.
        """

        use_local = bool(cfg.get("use_local_residual_decoder", False))
        self.use_local_residual_decoder = use_local
        self.local_residual_lambda = float(cfg.get("local_residual_lambda", 0.01))
        self.global_aux_alpha = float(cfg.get("global_aux_alpha", 0.2))
        if self.local_residual_lambda < 0.0:
            raise ValueError("local_residual_lambda must be non-negative")
        if self.global_aux_alpha < 0.0:
            raise ValueError("global_aux_alpha must be non-negative")
        self.local_residual_decoder: nn.Module | None = None
        if not use_local:
            return
        effective_future_dim = (
            self.dimensions.exogenous_dim + 1 + self.dimensions.static_dim
            if future_context_dim is None
            else int(future_context_dim)
        )
        if effective_future_dim <= 0:
            raise ValueError("future_context_dim must be positive")
        local_input_dim = int(latent_dim) + effective_future_dim
        local_hidden_size = _positive_int(
            cfg, "local_residual_hidden_size", max(16, min(128, int(latent_dim)))
        )
        local_layers = _positive_int(cfg, "local_residual_num_layers", 1)
        local_dropout = float(cfg.get("local_residual_dropout_rate", dropout))
        if not 0.0 <= local_dropout < 1.0:
            raise ValueError("local_residual_dropout_rate must be in [0, 1)")
        self.local_residual_decoder = _make_mlp(
            local_input_dim,
            local_hidden_size,
            1,
            num_hidden_layers=local_layers,
            activation=cfg.get("local_residual_activation", activation),
            dropout=local_dropout,
        )

    def _apply_local_residual(
        self,
        y_global: torch.Tensor,
        history_embedding: torch.Tensor,
        future_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the final forecast and Stage 2 diagnostic extras."""

        extras: dict[str, Any] = {
            "use_local_residual_decoder": bool(self.use_local_residual_decoder),
        }
        if not self.use_local_residual_decoder:
            return y_global, extras
        if self.local_residual_decoder is None:
            raise RuntimeError("Local residual decoder is enabled but missing")
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        residual_input = torch.cat((repeated_embedding, future_context), dim=-1)
        delta_local = self.local_residual_decoder(residual_input)
        if delta_local.shape != y_global.shape:
            raise ValueError("delta_local shape must match y_global")
        y_pred = y_global + delta_local
        extras.update(
            {
                GLOBAL_COMPONENT_FIELD: y_global,
                LOCAL_RESIDUAL_FIELD: delta_local,
                "local_residual_lambda": self.local_residual_lambda,
                "global_aux_alpha": self.global_aux_alpha,
                "local_residual_mean_abs": delta_local.detach().abs().mean(),
            }
        )
        return y_pred, extras

    def _configure_agnostic_auxiliary_heads(
        self,
        cfg: Mapping[str, Any],
        *,
        latent_dim: int,
        activation: object,
        dropout: float,
        future_context_dim: int | None = None,
    ) -> None:
        """Configure Stage 2.2 auxiliary heads shared by saldo and variacion."""

        self.use_event_head = bool(cfg.get("use_event_head", False))
        self.use_magnitude_head = bool(cfg.get("use_magnitude_head", False))
        self.use_direction_head = bool(cfg.get("use_direction_head", False))
        self.event_loss_weight = float(cfg.get("event_loss_weight", 0.1))
        self.magnitude_loss_weight = float(cfg.get("magnitude_loss_weight", 0.1))
        self.direction_loss_weight = float(cfg.get("direction_loss_weight", 0.05))
        self.use_auxiliary_loss_block = bool(cfg.get("use_auxiliary_loss_block", False))
        self.auxiliary_loss_weight = float(cfg.get("auxiliary_loss_weight", 0.20))
        self.event_loss_share = float(cfg.get("event_loss_share", 0.40))
        self.magnitude_loss_share = float(cfg.get("magnitude_loss_share", 0.40))
        self.direction_loss_share = float(cfg.get("direction_loss_share", 0.20))
        for name, value in (
            ("event_loss_weight", self.event_loss_weight),
            ("magnitude_loss_weight", self.magnitude_loss_weight),
            ("direction_loss_weight", self.direction_loss_weight),
            ("auxiliary_loss_weight", self.auxiliary_loss_weight),
            ("event_loss_share", self.event_loss_share),
            ("magnitude_loss_share", self.magnitude_loss_share),
            ("direction_loss_share", self.direction_loss_share),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        share_sum = self.event_loss_share + self.magnitude_loss_share + self.direction_loss_share
        if self.use_auxiliary_loss_block and not math.isclose(share_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "event_loss_share + magnitude_loss_share + direction_loss_share must equal 1.0 "
                "when use_auxiliary_loss_block=True"
            )
        effective_future_dim = (
            self.dimensions.exogenous_dim + 1 + self.dimensions.static_dim
            if future_context_dim is None
            else int(future_context_dim)
        )
        if effective_future_dim <= 0:
            raise ValueError("future_context_dim must be positive")
        aux_input_dim = int(latent_dim) + effective_future_dim
        aux_hidden_size = _positive_int(
            cfg, "auxiliary_head_hidden_size", max(16, min(128, int(latent_dim)))
        )
        aux_layers = _positive_int(cfg, "auxiliary_head_num_layers", 1)
        aux_dropout = float(cfg.get("auxiliary_head_dropout_rate", dropout))
        if not 0.0 <= aux_dropout < 1.0:
            raise ValueError("auxiliary_head_dropout_rate must be in [0, 1)")
        self.event_head: nn.Module | None = None
        self.magnitude_head: nn.Module | None = None
        self.direction_head: nn.Module | None = None
        if self.use_event_head:
            self.event_head = _make_mlp(
                aux_input_dim,
                aux_hidden_size,
                1,
                num_hidden_layers=aux_layers,
                activation=cfg.get("auxiliary_head_activation", activation),
                dropout=aux_dropout,
            )
        if self.use_magnitude_head:
            self.magnitude_head = _make_mlp(
                aux_input_dim,
                aux_hidden_size,
                1,
                num_hidden_layers=aux_layers,
                activation=cfg.get("auxiliary_head_activation", activation),
                dropout=aux_dropout,
            )
        if self.use_direction_head:
            self.direction_head = _make_mlp(
                aux_input_dim,
                aux_hidden_size,
                3,
                num_hidden_layers=aux_layers,
                activation=cfg.get("auxiliary_head_activation", activation),
                dropout=aux_dropout,
            )

    def _apply_agnostic_auxiliary_heads(
        self,
        history_embedding: torch.Tensor,
        future_context: torch.Tensor,
    ) -> dict[str, Any]:
        """Predict event/magnitude/direction auxiliary tensors when enabled."""

        extras: dict[str, Any] = {
            "use_event_head": bool(getattr(self, "use_event_head", False)),
            "use_magnitude_head": bool(getattr(self, "use_magnitude_head", False)),
            "use_direction_head": bool(getattr(self, "use_direction_head", False)),
        }
        if not any((extras["use_event_head"], extras["use_magnitude_head"], extras["use_direction_head"])):
            return extras
        extras.update(
            {
                "use_auxiliary_loss_block": bool(getattr(self, "use_auxiliary_loss_block", False)),
                "auxiliary_loss_weight": float(getattr(self, "auxiliary_loss_weight", 0.0)),
                "event_loss_share": float(getattr(self, "event_loss_share", 0.0)),
                "magnitude_loss_share": float(getattr(self, "magnitude_loss_share", 0.0)),
                "direction_loss_share": float(getattr(self, "direction_loss_share", 0.0)),
            }
        )
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        aux_input = torch.cat((repeated_embedding, future_context), dim=-1)
        if extras["use_event_head"]:
            if self.event_head is None:
                raise RuntimeError("Event head is enabled but missing")
            extras[EVENT_LOGITS_FIELD] = self.event_head(aux_input)
            extras["event_loss_weight"] = self.event_loss_weight
        if extras["use_magnitude_head"]:
            if self.magnitude_head is None:
                raise RuntimeError("Magnitude head is enabled but missing")
            extras[MAGNITUDE_PRED_FIELD] = self.magnitude_head(aux_input)
            extras["magnitude_loss_weight"] = self.magnitude_loss_weight
        if extras["use_direction_head"]:
            if self.direction_head is None:
                raise RuntimeError("Direction head is enabled but missing")
            extras[DIRECTION_LOGITS_FIELD] = self.direction_head(aux_input)
            extras["direction_loss_weight"] = self.direction_loss_weight
        return extras

    def _configure_auxiliary_autoencoder(
        self,
        cfg: Mapping[str, Any],
        *,
        latent_dim: int,
        activation: object,
        dropout: float,
    ) -> None:
        use_auxiliary = cfg.get("use_auxiliary_autoencoder", True)
        if not isinstance(use_auxiliary, bool):
            raise TypeError("use_auxiliary_autoencoder must be a boolean")
        self.use_auxiliary_autoencoder = use_auxiliary
        self.beta_ae = float(cfg.get("beta_ae", 0.0)) if use_auxiliary else 0.0
        if not 0.0 <= self.beta_ae <= 10.0:
            raise ValueError("beta_ae must be in [0, 10]")
        self.context_reconstruction_head: nn.Module | None = None
        if use_auxiliary:
            self.ae_hidden_size = _positive_int(
                cfg, "ae_hidden_size", max(32, latent_dim)
            )
            self.ae_num_layers = _positive_int(cfg, "ae_num_layers", 1)
            self.context_reconstruction_head = _make_mlp(
                latent_dim,
                self.ae_hidden_size,
                self.dimensions.window_size,
                num_hidden_layers=self.ae_num_layers,
                activation=cfg.get("ae_activation", activation),
                dropout=float(cfg.get("ae_dropout_rate", dropout)),
            )

    def _output_with_reconstruction(
        self,
        prediction: torch.Tensor,
        history_embedding: torch.Tensor,
        y_context: torch.Tensor,
        *,
        extras: Mapping[str, Any] | None = None,
        losses: Mapping[str, torch.Tensor] | None = None,
    ) -> GlobalModelOutput:
        if not self.use_auxiliary_autoencoder:
            combined_extras = dict(extras or {})
            combined_extras.update(
                {
                    "use_auxiliary_autoencoder": False,
                    "beta_ae": 0.0,
                }
            )
            return _global_output(
                prediction,
                history_embedding=history_embedding,
                reconstruction=None,
                extras=combined_extras,
                losses=losses,
            )
        if self.context_reconstruction_head is None:
            raise RuntimeError("Auxiliary autoencoder is enabled but its head is missing")
        reconstruction = self.context_reconstruction_head(history_embedding).unsqueeze(-1)
        reconstruction_loss = (reconstruction - y_context).square().mean()
        combined_losses = dict(losses or {})
        combined_losses.update({
            "reconstruction": reconstruction_loss,
            "weighted_reconstruction": self.beta_ae * reconstruction_loss,
        })
        combined_extras = dict(extras or {})
        combined_extras["use_auxiliary_autoencoder"] = True
        combined_extras["beta_ae"] = self.beta_ae
        return _global_output(
            prediction,
            history_embedding=history_embedding,
            reconstruction=reconstruction,
            extras=combined_extras,
            losses=combined_losses,
        )

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            for name, parameter in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)


def _activation(name: object) -> nn.Module:
    key = str(name or "gelu").strip().lower()
    try:
        return _ACTIVATIONS[key]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported activation={name!r}; expected {tuple(_ACTIVATIONS)}"
        ) from exc


def _positive_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _dropout(cfg: Mapping[str, Any]) -> float:
    value = float(cfg.get("dropout_rate", cfg.get("dropout", 0.1)))
    if not 0.0 <= value < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    return value


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_hidden_layers: int,
    activation: object,
    dropout: float,
) -> nn.Sequential:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(num_hidden_layers):
        layers.extend(
            (
                nn.Linear(current, hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
            )
        )
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class _ZeroVectorEncoder(nn.Module):
    """Return a deterministic zero representation for an empty modality."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros((values.shape[0], self.output_dim))


class _GlobalMLPBase(GlobalForecastModel):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        dimensions: GlobalModelDimensions,
    ) -> None:
        super().__init__(dimensions)
        self.cfg = dict(cfg)
        self.dropout_rate = _dropout(cfg)
        self.latent_dim = _positive_int(cfg, "latent_dim", 64)
        self.encoder_hidden_size = _positive_int(cfg, "enc_hidden_size", 128)
        self.encoder_layers = _positive_int(cfg, "enc_num_layers", 2)
        self.decoder_hidden_size = _positive_int(cfg, "dec_hidden_size", 128)
        self.decoder_layers = _positive_int(cfg, "dec_num_layers", 2)
        self.activation_name = cfg.get("activation", cfg.get("enc_activation", "gelu"))
        self.use_modality_specific_encoders = bool(
            cfg.get("use_modality_specific_encoders", False)
        )

        history_feature_dim = 1 + dimensions.exogenous_dim
        self.history_input_dim = dimensions.window_size * history_feature_dim + dimensions.static_dim

        if self.use_modality_specific_encoders:
            self._configure_mlp_modality_encoders(cfg)
            self.future_context_dim = self.future_encoder_dim + self.static_encoder_dim
        else:
            self.future_context_dim = dimensions.exogenous_dim + 1 + dimensions.static_dim

        self.future_input_dim = self.latent_dim + self.future_context_dim
        self.decoder = _make_mlp(
            self.future_input_dim,
            self.decoder_hidden_size,
            1,
            num_hidden_layers=self.decoder_layers,
            activation=cfg.get("dec_activation", self.activation_name),
            dropout=self.dropout_rate,
        )
        self._configure_auxiliary_autoencoder(
            cfg,
            latent_dim=self.latent_dim,
            activation=self.activation_name,
            dropout=self.dropout_rate,
        )
        self._configure_local_residual_decoder(
            cfg,
            latent_dim=self.latent_dim,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            future_context_dim=self.future_context_dim,
        )
        self._configure_agnostic_auxiliary_heads(
            cfg,
            latent_dim=self.latent_dim,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            future_context_dim=self.future_context_dim,
        )

    def _configure_mlp_modality_encoders(self, cfg: Mapping[str, Any]) -> None:
        activation = cfg.get("modality_encoder_activation", self.activation_name)
        dropout = float(cfg.get("modality_encoder_dropout_rate", self.dropout_rate))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("modality_encoder_dropout_rate must be in [0, 1)")
        self.target_encoder_dim = _positive_int(cfg, "target_encoder_dim", 32)
        self.historical_encoder_dim = _positive_int(cfg, "historical_encoder_dim", 32)
        self.future_encoder_dim = _positive_int(cfg, "future_encoder_dim", 32)
        self.static_encoder_dim = _positive_int(cfg, "static_encoder_dim", 16)
        self.fusion_hidden_size = _positive_int(cfg, "fusion_hidden_size", 64)
        target_layers = _positive_int(cfg, "target_encoder_num_layers", 1)
        historical_layers = _positive_int(cfg, "historical_encoder_num_layers", 1)
        future_layers = _positive_int(cfg, "future_encoder_num_layers", 1)
        static_layers = _positive_int(cfg, "static_encoder_num_layers", 1)
        fusion_layers = _positive_int(cfg, "fusion_num_layers", 1)

        self.target_encoder = _make_mlp(
            self.dimensions.window_size,
            max(self.target_encoder_dim, self.fusion_hidden_size),
            self.target_encoder_dim,
            num_hidden_layers=target_layers,
            activation=activation,
            dropout=dropout,
        )
        historical_input_dim = self.dimensions.window_size * self.dimensions.exogenous_dim
        self.historical_exogenous_encoder = (
            _make_mlp(
                historical_input_dim,
                max(self.historical_encoder_dim, self.fusion_hidden_size),
                self.historical_encoder_dim,
                num_hidden_layers=historical_layers,
                activation=activation,
                dropout=dropout,
            )
            if historical_input_dim > 0
            else _ZeroVectorEncoder(self.historical_encoder_dim)
        )
        self.future_exogenous_encoder = _make_mlp(
            self.dimensions.exogenous_dim + 1,
            max(self.future_encoder_dim, self.fusion_hidden_size),
            self.future_encoder_dim,
            num_hidden_layers=future_layers,
            activation=activation,
            dropout=dropout,
        )
        self.static_context_encoder = _make_mlp(
            self.dimensions.static_dim,
            max(self.static_encoder_dim, self.fusion_hidden_size),
            self.static_encoder_dim,
            num_hidden_layers=static_layers,
            activation=activation,
            dropout=dropout,
        )
        fusion_input_dim = (
            self.target_encoder_dim
            + self.historical_encoder_dim
            + self.static_encoder_dim
        )
        # ``encoder`` remains the shared fusion module for backward-compatible
        # introspection while raw modalities now have independent encoders.
        self.encoder = _make_mlp(
            fusion_input_dim,
            self.fusion_hidden_size,
            self.latent_dim,
            num_hidden_layers=fusion_layers,
            activation=activation,
            dropout=dropout,
        )

    def _encode_modalities(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        target_embedding = self.target_encoder(y_context.squeeze(-1))
        historical_embedding = self.historical_exogenous_encoder(
            x_history.flatten(start_dim=1)
        )
        static_embedding = self.static_context_encoder(x_static)
        history_embedding = self.encoder(
            torch.cat((target_embedding, historical_embedding, static_embedding), dim=-1)
        )
        future_inputs = torch.cat((x_future, self._horizon_positions(x_future)), dim=-1)
        future_embedding = self.future_exogenous_encoder(future_inputs)
        repeated_static = static_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        future_context = torch.cat((future_embedding, repeated_static), dim=-1)
        extras = {
            "target_embedding": target_embedding,
            "historical_exogenous_embedding": historical_embedding,
            "future_exogenous_embedding": future_embedding,
            "static_context_embedding": static_embedding,
        }
        return history_embedding, future_context, extras

    def _decode(
        self, history_embedding: torch.Tensor, future_context: torch.Tensor
    ) -> torch.Tensor:
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        decoder_input = torch.cat((repeated_embedding, future_context), dim=-1)
        return self.decoder(decoder_input)


class GlobalMLPEncoderDecoder(_GlobalMLPBase):
    """MLP global con fusión opcional de encoders por modalidad."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        window_size: int,
        horizon: int,
        exogenous_dim: int,
        static_dim: int = 1,
    ) -> None:
        dimensions = GlobalModelDimensions(window_size, horizon, exogenous_dim, static_dim)
        super().__init__(cfg, dimensions)
        if not self.use_modality_specific_encoders:
            self.encoder = _make_mlp(
                self.history_input_dim,
                self.encoder_hidden_size,
                self.latent_dim,
                num_hidden_layers=self.encoder_layers,
                activation=self.activation_name,
                dropout=self.dropout_rate,
            )
        self.apply(self._initialize_weights)

    def forward(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> GlobalModelOutput:
        y_context, x_history, x_future, x_static = self._validate_and_prepare(
            y_context, x_history, x_future, x_static
        )
        modality_extras: dict[str, Any] = {
            "use_modality_specific_encoders": self.use_modality_specific_encoders
        }
        if self.use_modality_specific_encoders:
            history_embedding, future_context, encoded = self._encode_modalities(
                y_context, x_history, x_future, x_static
            )
            modality_extras.update(encoded)
        else:
            history = self._history_features(y_context, x_history)
            history_embedding = self.encoder(
                torch.cat((history.flatten(start_dim=1), x_static), dim=-1)
            )
            future_context = self._future_features(x_future, x_static)

        y_global = self._decode(history_embedding, future_context)
        prediction, residual_extras = self._apply_local_residual(
            y_global, history_embedding, future_context
        )
        auxiliary_extras = self._apply_agnostic_auxiliary_heads(
            history_embedding, future_context
        )
        modality_extras.update(residual_extras)
        modality_extras.update(auxiliary_extras)
        return self._output_with_reconstruction(
            prediction, history_embedding, y_context, extras=modality_extras
        )


class GlobalMLPVAEEncoderDecoder(_GlobalMLPBase):
    """VAE global con fusión opcional de encoders por modalidad."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        window_size: int,
        horizon: int,
        exogenous_dim: int,
        static_dim: int = 1,
    ) -> None:
        dimensions = GlobalModelDimensions(window_size, horizon, exogenous_dim, static_dim)
        super().__init__(cfg, dimensions)
        if self.use_modality_specific_encoders:
            # Replace the deterministic fusion output with a shared hidden state
            # from which mu/logvar are parameterized.
            fusion_input_dim = (
                self.target_encoder_dim
                + self.historical_encoder_dim
                + self.static_encoder_dim
            )
            fusion_layers = _positive_int(cfg, "fusion_num_layers", 1)
            activation = cfg.get("modality_encoder_activation", self.activation_name)
            dropout = float(cfg.get("modality_encoder_dropout_rate", self.dropout_rate))
            del self.encoder
            self.encoder_backbone = _make_mlp(
                fusion_input_dim,
                self.fusion_hidden_size,
                self.fusion_hidden_size,
                num_hidden_layers=fusion_layers,
                activation=activation,
                dropout=dropout,
            )
            variational_hidden_dim = self.fusion_hidden_size
        else:
            self.encoder_backbone = _make_mlp(
                self.history_input_dim,
                self.encoder_hidden_size,
                self.encoder_hidden_size,
                num_hidden_layers=self.encoder_layers,
                activation=self.activation_name,
                dropout=self.dropout_rate,
            )
            variational_hidden_dim = self.encoder_hidden_size
        self.fc_mu = nn.Linear(variational_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(variational_hidden_dim, self.latent_dim)
        self.logvar_min = float(cfg.get("logvar_min", -12.0))
        self.logvar_max = float(cfg.get("logvar_max", 8.0))
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        self.beta_kl = float(cfg.get("beta_kl", cfg.get("beta", 1.0)))
        if self.beta_kl < 0.0:
            raise ValueError("beta_kl must be non-negative")
        self.apply(self._initialize_weights)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> GlobalModelOutput:
        y_context, x_history, x_future, x_static = self._validate_and_prepare(
            y_context, x_history, x_future, x_static
        )
        modality_extras: dict[str, Any] = {
            "use_modality_specific_encoders": self.use_modality_specific_encoders
        }
        if self.use_modality_specific_encoders:
            target_embedding = self.target_encoder(y_context.squeeze(-1))
            historical_embedding = self.historical_exogenous_encoder(
                x_history.flatten(start_dim=1)
            )
            static_embedding = self.static_context_encoder(x_static)
            hidden = self.encoder_backbone(
                torch.cat((target_embedding, historical_embedding, static_embedding), dim=-1)
            )
            future_inputs = torch.cat((x_future, self._horizon_positions(x_future)), dim=-1)
            future_embedding = self.future_exogenous_encoder(future_inputs)
            future_context = torch.cat(
                (
                    future_embedding,
                    static_embedding.unsqueeze(1).expand(-1, self.dimensions.horizon, -1),
                ),
                dim=-1,
            )
            modality_extras.update(
                {
                    "target_embedding": target_embedding,
                    "historical_exogenous_embedding": historical_embedding,
                    "future_exogenous_embedding": future_embedding,
                    "static_context_embedding": static_embedding,
                }
            )
        else:
            history = self._history_features(y_context, x_history)
            hidden = self.encoder_backbone(
                torch.cat((history.flatten(start_dim=1), x_static), dim=-1)
            )
            future_context = self._future_features(x_future, x_static)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(self.logvar_min, self.logvar_max)
        history_embedding = self._sample(mu, logvar)
        y_global = self._decode(history_embedding, future_context)
        prediction, residual_extras = self._apply_local_residual(
            y_global, history_embedding, future_context
        )
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
        kl_loss = kl.mean()
        auxiliary_extras = self._apply_agnostic_auxiliary_heads(
            history_embedding, future_context
        )
        extras: dict[str, Any] = {
            "mu": mu,
            "logvar": logvar,
            "beta_kl": self.beta_kl,
            **modality_extras,
            **residual_extras,
            **auxiliary_extras,
        }
        return self._output_with_reconstruction(
            prediction,
            history_embedding,
            y_context,
            losses={"kl": kl_loss, "weighted_kl": self.beta_kl * kl_loss},
            extras=extras,
        )


class _GlobalRNNEncoderDecoder(GlobalForecastModel):
    """GRU global con encoders separados por modalidad cuando se habilitan."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        window_size: int,
        horizon: int,
        exogenous_dim: int,
        static_dim: int = 1,
        bidirectional_encoder: bool,
    ) -> None:
        dimensions = GlobalModelDimensions(window_size, horizon, exogenous_dim, static_dim)
        super().__init__(dimensions)
        self.cfg = dict(cfg)
        self.bidirectional_encoder = bool(bidirectional_encoder)
        self.hidden_size = _positive_int(cfg, "rnn_hidden_size", 64)
        self.encoder_layers = _positive_int(cfg, "rnn_num_layers", 1)
        self.decoder_layers = _positive_int(cfg, "decoder_num_layers", 1)
        self.latent_dim = _positive_int(cfg, "latent_dim", self.hidden_size)
        self.dropout_rate = _dropout(cfg)
        self.use_modality_specific_encoders = bool(
            cfg.get("use_modality_specific_encoders", False)
        )
        decoder_dropout = self.dropout_rate if self.decoder_layers > 1 else 0.0

        if self.use_modality_specific_encoders:
            self._configure_rnn_modality_encoders(cfg)
            self.future_context_dim = self.future_encoder_dim + self.static_encoder_dim
        else:
            history_feature_dim = 1 + exogenous_dim
            recurrent_dropout = self.dropout_rate if self.encoder_layers > 1 else 0.0
            self.encoder = nn.GRU(
                input_size=history_feature_dim,
                hidden_size=self.hidden_size,
                num_layers=self.encoder_layers,
                dropout=recurrent_dropout,
                bidirectional=self.bidirectional_encoder,
                batch_first=True,
            )
            encoder_output_dim = self.hidden_size * (2 if self.bidirectional_encoder else 1)
            self.history_projection = nn.Sequential(
                nn.Linear(encoder_output_dim + static_dim, self.latent_dim),
                _activation(cfg.get("rnn_activation", "tanh")),
                nn.Dropout(self.dropout_rate),
            )
            self.future_context_dim = exogenous_dim + 1 + static_dim

        self.decoder_state_projection = nn.Linear(self.latent_dim, self.hidden_size)
        self.decoder = nn.GRU(
            input_size=self.future_context_dim,
            hidden_size=self.hidden_size,
            num_layers=self.decoder_layers,
            dropout=decoder_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.output_head = _make_mlp(
            self.hidden_size + self.latent_dim,
            _positive_int(cfg, "dec_hidden_size", self.hidden_size),
            1,
            num_hidden_layers=_positive_int(cfg, "dec_num_layers", 1),
            activation=cfg.get("dec_activation", "gelu"),
            dropout=self.dropout_rate,
        )
        self._configure_auxiliary_autoencoder(
            cfg,
            latent_dim=self.latent_dim,
            activation=cfg.get("rnn_activation", "tanh"),
            dropout=self.dropout_rate,
        )
        self._configure_local_residual_decoder(
            cfg,
            latent_dim=self.latent_dim,
            activation=cfg.get("local_residual_activation", cfg.get("rnn_activation", "tanh")),
            dropout=self.dropout_rate,
            future_context_dim=self.future_context_dim,
        )
        self._configure_agnostic_auxiliary_heads(
            cfg,
            latent_dim=self.latent_dim,
            activation=cfg.get("auxiliary_head_activation", cfg.get("rnn_activation", "tanh")),
            dropout=self.dropout_rate,
            future_context_dim=self.future_context_dim,
        )
        self.apply(self._initialize_weights)

    def _configure_rnn_modality_encoders(self, cfg: Mapping[str, Any]) -> None:
        activation = cfg.get("modality_encoder_activation", cfg.get("rnn_activation", "tanh"))
        dropout = float(cfg.get("modality_encoder_dropout_rate", self.dropout_rate))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("modality_encoder_dropout_rate must be in [0, 1)")
        self.target_encoder_dim = _positive_int(cfg, "target_encoder_dim", self.hidden_size)
        self.historical_encoder_dim = _positive_int(cfg, "historical_encoder_dim", self.hidden_size)
        self.future_encoder_dim = _positive_int(cfg, "future_encoder_dim", 32)
        self.static_encoder_dim = _positive_int(cfg, "static_encoder_dim", 16)
        self.fusion_hidden_size = _positive_int(cfg, "fusion_hidden_size", max(64, self.latent_dim))
        self.target_encoder_layers = _positive_int(cfg, "target_encoder_num_layers", 1)
        self.historical_encoder_layers = _positive_int(cfg, "historical_encoder_num_layers", 1)
        future_layers = _positive_int(cfg, "future_encoder_num_layers", 1)
        static_layers = _positive_int(cfg, "static_encoder_num_layers", 1)
        fusion_layers = _positive_int(cfg, "fusion_num_layers", 1)
        target_dropout = dropout if self.target_encoder_layers > 1 else 0.0
        historical_dropout = dropout if self.historical_encoder_layers > 1 else 0.0

        self.target_encoder = nn.GRU(
            input_size=1,
            hidden_size=self.target_encoder_dim,
            num_layers=self.target_encoder_layers,
            dropout=target_dropout,
            bidirectional=self.bidirectional_encoder,
            batch_first=True,
        )
        self.historical_exogenous_encoder: nn.Module
        if self.dimensions.exogenous_dim > 0:
            self.historical_exogenous_encoder = nn.GRU(
                input_size=self.dimensions.exogenous_dim,
                hidden_size=self.historical_encoder_dim,
                num_layers=self.historical_encoder_layers,
                dropout=historical_dropout,
                bidirectional=self.bidirectional_encoder,
                batch_first=True,
            )
        else:
            self.historical_exogenous_encoder = _ZeroVectorEncoder(
                self.historical_encoder_dim
            )
        self.future_exogenous_encoder = _make_mlp(
            self.dimensions.exogenous_dim + 1,
            max(self.future_encoder_dim, self.fusion_hidden_size),
            self.future_encoder_dim,
            num_hidden_layers=future_layers,
            activation=activation,
            dropout=dropout,
        )
        self.static_context_encoder = _make_mlp(
            self.dimensions.static_dim,
            max(self.static_encoder_dim, self.fusion_hidden_size),
            self.static_encoder_dim,
            num_hidden_layers=static_layers,
            activation=activation,
            dropout=dropout,
        )
        directions = 2 if self.bidirectional_encoder else 1
        self.target_state_projection: nn.Module = (
            nn.Linear(self.target_encoder_dim * directions, self.target_encoder_dim)
            if directions > 1
            else nn.Identity()
        )
        self.historical_state_projection: nn.Module = (
            nn.Linear(self.historical_encoder_dim * directions, self.historical_encoder_dim)
            if directions > 1
            else nn.Identity()
        )
        fusion_input_dim = (
            self.target_encoder_dim
            + self.historical_encoder_dim
            + self.static_encoder_dim
        )
        self.history_projection = _make_mlp(
            fusion_input_dim,
            self.fusion_hidden_size,
            self.latent_dim,
            num_hidden_layers=fusion_layers,
            activation=activation,
            dropout=dropout,
        )

    @property
    def encoder(self) -> nn.Module:
        """Backward-compatible alias for the primary historical encoder."""
        if self.use_modality_specific_encoders:
            return self.target_encoder
        return self._modules["encoder"]

    @encoder.setter
    def encoder(self, module: nn.Module) -> None:
        self._modules["encoder"] = module

    def _top_recurrent_state(
        self,
        hidden: torch.Tensor,
        *,
        layers: int,
        hidden_size: int,
    ) -> torch.Tensor:
        batch_size = hidden.shape[1]
        directions = 2 if self.bidirectional_encoder else 1
        hidden = hidden.view(layers, directions, batch_size, hidden_size)
        top_layer = hidden[-1]
        if self.bidirectional_encoder:
            return torch.cat((top_layer[0], top_layer[1]), dim=-1)
        return top_layer[0]

    def _top_encoder_state(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._top_recurrent_state(
            hidden,
            layers=self.encoder_layers,
            hidden_size=self.hidden_size,
        )

    def _encode_modalities(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        _, target_hidden = self.target_encoder(y_context)
        target_embedding = self.target_state_projection(
            self._top_recurrent_state(
                target_hidden,
                layers=self.target_encoder_layers,
                hidden_size=self.target_encoder_dim,
            )
        )
        if self.dimensions.exogenous_dim > 0:
            _, historical_hidden = self.historical_exogenous_encoder(x_history)
            historical_embedding = self.historical_state_projection(
                self._top_recurrent_state(
                    historical_hidden,
                    layers=self.historical_encoder_layers,
                    hidden_size=self.historical_encoder_dim,
                )
            )
        else:
            historical_embedding = self.historical_exogenous_encoder(x_history)
        static_embedding = self.static_context_encoder(x_static)
        history_embedding = self.history_projection(
            torch.cat((target_embedding, historical_embedding, static_embedding), dim=-1)
        )
        future_inputs = torch.cat((x_future, self._horizon_positions(x_future)), dim=-1)
        future_embedding = self.future_exogenous_encoder(future_inputs)
        future_context = torch.cat(
            (
                future_embedding,
                static_embedding.unsqueeze(1).expand(-1, self.dimensions.horizon, -1),
            ),
            dim=-1,
        )
        extras = {
            "target_embedding": target_embedding,
            "historical_exogenous_embedding": historical_embedding,
            "future_exogenous_embedding": future_embedding,
            "static_context_embedding": static_embedding,
        }
        return history_embedding, future_context, extras

    def forward(
        self,
        y_context: torch.Tensor,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> GlobalModelOutput:
        y_context, x_history, x_future, x_static = self._validate_and_prepare(
            y_context, x_history, x_future, x_static
        )
        modality_extras: dict[str, Any] = {
            "use_modality_specific_encoders": self.use_modality_specific_encoders
        }
        if self.use_modality_specific_encoders:
            history_embedding, future_context, encoded = self._encode_modalities(
                y_context, x_history, x_future, x_static
            )
            modality_extras.update(encoded)
        else:
            history = self._history_features(y_context, x_history)
            _, encoder_hidden = self.encoder(history)
            top_state = self._top_encoder_state(encoder_hidden)
            history_embedding = self.history_projection(torch.cat((top_state, x_static), dim=-1))
            future_context = self._future_features(x_future, x_static)

        initial_state = torch.tanh(self.decoder_state_projection(history_embedding))
        initial_state = initial_state.unsqueeze(0).expand(self.decoder_layers, -1, -1).contiguous()
        decoder_output, _ = self.decoder(future_context, initial_state)
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        head_input = torch.cat((decoder_output, repeated_embedding), dim=-1)
        y_global = self.output_head(head_input)
        prediction, residual_extras = self._apply_local_residual(
            y_global, history_embedding, future_context
        )
        auxiliary_extras = self._apply_agnostic_auxiliary_heads(
            history_embedding, future_context
        )
        modality_extras.update(residual_extras)
        modality_extras.update(auxiliary_extras)
        return self._output_with_reconstruction(
            prediction, history_embedding, y_context, extras=modality_extras
        )


class GlobalRNNEncoderDecoder(_GlobalRNNEncoderDecoder):
    """GRU global con encoders históricos unidireccionales."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        window_size: int,
        horizon: int,
        exogenous_dim: int,
        static_dim: int = 1,
    ) -> None:
        super().__init__(
            cfg,
            window_size=window_size,
            horizon=horizon,
            exogenous_dim=exogenous_dim,
            static_dim=static_dim,
            bidirectional_encoder=False,
        )


class GlobalRNNBiEncoderDecoder(_GlobalRNNEncoderDecoder):
    """GRU global con encoders históricos bidireccionales sobre contexto observado."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        window_size: int,
        horizon: int,
        exogenous_dim: int,
        static_dim: int = 1,
    ) -> None:
        super().__init__(
            cfg,
            window_size=window_size,
            horizon=horizon,
            exogenous_dim=exogenous_dim,
            static_dim=static_dim,
            bidirectional_encoder=True,
        )


GLOBAL_MODEL_REGISTRY: Final[Mapping[str, Type[GlobalForecastModel]]] = {
    "mlp": GlobalMLPEncoderDecoder,
    "mlp_vae": GlobalMLPVAEEncoderDecoder,
    "rnn": GlobalRNNEncoderDecoder,
    "rnn_bi": GlobalRNNBiEncoderDecoder,
}


def get_global_model_class(name: str) -> Type[GlobalForecastModel]:
    """Devuelve la arquitectura global sin construirla."""

    key = str(name or "").strip().lower()
    try:
        return GLOBAL_MODEL_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown global model {name!r}. Available: {tuple(GLOBAL_MODEL_REGISTRY)}"
        ) from exc


def build_global_model(
    name: str,
    cfg: Mapping[str, Any],
    *,
    window_size: int,
    horizon: int,
    exogenous_dim: int,
    static_dim: int = 1,
) -> GlobalForecastModel:
    """Construye una arquitectura global bajo el contrato canónico."""

    model_class = get_global_model_class(name)
    return model_class(
        cfg,
        window_size=window_size,
        horizon=horizon,
        exogenous_dim=exogenous_dim,
        static_dim=static_dim,
    )


def list_global_models() -> Tuple[str, ...]:
    """Lista estable de arquitecturas globales soportadas."""

    return tuple(GLOBAL_MODEL_REGISTRY)


def validate_global_forward_contract(
    model_class: Type[GlobalForecastModel],
) -> Tuple[str, ...]:
    """Comprueba que ``forward`` expone sólo los cuatro tensores autorizados."""

    parameters = tuple(inspect.signature(model_class.forward).parameters)
    fields = parameters[1:] if parameters and parameters[0] == "self" else parameters
    if fields != MODEL_INPUT_FIELDS:
        raise TypeError(
            f"Invalid forward contract for {model_class.__name__}: "
            f"expected {MODEL_INPUT_FIELDS}, got {fields}"
        )
    return fields


if tuple(GLOBAL_MODEL_REGISTRY) != SUPPORTED_ARCHITECTURES:
    raise RuntimeError(
        "Global registry and supported architecture contract are inconsistent: "
        f"{tuple(GLOBAL_MODEL_REGISTRY)} != {SUPPORTED_ARCHITECTURES}"
    )
for _model_class in GLOBAL_MODEL_REGISTRY.values():
    validate_global_forward_contract(_model_class)
