"""Arquitecturas globales de Deep Representation Learning para forecasting.

Checkpoint 3 define únicamente cuatro modelos globales bajo un contrato común:

- ``GlobalMLPEncoderDecoder``;
- ``GlobalMLPVAEEncoderDecoder``;
- ``GlobalRNNEncoderDecoder``;
- ``GlobalRNNBiEncoderDecoder``.

Todos comparten pesos entre series y reciben contexto escalado linealmente,
covariables temporales del calendario y un vector ``x_static`` no identificador
(tipo, divisa, escala contextual y edad causal). Generan el horizonte completo
con una única cabeza de forecasting. Un decoder autoencoder auxiliar puede
reconstruir el contexto escalado desde el espacio latente. Los identificadores
contables permanecen fuera de ``forward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Any, Dict, Final, Mapping, Sequence, Tuple, Type

import torch
from torch import nn

from global_contracts import MODEL_INPUT_FIELDS, SUPPORTED_ARCHITECTURES


GlobalModelOutput = Dict[str, Any]
GLOBAL_OUTPUT_FIELD: Final[str] = "y_pred"
GLOBAL_LATENT_FIELD: Final[str] = "history_embedding"
RECONSTRUCTION_FIELD: Final[str] = "context_reconstruction"


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

    def _future_features(self, x_future: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        batch_size = x_future.shape[0]
        positions = torch.linspace(
            0.0,
            1.0,
            steps=self.dimensions.horizon,
            device=x_future.device,
            dtype=x_future.dtype,
        ).view(1, self.dimensions.horizon, 1)
        positions = positions.expand(batch_size, -1, -1)
        repeated_static = x_static.unsqueeze(1).expand(-1, self.dimensions.horizon, -1)
        return torch.cat((x_future, positions, repeated_static), dim=-1)

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

        history_feature_dim = 1 + dimensions.exogenous_dim
        self.history_input_dim = dimensions.window_size * history_feature_dim + dimensions.static_dim
        self.future_input_dim = self.latent_dim + dimensions.exogenous_dim + 1 + dimensions.static_dim

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

    def _decode(
        self, history_embedding: torch.Tensor, x_future: torch.Tensor, x_static: torch.Tensor
    ) -> torch.Tensor:
        future_features = self._future_features(x_future, x_static)
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        decoder_input = torch.cat((repeated_embedding, future_features), dim=-1)
        return self.decoder(decoder_input)


class GlobalMLPEncoderDecoder(_GlobalMLPBase):
    """MLP global: codifica la ventana completa y decodifica cada paso futuro."""

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
        history = self._history_features(y_context, x_history)
        history_embedding = self.encoder(torch.cat((history.flatten(start_dim=1), x_static), dim=-1))
        prediction = self._decode(history_embedding, x_future, x_static)
        return self._output_with_reconstruction(
            prediction, history_embedding, y_context
        )


class GlobalMLPVAEEncoderDecoder(_GlobalMLPBase):
    """VAE global: representación histórica estocástica con regularización KL."""

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
        self.encoder_backbone = _make_mlp(
            self.history_input_dim,
            self.encoder_hidden_size,
            self.encoder_hidden_size,
            num_hidden_layers=self.encoder_layers,
            activation=self.activation_name,
            dropout=self.dropout_rate,
        )
        self.fc_mu = nn.Linear(self.encoder_hidden_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_hidden_size, self.latent_dim)
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
        history = self._history_features(y_context, x_history)
        hidden = self.encoder_backbone(torch.cat((history.flatten(start_dim=1), x_static), dim=-1))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(self.logvar_min, self.logvar_max)
        history_embedding = self._sample(mu, logvar)
        prediction = self._decode(history_embedding, x_future, x_static)
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
        kl_loss = kl.mean()
        return self._output_with_reconstruction(
            prediction,
            history_embedding,
            y_context,
            losses={"kl": kl_loss, "weighted_kl": self.beta_kl * kl_loss},
            extras={"mu": mu, "logvar": logvar, "beta_kl": self.beta_kl},
        )


class _GlobalRNNEncoderDecoder(GlobalForecastModel):
    """GRU encoder-decoder global; el decoder futuro siempre es causal."""

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

        history_feature_dim = 1 + exogenous_dim
        future_feature_dim = exogenous_dim + 1 + static_dim
        recurrent_dropout = self.dropout_rate if self.encoder_layers > 1 else 0.0
        decoder_dropout = self.dropout_rate if self.decoder_layers > 1 else 0.0

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
        self.decoder_state_projection = nn.Linear(self.latent_dim, self.hidden_size)
        self.decoder = nn.GRU(
            input_size=future_feature_dim,
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
        self.apply(self._initialize_weights)

    def _top_encoder_state(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[1]
        directions = 2 if self.bidirectional_encoder else 1
        hidden = hidden.view(self.encoder_layers, directions, batch_size, self.hidden_size)
        top_layer = hidden[-1]
        if self.bidirectional_encoder:
            return torch.cat((top_layer[0], top_layer[1]), dim=-1)
        return top_layer[0]

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
        history = self._history_features(y_context, x_history)
        _, encoder_hidden = self.encoder(history)
        top_state = self._top_encoder_state(encoder_hidden)
        history_embedding = self.history_projection(torch.cat((top_state, x_static), dim=-1))

        initial_state = torch.tanh(self.decoder_state_projection(history_embedding))
        initial_state = initial_state.unsqueeze(0).expand(self.decoder_layers, -1, -1).contiguous()
        decoder_output, _ = self.decoder(self._future_features(x_future, x_static), initial_state)
        repeated_embedding = history_embedding.unsqueeze(1).expand(
            -1, self.dimensions.horizon, -1
        )
        head_input = torch.cat((decoder_output, repeated_embedding), dim=-1)
        prediction = self.output_head(head_input)
        return self._output_with_reconstruction(
            prediction, history_embedding, y_context
        )


class GlobalRNNEncoderDecoder(_GlobalRNNEncoderDecoder):
    """GRU global con encoder histórico unidireccional y decoder causal."""

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
    """GRU global con encoder histórico bidireccional y decoder causal."""

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
