# models.py
from __future__ import annotations

from typing import Any, Dict, List, Type, Callable, Optional

import torch
import torch.nn as nn

ModelOutput = Dict[str, Any]

_ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "Mish": nn.Mish,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
}


# ---------------------------------------------------------------------
# MLP (MLP_E_D)
# ---------------------------------------------------------------------
class MultiHeadMLPModel(nn.Module):
    """MLP encoder + MLP head (seq2seq regression). Returns dict output."""

    def __init__(self, cfg: Dict[str, Any], input_dim: int) -> None:
        super().__init__()

        mlp_layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(int(cfg.get("enc_num_layers", 2))):
            mlp_layers.append(nn.Linear(in_dim, int(cfg["enc_hidden_size"])))
            mlp_layers.append(_ACTIVATIONS[cfg.get("enc_activation", "ReLU")]())
            mlp_layers.append(nn.Dropout(float(cfg["dropout_rate"])))
            in_dim = int(cfg["enc_hidden_size"])
        self.encoder = nn.Sequential(*mlp_layers)
        self.enc_out_dim = in_dim

        head_layers: List[nn.Module] = []
        in_h = self.enc_out_dim
        for _ in range(int(cfg.get("reg_num_hidden_layers", 1))):
            head_layers.append(nn.Linear(in_h, int(cfg["reg_hidden_size"])))
            head_layers.append(_ACTIVATIONS[cfg.get("reg_activation", "ReLU")]())
            head_layers.append(nn.Dropout(float(cfg["dropout_rate"])))
            in_h = int(cfg["reg_hidden_size"])
        head_layers.append(nn.Linear(in_h, 1))
        self.head = nn.Sequential(*head_layers)

        self.apply(self._xavier_init_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, W, F = x.shape
        x_flat = x.reshape(B * W, F)
        h_flat = self.encoder(x_flat)
        y_flat = self.head(h_flat)
        y = y_flat.view(B, W)
        return {"y_pred": y}

    @staticmethod
    def _xavier_init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
# VAE (MLP_VaE_D)
# ---------------------------------------------------------------------
class VAEEncoder(nn.Module):
    """VAE encoder MLP -> (mu, logvar) -> reparam z + KL divergence per sample."""

    def __init__(self, cfg: Dict[str, Any], input_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(int(cfg.get("enc_num_layers", 2))):
            layers.append(nn.Linear(in_dim, int(cfg["enc_hidden_size"])))
            layers.append(_ACTIVATIONS[cfg.get("enc_activation", "ReLU")]())
            layers.append(nn.Dropout(float(cfg["dropout_rate"])))
            in_dim = int(cfg["enc_hidden_size"])
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, int(cfg["latent_dim"]))
        self.fc_logvar = nn.Linear(in_dim, int(cfg["latent_dim"]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return z, kl


class MultiHeadVAEModel(nn.Module):
    """VAE encoder + MLP head (seq2seq). Returns dict output with KL loss."""

    def __init__(self, cfg: Dict[str, Any], input_dim: int) -> None:
        super().__init__()
        self.encoder = VAEEncoder(cfg, input_dim)

        # beta normalmente vive en cfg (lo usará Scientist para ponderar KL)
        self.beta = float(cfg.get("beta", 1.0))
        latent_dim = int(cfg["latent_dim"])
        self.enc_out_dim = latent_dim

        head_layers: List[nn.Module] = []
        in_h = latent_dim
        for _ in range(int(cfg.get("reg_num_hidden_layers", 1))):
            head_layers.append(nn.Linear(in_h, int(cfg["reg_hidden_size"])))
            head_layers.append(_ACTIVATIONS[cfg.get("reg_activation", "ReLU")]())
            head_layers.append(nn.Dropout(float(cfg["dropout_rate"])))
            in_h = int(cfg["reg_hidden_size"])
        head_layers.append(nn.Linear(in_h, 1))
        self.head = nn.Sequential(*head_layers)

        self.apply(self._xavier_init_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, W, F = x.shape
        x_flat = x.view(B * W, F)

        z_flat, kl_flat = self.encoder(x_flat)
        y = self.head(z_flat).view(B, W)
        kl_loss = kl_flat.mean()

        return {
            "y_pred": y,
            "losses": {"kl": kl_loss},
            # extras opcionales (útiles para debug/monitoring):
            # "extras": {"z": z_flat},
        }

    @staticmethod
    def _xavier_init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
# RNN (RNN_E_D / RNNBi_E_D)
# ---------------------------------------------------------------------
class RNNSeq2SeqModel(nn.Module):
    """RNN encoder (GRU/LSTM) + dense head (seq2seq). Returns dict output."""

    def __init__(self, cfg: Dict[str, Any], input_dim: int) -> None:
        super().__init__()

        rnn_cls = nn.LSTM if cfg.get("rnn_type", "GRU") == "LSTM" else nn.GRU
        self.bidirectional: bool = bool(cfg.get("bidirectional", False))

        self.encoder = rnn_cls(
            input_size=input_dim,
            hidden_size=int(cfg["rnn_hidden_size"]),
            num_layers=int(cfg["rnn_num_layers"]),
            dropout=float(cfg["dropout_rate"]) if int(cfg["rnn_num_layers"]) > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=False,  # (seq_len, batch, feat)
        )

        self.rnn_activation = _ACTIVATIONS[cfg.get("rnn_activation", "ReLU")]()
        self.rnn_dropout = nn.Dropout(float(cfg["dropout_rate"]))

        dirs = 2 if self.bidirectional else 1
        rnn_out_size = int(cfg["rnn_hidden_size"]) * dirs

        head_layers: List[nn.Module] = []
        in_dim_h = rnn_out_size
        for _ in range(int(cfg.get("reg_num_hidden_layers", 1))):
            head_layers.extend([
                nn.Linear(in_dim_h, int(cfg["reg_hidden_size"])),
                _ACTIVATIONS[cfg.get("reg_activation", "ReLU")](),
                nn.Dropout(float(cfg["dropout_rate"]))
            ])
            in_dim_h = int(cfg["reg_hidden_size"])
        head_layers.append(nn.Linear(in_dim_h, 1))
        self.head = nn.Sequential(*head_layers)

        self.decoder_dropout = nn.Dropout(float(cfg["dropout_rate"]))

        self.apply(self._xavier_init_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x: (B, W, F) -> encoder espera (W, B, F)
        x_seq, _ = self.encoder(x.permute(1, 0, 2))  # (W, B, H*dirs)

        x_bwh = x_seq.permute(1, 0, 2)              # (B, W, H*dirs)
        x_bwh = self.rnn_activation(x_bwh)
        x_bwh = self.rnn_dropout(x_bwh)

        y_seq = self.head(self.decoder_dropout(x_bwh))  # (B, W, 1)
        y = y_seq.squeeze(-1)                            # (B, W)

        return {"y_pred": y}

    @staticmethod
    def _xavier_init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mlp": MultiHeadMLPModel,
    "mlp_vae": MultiHeadVAEModel,
    "rnn": RNNSeq2SeqModel,
    "rnn_bi": RNNSeq2SeqModel,  # mismo modelo, pero cfg bidirectional=True en el suggest
}


def get_model_class(name: str) -> Type[nn.Module]:
    key = (name or "").strip().lower()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())