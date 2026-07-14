# suggests.py
from __future__ import annotations

from typing import Any, Dict, Callable, List
import optuna

from models import _ACTIVATIONS


# -----------------------------
# MLP (MLP_E_D)
# -----------------------------
def suggest_mlp(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "enc_num_layers": trial.suggest_int("enc_num_layers", 2, 3),
        "enc_hidden_size": trial.suggest_categorical("enc_hidden_size", [32, 64, 128, 256]),
        "enc_activation": trial.suggest_categorical("enc_activation", list(_ACTIVATIONS.keys())),
        "reg_num_hidden_layers": trial.suggest_int("reg_layers", 2, 3),
        "reg_hidden_size": trial.suggest_categorical("reg_size", [32, 64, 128, 256]),
        "reg_activation": trial.suggest_categorical("reg_act", list(_ACTIVATIONS.keys())),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0.2]),
        "lr": trial.suggest_float("lr", 1e-8, 1e-2, log=True),
        "weight_decay": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("opt", ["AdamW"]),
        "sliding_window": trial.suggest_int("window", 1, 25),
    }


# -----------------------------
# MLP + VAE (MLP_VaE_D)
# -----------------------------
def suggest_mlp_vae(trial: optuna.Trial) -> Dict[str, Any]:
    cfg = suggest_mlp(trial)
    cfg.update({
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128, 256]),
        "beta": trial.suggest_float("beta", 1e-6, 1e-2, log=True),
        "use_vae": True,  # explícito, evita activaciones accidentales
    })
    return cfg


# -----------------------------
# RNN (RNN_E_D)
# -----------------------------
def suggest_rnn(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "rnn_type": trial.suggest_categorical("rnn_type", ["GRU"]),
        "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [32, 64, 128, 256]),
        "rnn_num_layers": trial.suggest_int("rnn_num_layers", 2, 3),
        "bidirectional": trial.suggest_categorical("bidirectional", [False]),
        "rnn_activation": trial.suggest_categorical("rnn_activation", list(_ACTIVATIONS.keys())),
        "reg_num_hidden_layers": trial.suggest_int("reg_layers", 2, 3),
        "reg_hidden_size": trial.suggest_categorical("reg_size", [32, 64, 128, 256]),
        "reg_activation": trial.suggest_categorical("reg_act", list(_ACTIVATIONS.keys())),
        "dropout_rate": 0.2,
        "lr": trial.suggest_float("lr", 1e-8, 1e-2, log=True),
        "weight_decay": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("opt", ["AdamW"]),
        "sliding_window": trial.suggest_int("window", 1, 25),
    }


# -----------------------------
# RNN bidireccional (RNNBi_E_D)
# -----------------------------
def suggest_rnn_bi(trial: optuna.Trial) -> Dict[str, Any]:
    cfg = suggest_rnn(trial)
    cfg["bidirectional"] = True  # fija True
    return cfg


# -----------------------------
# Registry
# -----------------------------
SUGGEST_REGISTRY: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {
    "mlp": suggest_mlp,
    "mlp_vae": suggest_mlp_vae,
    "rnn": suggest_rnn,
    "rnn_bi": suggest_rnn_bi,
}


def get_suggest_fn(name: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
    key = (name or "").strip().lower()
    if key in SUGGEST_REGISTRY:
        return SUGGEST_REGISTRY[key]
    raise KeyError(f"Unknown suggest '{name}'. Available: {sorted(SUGGEST_REGISTRY)}")


def list_suggests() -> List[str]:
    return sorted(SUGGEST_REGISTRY.keys())