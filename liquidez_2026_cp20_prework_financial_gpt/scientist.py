# scientist.py
# ---------------------------------------------------------------------------
# Registry-based Scientist (models.py + suggests.py)
# Standard output contract: dict -> {"y_pred": Tensor, "losses": {...}, "extras": {...}}
# Backward-compatible: also accepts legacy Tensor or (y_pred, kl_loss) outputs
#
# IMPORTANT FIX:
# - In warm_up(), do NOT call _suggest(study.best_trial) because best_trial is a FrozenTrial.
#   Instead, rebuild cfg from best_trial.params via _cfg_from_params().
# ---------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from losses import (
    LearnableHuberLoss,
    log_cosh_loss,
    mae_loss,
    mse_loss,
    rmse_loss,
    smape_loss,
    wmape_loss,
)
from tools import Tools, ChecklistMixin

from models import get_model_class
from suggests import get_suggest_fn


# ---------------------------------------------------------------------------
# Static configuration values
# ---------------------------------------------------------------------------
class _Static:  # noqa: D401
    """Namespace for constants (no instantiation)."""

    SEED: int = 42
    EARLY_STOP_PATIENCE: int = 15
    LR_PATIENCE: int = 5


np.random.seed(_Static.SEED)
torch.manual_seed(_Static.SEED)


# ---------------------------------------------------------------------------
# Scientist class
# ---------------------------------------------------------------------------
class Scientist(ChecklistMixin):
    """Owns the full modelling life-cycle for multiple time-series."""

    device: torch.device
    best_params: Dict[str, Dict[str, Any]]
    models: Dict[str, nn.Module]

    # --------------------------- SET-UP ---------------------------------
    def __init__(
        self,
        *,
        device: Optional[str] = None,
        loss_type: str = "rmse",
        model_name: str = "mlp",
        suggest_name: str = "mlp",
    ) -> None:
        super().__init__()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tools = Tools()

        self.model_name = (model_name or "mlp").strip().lower()
        self.suggest_name = (suggest_name or "mlp").strip().lower()

        # Loss map
        self._loss_map = {
            "rmse": rmse_loss,
            "mae": mae_loss,
            "mse": mse_loss,
            "smape": smape_loss,
            "wmape": wmape_loss,
            "log_cosh": log_cosh_loss,
            "huber": LearnableHuberLoss(),
        }
        if loss_type not in self._loss_map:
            raise ValueError(f"[Scientist] Unknown loss_type '{loss_type}'. Supported: {list(self._loss_map)}")
        self.c_reg = self._loss_map[loss_type]

        self.best_params = {}
        self.models = {}

    @staticmethod
    def safe_series_name(series_name: str) -> str:
        """Return a filesystem-safe id while keeping the public serie name unchanged."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(series_name)).strip("._-")
        return safe or "serie"

    def model_filepath(self, model_dir: str | Path, series_name: str, stage: str) -> Path:
        return Path(model_dir) / f"{self.safe_series_name(series_name)}_{stage}.pt"

    # ---------------------------------------------------------------------
    # Output normalization (retrocompatible)
    # ---------------------------------------------------------------------
    @staticmethod
    def _as_output_dict(model_out: Any) -> Dict[str, Any]:
        """
        Normalize model output to dict contract.

        Accepted:
          - dict with "y_pred"
          - tuple: (y_pred, kl_loss)  -> {"y_pred": y_pred, "losses": {"kl": kl_loss}}
          - tensor -> {"y_pred": tensor}
        """
        if isinstance(model_out, dict):
            if "y_pred" not in model_out:
                raise ValueError("Model output dict must include key 'y_pred'.")
            return model_out

        if isinstance(model_out, tuple):
            if len(model_out) != 2:
                raise ValueError("Tuple output must be (y_pred, kl_loss).")
            y_pred, kl_loss = model_out
            return {"y_pred": y_pred, "losses": {"kl": kl_loss}}

        # plain tensor
        return {"y_pred": model_out}

    # ---------------------------------------------------------------------
    # Loss computation (dict contract)
    # ---------------------------------------------------------------------
    def _compute_loss(
        self,
        model: nn.Module,
        out: Dict[str, Any],
        y_true: torch.Tensor,
        *,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Unified loss:
          recon = loss(y_pred, y_true)
          if out["losses"]["kl"] exists:
              total = recon + beta * kl

        beta priority:
          1) cfg["beta"] (if present)
          2) getattr(model, "beta", 1.0)
          3) 1.0
        """
        y_pred = out["y_pred"]
        recon = self.c_reg(y_pred, y_true).mean()

        losses = out.get("losses") or {}
        if isinstance(losses, dict) and "kl" in losses:
            kl = losses["kl"]
            if cfg is not None and "beta" in cfg:
                beta = float(cfg["beta"])
            else:
                beta = float(getattr(model, "beta", 1.0))
            return recon + beta * kl

        return recon

    @staticmethod
    def _make_scheduler(optimizer: optim.Optimizer) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=_Static.LR_PATIENCE,
            factor=0.5,
            min_lr=1e-6,
        )

    # ---------------------------------------------------------------------
    # Suggest via registry
    # ---------------------------------------------------------------------
    def _suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generates cfg by calling suggests registry (Optuna suggest_*).

        NOTE: This must be called with a live optuna.Trial. Do NOT call it with FrozenTrial.
        """
        suggest_fn = get_suggest_fn(self.suggest_name)
        cfg = suggest_fn(trial)
        cfg.setdefault("_model_name", self.model_name)
        cfg.setdefault("_suggest_name", self.suggest_name)
        return cfg

    # ---------------------------------------------------------------------
    # Rebuild cfg from Optuna best_trial.params (FrozenTrial-safe)
    # ---------------------------------------------------------------------
    def _cfg_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Optuna params -> cfg expected by models.

        This is necessary because:
          - suggests.py uses param names like "reg_layers", "wd", "opt", "window"
          - cfg keys are "reg_num_hidden_layers", "weight_decay", "optimizer", "sliding_window"
        """
        s = self.suggest_name

        # Base defaults (safe)
        cfg: Dict[str, Any] = {}

        if s in ("mlp", "mlp_vae"):
            cfg = {
                "enc_num_layers": params["enc_num_layers"],
                "enc_hidden_size": params["enc_hidden_size"],
                "enc_activation": params["enc_activation"],
                "reg_num_hidden_layers": params["reg_layers"],
                "reg_hidden_size": params["reg_size"],
                "reg_activation": params["reg_act"],
                "dropout_rate": params.get("dropout_rate", 0.2),
                "lr": params["lr"],
                "weight_decay": params.get("wd", params.get("weight_decay", 0.0)),
                "optimizer": params.get("opt", params.get("optimizer", "AdamW")),
                "sliding_window": params.get("window", params.get("sliding_window")),
            }
            if cfg["sliding_window"] is None:
                raise KeyError("Missing 'window' (sliding_window) in Optuna params.")

            if s == "mlp_vae":
                cfg.update(
                    {
                        "latent_dim": params["latent_dim"],
                        "beta": params["beta"],
                        "use_vae": True,
                    }
                )
            return cfg

        if s in ("rnn", "rnn_bi"):
            cfg = {
                "rnn_type": params.get("rnn_type", "GRU"),
                "rnn_hidden_size": params["rnn_hidden_size"],
                "rnn_num_layers": params["rnn_num_layers"],
                "bidirectional": params.get("bidirectional", False),
                "rnn_activation": params["rnn_activation"],
                "reg_num_hidden_layers": params["reg_layers"],
                "reg_hidden_size": params["reg_size"],
                "reg_activation": params["reg_act"],
                # In your suggests, dropout_rate is constant 0.2 (not necessarily in params)
                "dropout_rate": params.get("dropout_rate", 0.2),
                "lr": params["lr"],
                "weight_decay": params.get("wd", params.get("weight_decay", 0.0)),
                "optimizer": params.get("opt", params.get("optimizer", "AdamW")),
                "sliding_window": params.get("window", params.get("sliding_window")),
            }
            if cfg["sliding_window"] is None:
                raise KeyError("Missing 'window' (sliding_window) in Optuna params.")

            if s == "rnn_bi":
                cfg["bidirectional"] = True

            return cfg

        raise KeyError(f"Unknown suggest_name='{self.suggest_name}'. Cannot rebuild cfg from params.")

    # ---------------------------------------------------------------------
    # Split with gap (no leakage)
    # ---------------------------------------------------------------------
    @staticmethod
    def _blocked_window_split(
        n: int,
        train: int,
        val: int,
        gap: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        start = 0
        while start + train + gap + val <= n:
            tr = np.arange(start, start + train)
            vl = np.arange(start + train + gap, start + train + gap + val)
            yield tr, vl
            start += train + gap + val

    @staticmethod
    def _safe_blocked_splits(n: int, train: int, val: int, gap: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = list(Scientist._blocked_window_split(n, train, val, gap))
        if splits:
            return splits
        if n <= 1:
            idx = np.arange(max(n, 1))
            return [(idx, idx)]
        cut = max(1, n - 1)
        return [(np.arange(0, cut), np.arange(cut, n))]

    # ---------------------------------------------------------------------
    # Supervised windows with Optuna-controlled autoregressive lags
    # ---------------------------------------------------------------------
    @staticmethod
    def _make_supervised_windows(
        X: np.ndarray,
        y: np.ndarray,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Build model windows with Optuna-controlled autoregressive lags.

        Stress-safe behavior:
        - non-finite X/y values are converted to zero in transformed space;
        - requested window is clamped to the maximum feasible window for the
          available rows, preventing sparse/short series from killing Optuna;
        - cfg["sliding_window"] is updated to the effective value so training,
          backtest, forecast and model input_dim stay aligned.
        """
        requested_window = int(cfg.get("sliding_window", 1))
        if requested_window < 1:
            requested_window = 1

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D (rows, features), got shape={X_arr.shape}.")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X/y length mismatch: len(X)={len(X_arr)} len(y)={len(y_arr)}.")
        if len(y_arr) < 2:
            raise ValueError(f"Need at least 2 rows to build supervised windows; got {len(y_arr)}.")

        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Need n - 2*w + 1 >= 1 after lag construction + sequence window.
        max_window = max(1, (len(y_arr) - 1) // 2)
        window = int(min(requested_window, max_window))
        cfg["_requested_sliding_window"] = requested_window
        cfg["sliding_window"] = window

        n = len(y_arr)
        lag_matrix = np.full((n, window), np.nan, dtype=np.float32)
        for lag in range(1, window + 1):
            lag_matrix[lag:, lag - 1] = y_arr[:-lag]

        X_aug_all = np.concatenate([X_arr, lag_matrix], axis=1)
        valid_rows = ~np.isnan(X_aug_all).any(axis=1)
        X_aug = X_aug_all[valid_rows].astype(np.float32)
        y_aug = y_arr[valid_rows].astype(np.float32)
        row_positions = np.flatnonzero(valid_rows)

        Xw, yw = Tools.sliding_window_transform(X_aug, y_aug, window)
        target_positions = row_positions[window - 1:] if len(row_positions) >= window else np.array([], dtype=int)

        if len(Xw) == 0:
            raise ValueError(
                f"Supervised windowing produced zero rows: n={n}, requested_window={requested_window}, effective_window={window}."
            )

        meta = {
            "window": window,
            "requested_window": requested_window,
            "n_exogenous_features": int(X_arr.shape[1]),
            "n_lag_features": window,
            "input_dim": int(X_arr.shape[1] + window),
            "target_offset": int(2 * window - 1),
            "target_positions": target_positions.astype(int),
        }
        return Xw.astype(np.float32), yw.astype(np.float32), meta

    def recursive_forecast_mc(
        self,
        series_name: str,
        X_history: np.ndarray,
        y_history: np.ndarray,
        X_future: np.ndarray,
        qt: Any,
        *,
        n_mc: int = 10_000,
    ) -> Dict[str, np.ndarray]:
        """Forecast future rows recursively when the model uses target lags.

        For t+1, lags come from the latest observed y_trans values. For later
        horizons, previous transformed predictions are fed back as lags. The
        returned arrays have one row per X_future row.
        """
        if series_name not in self.models:
            raise RuntimeError("[Scientist] fine_tune not executed.")

        cfg = self.best_params[series_name]["cfg"]
        window = int(cfg["sliding_window"])
        model = self.models[series_name].to(self.device)

        X_hist = np.asarray(X_history, dtype=np.float32)
        y_hist = np.asarray(y_history, dtype=np.float32).reshape(-1)
        X_fut = np.asarray(X_future, dtype=np.float32)

        if X_hist.ndim != 2 or X_fut.ndim != 2:
            raise ValueError("X_history and X_future must be 2-D arrays.")
        if len(X_hist) != len(y_hist):
            raise ValueError("X_history/y_history length mismatch.")
        if len(y_hist) < window:
            raise ValueError(
                f"Need at least {window} historical y values to build recursive lags; got {len(y_hist)}."
            )

        history = list(y_hist.astype(np.float32))
        aug_buffer: List[np.ndarray] = []

        # Historical augmented rows provide the sequence context for the first
        # forecast date. Row t uses y[t-1]...y[t-window].
        for pos in range(window, len(y_hist)):
            lag_values = np.array([history[pos - lag] for lag in range(1, window + 1)], dtype=np.float32)
            aug_buffer.append(np.concatenate([X_hist[pos], lag_values]).astype(np.float32))

        pred_t_all: List[float] = []
        low_t_all: List[float] = []
        up_t_all: List[float] = []
        var_t_all: List[float] = []

        for x_row in X_fut:
            lag_values = np.array([history[-lag] for lag in range(1, window + 1)], dtype=np.float32)
            aug_row = np.concatenate([x_row.astype(np.float32), lag_values]).astype(np.float32)
            aug_buffer.append(aug_row)

            if len(aug_buffer) < window:
                raise ValueError(
                    f"Not enough augmented history to build a window={window} forecast. "
                    f"Available rows: {len(aug_buffer)}."
                )

            X_window = np.asarray(aug_buffer[-window:], dtype=np.float32)[None, :, :]
            X_t = torch.tensor(X_window, dtype=torch.float32).to(self.device)
            mc_r = self._mc_simulate(model, X_t, n_mc)[:, 0]

            mean_t = float(mc_r.mean())
            pred_t_all.append(mean_t)
            low_t_all.append(float(np.percentile(mc_r, 2.5)))
            up_t_all.append(float(np.percentile(mc_r, 97.5)))
            var_t_all.append(float(mc_r.var(ddof=0)))

            # Recursive step: feed the transformed mean prediction into later lags.
            history.append(np.float32(mean_t))

        pred_t = np.asarray(pred_t_all, dtype=np.float32)
        low_t = np.asarray(low_t_all, dtype=np.float32)
        up_t = np.asarray(up_t_all, dtype=np.float32)

        pred_o = Tools.invert_transform(pred_t, qt)
        low_o = Tools.invert_transform(low_t, qt)
        up_o = Tools.invert_transform(up_t, qt)

        pred_o = np.nan_to_num(pred_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        low_o = np.nan_to_num(low_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        up_o = np.nan_to_num(up_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

        return {
            "pred_orig": pred_o,
            "lower_ci": low_o,
            "upper_ci": up_o,
            "pred_trans": pred_t,
            "var_pred": np.asarray(var_t_all, dtype=np.float32),
        }

    # ---------------------------------------------------------------------
    # Optuna objective
    # ---------------------------------------------------------------------
    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_epochs: int,
        batch: int,
    ) -> float:
        cfg = self._suggest(trial)
        Xw, yw, _ = self._make_supervised_windows(X, y, cfg)
        gap = int(cfg["sliding_window"]) - 1
        n = len(Xw)
        if n == 0:
            raise ValueError("Sliding window produced zero rows; check window size vs data length.")

        n_folds = 10
        val_win = max(1, n // (n_folds + 1))
        train_win = max(1, n - val_win * n_folds)

        ModelCls = get_model_class(self.model_name)

        losses: List[float] = []
        step_ctr = 0

        for tr_idx, vl_idx in self._safe_blocked_splits(n, train_win, val_win, gap):
            model = ModelCls(cfg, Xw.shape[2]).to(self.device)
            optm = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            sched = self._make_scheduler(optm)

            tr_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xw[tr_idx], dtype=torch.float32),
                    torch.tensor(yw[tr_idx], dtype=torch.float32),
                ),
                batch_size=batch,
                shuffle=False,
            )
            vl_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xw[vl_idx], dtype=torch.float32),
                    torch.tensor(yw[vl_idx], dtype=torch.float32),
                ),
                batch_size=batch,
                shuffle=False,
            )

            best_fold = float("inf")
            for _ in range(max_epochs):
                self._run_epoch(model, tr_loader, optm=optm, cfg=cfg)
                v = self._run_epoch(model, vl_loader, cfg=cfg)
                sched.step(v)
                trial.report(v, step_ctr)
                step_ctr += 1
                best_fold = min(best_fold, v)

            losses.append(best_fold)

        return float(np.mean(losses))

    # ---------------------------------------------------------------------
    # Train / validation epoch helper
    # ---------------------------------------------------------------------
    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        *,
        optm: Optional[optim.Optimizer] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> float:
        training = optm is not None
        model.train() if training else model.eval()

        total, n_samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            if training:
                optm.zero_grad()

            raw_out = model(xb)
            out = self._as_output_dict(raw_out)
            loss = self._compute_loss(model, out, yb, cfg=cfg)

            if training:
                loss.backward()
                optm.step()

            total += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        return total / n_samples if n_samples else 0.0

    # ---------------------------------------------------------------------
    # Warm-up (HPO)
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Warm-up (Optuna)")
    def warm_up(
        self,
        series_name: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_trials: int = 20,
        max_epochs: int = 30,
        batch: int = 32,
        model_dir: str | Path = "outputs",
    ) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=_Static.SEED),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_epochs, reduction_factor=3),
        )
        study.optimize(
            lambda t: self._objective(t, X, y, max_epochs=max_epochs, batch=batch),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # ✅ IMPORTANT: FrozenTrial-safe cfg rebuild
        cfg = self._cfg_from_params(study.best_trial.params)
        cfg.setdefault("_model_name", self.model_name)
        cfg.setdefault("_suggest_name", self.suggest_name)

        Xw, yw, meta = self._make_supervised_windows(X, y, cfg)
        if len(Xw) == 0:
            raise ValueError("Supervised windows produced zero rows; check window size vs data length.")

        self.best_params[series_name] = {
            "cfg": cfg,
            "params": study.best_trial.params,
            "input_dim": int(Xw.shape[2]),
            "model_name": self.model_name,
            "suggest_name": self.suggest_name,
            "window_meta": {k: v for k, v in meta.items() if k != "target_positions"},
        }

        ModelCls = get_model_class(self.model_name)
        model = ModelCls(cfg, Xw.shape[2]).to(self.device)

        optm = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = self._make_scheduler(optm)

        loader = DataLoader(
            TensorDataset(torch.tensor(Xw, dtype=torch.float32), torch.tensor(yw, dtype=torch.float32)),
            batch_size=batch,
            shuffle=True,
        )

        best, bad, best_state = float("inf"), 0, None
        for _ in range(max_epochs):
            loss = self._run_epoch(model, loader, optm=optm, cfg=cfg)
            sched.step(loss)

            if loss < best - 1e-4:
                best = loss
                bad = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= _Static.EARLY_STOP_PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError("[Scientist] warm_up failed: best_state is None.")

        model.load_state_dict(best_state)
        model.eval()
        self.models[series_name] = model

        self.save_model(series_name, self.model_filepath(model_dir, series_name, "wu"))

    # ---------------------------------------------------------------------
    # Fine-tune
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Fine-tune")
    def fine_tune(
        self,
        series_name: str,
        X: np.ndarray,
        y: np.ndarray,
        qt: Any,
        *,
        epochs: int = 50,
        batch: int = 128,
        model_dir: str | Path = "outputs",
    ) -> None:
        if series_name not in self.best_params:
            raise RuntimeError("[Scientist] warm_up must be executed first.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        input_dim = int(self.best_params[series_name]["input_dim"])
        wu_path = self.model_filepath(model_dir, series_name, "wu")
        self.load_model(series_name, input_dim, wu_path)

        cfg = self.best_params[series_name]["cfg"]
        Xw, yw, _ = self._make_supervised_windows(X, y, cfg)
        gap = int(cfg["sliding_window"]) - 1
        n = len(Xw)
        if n == 0:
            raise ValueError("Sliding window produced zero rows; check window size vs data length.")

        n_folds = 100
        val_win = max(1, n // (n_folds + 1))
        train_win = max(1, n - val_win * n_folds)

        model = self.models[series_name]
        model.to(self.device)

        optm = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = self._make_scheduler(optm)

        best, bad, best_state = float("inf"), 0, None

        for ep in range(epochs):
            for tr_idx, _ in self._safe_blocked_splits(n, train_win, val_win, gap):
                tr_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(Xw[tr_idx], dtype=torch.float32),
                        torch.tensor(yw[tr_idx], dtype=torch.float32),
                    ),
                    batch_size=batch,
                    shuffle=False,
                )
                self._run_epoch(model, tr_loader, optm=optm, cfg=cfg)

            v_losses: List[float] = []
            for _, vl_idx in self._safe_blocked_splits(n, train_win, val_win, gap):
                vl_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(Xw[vl_idx], dtype=torch.float32),
                        torch.tensor(yw[vl_idx], dtype=torch.float32),
                    ),
                    batch_size=batch,
                    shuffle=False,
                )
                v_losses.append(self._run_epoch(model, vl_loader, cfg=cfg))

            v = float(np.mean(v_losses)) if v_losses else best
            sched.step(v)

            if v < best - 1e-4:
                best = v
                bad = 0
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= _Static.EARLY_STOP_PATIENCE:
                    print(f"⏹️ Early-stop ep{ep+1} val={v:.4f}")
                    break

        if best_state is None:
            raise RuntimeError("[Scientist] fine_tune failed: best_state is None.")

        model.load_state_dict(best_state)
        model.eval()
        self.models[series_name] = model

        self.save_model(series_name, self.model_filepath(model_dir, series_name, "ft"))

        with torch.no_grad():
            raw_out = model(torch.tensor(Xw, dtype=torch.float32, device=self.device))
            out = self._as_output_dict(raw_out)
            preds_t = out["y_pred"][:, -1].detach().cpu().numpy()
        actual_t = yw[:, -1]

        print(
            f"[Fine-tune {series_name}] metrics:",
            self.tools.metrics_regression(
                Tools.invert_transform(actual_t, qt),
                Tools.invert_transform(preds_t, qt),
            ),
        )

    # ---------------------------------------------------------------------
    # Model persistence
    # ---------------------------------------------------------------------
    def save_model(self, series_name: str, filepath: Path) -> None:
        if series_name not in self.models:
            raise RuntimeError(f"[Scientist] No model trained for '{series_name}'.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.models[series_name].state_dict(), filepath)
        print(f"[Scientist] Saved '{series_name}' model to {filepath}")

    def load_model(self, series_name: str, input_dim: int, filepath: Path) -> None:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(filepath)
        if series_name not in self.best_params:
            raise RuntimeError(f"[Scientist] best_params for '{series_name}' not available.")

        cfg = self.best_params[series_name]["cfg"]
        model_name = (self.best_params[series_name].get("model_name") or self.model_name)
        ModelCls = get_model_class(model_name)

        model = ModelCls(cfg, input_dim).to(self.device)
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        model.eval()
        self.models[series_name] = model
        print(f"[Scientist] Loaded '{series_name}' model from {filepath}")

    # ---------------------------------------------------------------------
    # Monte-Carlo helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _activate_dropout(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            else:
                m.eval()

    def _mc_simulate(self, model: nn.Module, X_t: torch.Tensor, n_mc: int) -> np.ndarray:
        sims = []
        model.eval()
        self._activate_dropout(model)

        with torch.no_grad():
            for _ in range(n_mc):
                raw_out = model(X_t)
                out = self._as_output_dict(raw_out)
                y_pred = out["y_pred"]
                sims.append(y_pred[:, -1].detach().cpu().numpy())

        return np.stack(sims, axis=0)

    # ---------------------------------------------------------------------
    # Back-test MC-Dropout
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Back-test MC-Dropout")
    def backtest_mc(
        self,
        series_name: str,
        X_full: np.ndarray,
        y_trans: np.ndarray,
        qt: Any,
        *,
        n_mc: int = 10_000,
        train_split: float = 0.9,
    ) -> Dict[str, Any]:
        if series_name not in self.models:
            raise RuntimeError("[Scientist] fine_tune must be executed before backtest_mc().")

        cfg = self.best_params[series_name]["cfg"]
        win = int(cfg["sliding_window"])
        model = self.models[series_name].to(self.device)

        Xw, yw_seq, meta = self._make_supervised_windows(X_full, y_trans, cfg)
        if len(Xw) == 0:
            raise ValueError("Supervised windows returned zero rows.")
        y_last = yw_seq[:, -1]
        target_positions = np.asarray(meta["target_positions"], dtype=int)

        X_t = torch.tensor(Xw, dtype=torch.float32).to(self.device)
        mc_r = self._mc_simulate(model, X_t, n_mc)

        mean_t = mc_r.mean(axis=0)
        var_t = mc_r.var(axis=0, ddof=0)
        low_t = np.percentile(mc_r, 2.5, axis=0)
        up_t = np.percentile(mc_r, 97.5, axis=0)

        mean_o = Tools.invert_transform(mean_t, qt)
        low_o = Tools.invert_transform(low_t, qt)
        up_o = Tools.invert_transform(up_t, qt)
        act_o = Tools.invert_transform(y_last, qt)

        idx = np.arange(len(Xw))
        split = int(len(Xw) * train_split)

        df_r = pd.DataFrame(
            {
                "idx": idx,
                "target_pos": target_positions,
                "isTrain": idx < split,
                "actual_orig": act_o,
                "pred_orig": mean_o,
                "lower_ci": low_o,
                "upper_ci": up_o,
                "bias2": (mean_o - act_o) ** 2,
                "var_pred": var_t,
            }
        )
        df_r["serie"] = series_name

        mask_test = ~df_r["isTrain"]
        met_r = self.tools.metrics_regression(
            df_r.loc[mask_test, "actual_orig"].values,
            df_r.loc[mask_test, "pred_orig"].values,
        )
        met_r["Bias2_mean"] = float(df_r.loc[mask_test, "bias2"].mean())
        met_r["Var_mean"] = float(df_r.loc[mask_test, "var_pred"].mean())

        return {
            "df_regression": df_r,
            "df_regression_metrics": pd.DataFrame([met_r]),
        }

    # ---------------------------------------------------------------------
    # Forecast MC-Dropout
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Forecast MC-Dropout")
    def forecast_mc(
        self,
        series_name: str,
        X_windows: np.ndarray,
        qt: Any,
        *,
        n_mc: int = 10_000,
    ) -> Dict[str, np.ndarray]:
        if series_name not in self.models:
            raise RuntimeError("[Scientist] fine_tune not executed.")

        model = self.models[series_name].to(self.device)

        X_t = torch.tensor(X_windows, dtype=torch.float32).to(self.device)
        mc_r = self._mc_simulate(model, X_t, n_mc)

        pred_t = mc_r.mean(axis=0)
        low_t = np.percentile(mc_r, 2.5, axis=0)
        up_t = np.percentile(mc_r, 97.5, axis=0)

        pred_o = Tools.invert_transform(pred_t, qt)
        low_o = Tools.invert_transform(low_t, qt)
        up_o = Tools.invert_transform(up_t, qt)

        pred_o = np.nan_to_num(pred_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        low_o = np.nan_to_num(low_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        up_o = np.nan_to_num(up_o, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

        return {
            "pred_orig": pred_o,
            "lower_ci": low_o,
            "upper_ci": up_o,
            "pred_trans": pred_t,
        }